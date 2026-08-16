#!/usr/bin/env python3
"""ViTPose detection node: image -> (bbox provider) -> joint model -> VitposeResult.

Thin producer: all judgment (confidence gates, pose solving, mask
consumption) lives downstream in vitpose_process_node. Contract:
gate_tetra_overview.md; decisions + evidence: auv_vision/VITPOSE_PLAN.md.

Bbox providers: `full_frame`; `model` (in-process objectness — no fallback,
no fire = no result; retries at `search_rate` while absent; an optional
`tracker:` block makes it a seed and CropTracker carries the crop); `topic`
(external Detection2DArray). A config with no `model:` section is
detect-only: objectness boxes only, empty detections[] as heartbeat.

The node starts cold, holding just a config name. ~enable(true) loads
checkpoints synchronously so its response reports success; ~set_config takes
an object name or YAML path and, once loaded, swaps on a worker thread while
the old model keeps serving. ~enabled (param, default false; Bool status at
1 Hz) means "actually running".

VitposeNodeBase — the ~enable/~set_config/~enabled shell — is shared with
sim_vitpose_node, which is why torch is imported lazily here.
"""

import os

# Must run before torch import (PyTorch ApproximateClock workaround,
# https://github.com/pytorch/pytorch/issues/91516).
os.environ.setdefault("KINETO_DISABLED", "1")

import sys
import threading

import numpy as np
import rospy

from cv_bridge import CvBridge

import cv2

from auv_msgs.msg import Keypoint, VitposeResult
from auv_msgs.srv import SetString, SetStringResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.vitpose_utils import (  # noqa: E402
    CropTracker,
    RateGate,
    apply_camera,
    is_detect_only,
    load_object_config,
    model_kwargs,
    resolve_checkpoint_path,
    validate_detect_only,
)


def detection_array(bbox_xywh, score, class_id, header) -> Detection2DArray:
    """One-detection Detection2DArray; bbox None = the empty heartbeat that
    tells "looking, don't see it" from "node is dead". Shared with sim."""
    message = Detection2DArray()
    message.header = header
    if bbox_xywh is not None:
        detection = Detection2D()
        detection.header = header
        box = BoundingBox2D()
        box.center.x = bbox_xywh[0] + bbox_xywh[2] * 0.5
        box.center.y = bbox_xywh[1] + bbox_xywh[3] * 0.5
        box.size_x = float(bbox_xywh[2])
        box.size_y = float(bbox_xywh[3])
        detection.bbox = box
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = int(class_id)
        hypothesis.score = float(score)
        detection.results.append(hypothesis)
        message.detections.append(detection)
    return message


# ─────────────────────────────────────────────── bbox providers
#
# Interface: get_bboxes(img_rgb, header) -> list of (x, y, w, h) in source px.
# Images arrive RGB (the model contract's colour order).


class FullFrameProvider:
    """One bbox covering the whole image."""

    def __init__(self, params):
        pass

    def get_bboxes(self, img_rgb, header):
        h, w = img_rgb.shape[:2]
        return [(0.0, 0.0, float(w), float(h))]

    def shutdown(self):
        pass


class ModelProvider:
    """In-process objectness detector as the crop source.

    Publishes the box actually fed to the joint model on `publish_topic`
    (empty detections[] when nothing fires; the pose path never reads it
    back). With a `tracker:` block the detector only seeds and CropTracker
    (vitpose_utils SECTION 4 — rationale and measurements) carries the crop.

    params: checkpoint, device, threshold (detect, 0.5), measure_threshold
    (box EXTENT only, 0.7; see decode_box), search_rate (Hz cap on retries
    while the object is absent, 5.0; null/0 = every frame), publish_topic,
    class_id, tracker (CropTracker kwargs).
    """

    def __init__(self, params):
        from utils.vitpose_inference import load_objectness  # torch: lazy

        checkpoint = resolve_checkpoint_path(params["checkpoint"])
        self._detector = load_objectness(
            checkpoint,
            device=params.get("device", "cuda"),
            threshold=float(params.get("threshold", 0.5)),
            measure_threshold=(
                None
                if params.get("measure_threshold", 0.7) is None
                else float(params.get("measure_threshold", 0.7))
            ),
        )
        self._class_id = int(params.get("class_id", 0))
        topic = params.get("publish_topic")
        self._pub = (
            rospy.Publisher(topic, Detection2DArray, queue_size=1) if topic else None
        )
        tracker_cfg = params.get("tracker")
        self._tracker = CropTracker(**tracker_cfg) if tracker_cfg else None
        self._score = 0.0  # most recent detector score, carried for telemetry
        search_rate = params.get("search_rate", 5.0)
        self._search_period = 1.0 / float(search_rate) if search_rate else 0.0
        self._next_search = 0.0  # earliest detector attempt while absent

    def _detect(self, img_rgb, now):
        """Run the detector; a miss holds off retries for one search period."""
        bbox, self._score = self._detector.predict(img_rgb)
        self._next_search = now + self._search_period if bbox is None else 0.0
        return bbox

    def get_bboxes(self, img_rgb, header):
        now = header.stamp.to_sec() or rospy.get_time()
        throttled = now < self._next_search  # absent-state search cap
        if self._tracker is None:
            bbox = None if throttled else self._detect(img_rgb, now)
        else:
            if self._tracker.needs_seed(now) and not throttled:
                self._tracker.seed(self._detect(img_rgb, now), now)
            bbox = self._tracker.box

        if self._pub is not None:
            self._pub.publish(
                detection_array(bbox, self._score, self._class_id, header)
            )
        if bbox is None:
            rospy.logdebug_throttle(
                5.0, f"objectness: nothing above threshold (best {self._score:.2f})"
            )
            return []
        return [tuple(float(v) for v in bbox)]

    def feedback(self, kps, scores, mask_probs, mask_threshold, image_shape):
        """Post-inference hook: propagate the crop and health-check it."""
        if self._tracker is None:
            return
        before = self._tracker.box
        self._tracker.update(kps, scores, mask_probs, mask_threshold, image_shape)
        if self._tracker.box is None and before is not None:
            rospy.logdebug_throttle(
                2.0, f"crop dropped, re-seeding: {self._tracker.last_reason}"
            )

    def shutdown(self):
        if self._pub is not None:
            self._pub.unregister()


class TopicProvider:
    """Bboxes from an external Detection2DArray topic.

    params: topic, class_id (optional top-hypothesis filter), max_age
    (staleness gate, seconds).
    """

    def __init__(self, params):
        self._class_id = params.get("class_id")
        self._max_age = float(params.get("max_age", 0.5))
        self._lock = threading.Lock()
        self._latest = None
        self._sub = rospy.Subscriber(
            params["topic"], Detection2DArray, self._cb, queue_size=1
        )

    def _cb(self, msg):
        with self._lock:
            self._latest = msg

    def get_bboxes(self, img_rgb, header):
        with self._lock:
            msg = self._latest
        if msg is None:
            return []
        if abs((header.stamp - msg.header.stamp).to_sec()) > self._max_age:
            return []
        bboxes = []
        for det in msg.detections:
            if self._class_id is not None:
                if not det.results or det.results[0].id != self._class_id:
                    continue
            cx, cy = det.bbox.center.x, det.bbox.center.y
            w, h = det.bbox.size_x, det.bbox.size_y
            bboxes.append((cx - w * 0.5, cy - h * 0.5, w, h))
        return bboxes

    def shutdown(self):
        self._sub.unregister()


_BBOX_PROVIDERS = {
    "full_frame": FullFrameProvider,
    "model": ModelProvider,
    "topic": TopicProvider,
}


# ─────────────────────────────────────────────── active pipeline (swappable)


class _Pipeline:
    """Everything derived from one object config: model, provider, topics.

    Built off the callback thread, installed atomically, torn down as a unit
    — a swap can never mix the old model with the new config. The image
    subscriber passes THIS pipeline to the callback, which checks it is still
    the installed one, so a frame can never run through the wrong config's
    model even mid-swap.

    Detect-only (`model:` absent): no joint model, no VitposeResult — the
    only output is the provider's Detection2DArray, and `detection.rate` is
    the cost control (every processed frame is a full objectness pass).
    """

    def __init__(self, config, image_cb):
        self.config = config
        self.object_name = config["object"]
        detection_cfg = config["detection"]
        self.detect_only = is_detect_only(config)

        provider_cfg = dict(
            detection_cfg.get("bbox_provider") or {"type": "full_frame"}
        )

        if self.detect_only:
            self.model = None
            self.keypoint_names = []
            self.mask_classes = []
            validate_detect_only(self.object_name, provider_cfg)
        else:
            from utils.vitpose_inference import load_vitpose  # torch: lazy

            model_cfg = config["model"]
            checkpoint = resolve_checkpoint_path(model_cfg["checkpoint"])
            self.model = load_vitpose(checkpoint, **model_kwargs(config))

            self.keypoint_names = list(
                model_cfg.get("keypoint_names")
                or [f"kp_{i}" for i in range(self.model.num_kps)]
            )
            self.mask_classes = list(
                model_cfg.get("mask_classes")
                or [f"mask_{i}" for i in range(self.model.num_masks)]
            )
            if len(self.keypoint_names) != self.model.num_kps:
                raise ValueError(
                    f"{self.object_name}: {len(self.keypoint_names)} keypoint_names "
                    f"but checkpoint has K={self.model.num_kps}"
                )
            if len(self.mask_classes) != self.model.num_masks:
                raise ValueError(
                    f"{self.object_name}: {len(self.mask_classes)} mask_classes "
                    f"but checkpoint has C={self.model.num_masks}"
                )

        provider_type = provider_cfg.pop("type", "full_frame")
        if provider_type not in _BBOX_PROVIDERS:
            raise ValueError(
                f"unknown bbox_provider '{provider_type}' "
                f"(available: {sorted(_BBOX_PROVIDERS)})"
            )
        self.provider = _BBOX_PROVIDERS[provider_type](provider_cfg)

        self._rate = RateGate(detection_cfg.get("rate"))

        self.result_pub = (
            None
            if self.detect_only
            else rospy.Publisher(
                detection_cfg["result_topic"], VitposeResult, queue_size=1
            )
        )
        self.image_sub = rospy.Subscriber(
            detection_cfg["image_topic"],
            Image,
            lambda msg, pipeline=self: image_cb(msg, pipeline),
            queue_size=1,
            buff_size=2**24,
        )

    def due(self, header) -> bool:
        """Rate gate, checked before the image is even decoded."""
        return self._rate.due(header.stamp.to_sec() or rospy.get_time())

    def shutdown(self):
        self.image_sub.unregister()
        self.provider.shutdown()
        if self.result_pub is not None:
            self.result_pub.unregister()


# ─────────────────────────────────────────────── node shell (shared with sim)


class VitposeNodeBase:
    """The ~enable / ~set_config / ~enabled shell, shared with
    sim_vitpose_node so the two interfaces cannot drift. Subclasses provide:

        log_name                  log-line prefix
        _build_pipeline(name)     construct the pipeline flavour
        _swap_config(name)        ~set_config once loaded; called HOLDING
                                  _pipeline_lock
        _image_cb(msg, pipeline)  per-frame work; must start with
                                  _pipeline_active(pipeline)

    _pipeline_lock guards the installed-pipeline reference (held briefly);
    _load_lock serializes slow builds so concurrent ~enable calls cannot each
    build a pipeline and leak the loser's live subscribers.
    """

    log_name = "vitpose_node"

    def _init_shell(self):
        """Call at the END of the subclass __init__: registers services, then
        honours ~enabled (which may synchronously load the pipeline)."""
        self.bridge = CvBridge()
        self.enabled = False
        self._pipeline_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._config_name = rospy.get_param("~config", "gate")
        self._pipeline = None

        rospy.Service("~enable", SetBool, self._handle_enable)
        rospy.Service("~set_config", SetString, self._handle_set_config)
        self._enabled_pub = rospy.Publisher("~enabled", Bool, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(1.0), self._publish_enabled)

        if bool(rospy.get_param("~enabled", False)):
            try:
                self._ensure_pipeline()
                self.enabled = True
            except Exception as exc:
                rospy.logerr(
                    f"{self.log_name}: ~enabled was true but loading "
                    f"'{self._config_name}' failed: {exc}. Staying up and "
                    "DISABLED — fix it and call ~enable."
                )
        self._publish_enabled(None)

        rospy.loginfo(
            f"{self.log_name} ready as '{rospy.get_name()}' "
            f"(config={self._config_name}, enabled={self.enabled}, "
            f"loaded={self._pipeline is not None})"
        )

    # ------------------------------------------------------------- hooks

    def _build_pipeline(self, name_or_path):
        raise NotImplementedError

    def _swap_config(self, name_or_path):
        raise NotImplementedError

    # ------------------------------------------------------------- pipeline

    def _pipeline_active(self, pipeline) -> bool:
        """True iff enabled AND `pipeline` is the installed one. Subscribers
        exist before installation and until shutdown; this drops frames from
        both windows."""
        if not self.enabled:
            return False
        with self._pipeline_lock:
            return pipeline is self._pipeline

    def _ensure_pipeline(self):
        """Load the configured pipeline if not up yet; raises on failure.
        Synchronous on purpose: nothing is served yet, and the ~enable caller
        gets a truthful success/failure."""
        with self._load_lock:
            with self._pipeline_lock:
                if self._pipeline is not None:
                    return
            pipeline = self._build_pipeline(self._config_name)
            with self._pipeline_lock:
                self._pipeline = pipeline
            rospy.loginfo(
                f"{self.log_name} loaded '{pipeline.object_name}' "
                f"({'detect-only' if pipeline.detect_only else 'pose'})"
            )

    # ------------------------------------------------------------- services

    def _publish_enabled(self, _event):
        self._enabled_pub.publish(Bool(data=self.enabled))

    def _handle_enable(self, req):
        if not bool(req.data):
            self.enabled = False
            self._publish_enabled(None)
            # Pipeline stays loaded: re-enable is instant.
            return SetBoolResponse(success=True, message="disabled")
        try:
            self._ensure_pipeline()
        except Exception as exc:
            self.enabled = False
            self._publish_enabled(None)
            message = f"failed to load '{self._config_name}': {exc}"
            rospy.logerr(message)
            return SetBoolResponse(success=False, message=message)

        self.enabled = True
        self._publish_enabled(None)
        message = f"enabled (object={self._pipeline.object_name})"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _handle_set_config(self, req):
        with self._pipeline_lock:
            if self._pipeline is None:
                # Cold: just record the name; loading waits for ~enable.
                self._config_name = req.data
                return SetStringResponse(
                    success=True,
                    message=f"config set to '{req.data}' (loads on ~enable)",
                )

        # Do not hold _pipeline_lock while swapping. Pipeline shutdown may wait
        # for an image callback, and callbacks use this same lock to check that
        # their pipeline is still active.
        return self._swap_config(req.data)

    def run(self):
        rospy.spin()


# ─────────────────────────────────────────────── node


class VitposeDetectionNode(VitposeNodeBase):
    log_name = "vitpose_detection_node"

    def __init__(self):
        rospy.init_node("vitpose_detection_node")
        self._swap_thread = None
        self._init_shell()

    # ------------------------------------------------------------- pipeline

    def _build_pipeline(self, name_or_path):
        ns = rospy.get_namespace().strip("/") or "taluy"
        config = load_object_config(name_or_path, ns)
        camera = rospy.get_param("~camera", "")
        if camera:
            apply_camera(config, camera, ns)
            rospy.logwarn(
                f"~camera override: '{config['object']}' running on cam_{camera}"
            )
        return _Pipeline(config, self._image_cb)

    def _swap_config(self, name_or_path):
        """Load on a worker thread so the old model serves until the swap."""
        with self._pipeline_lock:
            if self._swap_thread is not None and self._swap_thread.is_alive():
                return SetStringResponse(
                    success=False, message="a config swap is already in progress"
                )
            self._swap_thread = threading.Thread(
                target=self._swap_worker, args=(name_or_path,), daemon=True
            )
            self._swap_thread.start()
        return SetStringResponse(
            success=True,
            message=f"loading '{name_or_path}' in the background "
            "(old config keeps serving until the swap)",
        )

    def _swap_worker(self, name_or_path):
        try:
            with self._load_lock:
                new_pipeline = self._build_pipeline(name_or_path)
        except Exception as exc:
            rospy.logerr(f"set_config('{name_or_path}') failed: {exc}")
            return
        with self._pipeline_lock:
            old, self._pipeline = self._pipeline, new_pipeline
            self._config_name = name_or_path
        old.shutdown()
        rospy.loginfo(
            f"vitpose_detection_node switched to object "
            f"'{new_pipeline.object_name}' ({new_pipeline.config['_path']})"
        )

    # ------------------------------------------------------------- inference

    def _image_cb(self, msg, pipeline):
        if not self._pipeline_active(pipeline):
            return
        # Rate gate first: dropping a frame must not cost an image decode.
        if not pipeline.due(msg.header):
            return
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logerr_throttle(5.0, f"image decode failed: {exc}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_bgr.shape[:2]
        full_frame = (0.0, 0.0, float(w_img), float(h_img))

        bboxes = pipeline.provider.get_bboxes(img_rgb, msg.header)
        if pipeline.detect_only:
            return  # the provider already published its Detection2DArray

        for bbox in bboxes:
            kps, scores, mask_probs = pipeline.model.predict(img_rgb, bbox)
            result = VitposeResult()
            result.header = msg.header
            result.object = pipeline.object_name
            result.bbox = [] if bbox == full_frame else [float(v) for v in bbox]
            result.keypoint_names = pipeline.keypoint_names
            result.mask_classes = pipeline.mask_classes
            result.mask_threshold = pipeline.model.mask_threshold
            for i in range(len(kps)):
                kp = Keypoint()
                kp.id = i
                kp.x = float(kps[i, 0])
                kp.y = float(kps[i, 1])
                kp.confidence = float(scores[i, 0])
                result.keypoints.append(kp)
            if mask_probs is not None:
                for plane in mask_probs:
                    mono = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
                    image = self.bridge.cv2_to_imgmsg(mono, encoding="mono8")
                    image.header = msg.header
                    result.masks.append(image)
            pipeline.result_pub.publish(result)

            # Opt-in provider hook: the crop tracker propagates and
            # health-checks its box from the model's own output here.
            feedback = getattr(pipeline.provider, "feedback", None)
            if feedback is not None:
                try:
                    feedback(
                        kps,
                        scores[:, 0],
                        mask_probs,
                        pipeline.model.mask_threshold,
                        img_rgb.shape,
                    )
                except Exception as exc:
                    rospy.logerr_throttle(5.0, f"bbox provider feedback raised: {exc}")


if __name__ == "__main__":
    try:
        VitposeDetectionNode().run()
    except rospy.ROSInterruptException:
        pass
