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

The node starts cold, holding just a config name, and runs in one of three
modes (~set_mode, SetString): "off" (nothing loaded), "detect" (objectness
boxes only — the joint model is not on the GPU; the frame gate falls back to
`detection.detect_rate`, default `search_rate`, because there is no tracker
to amortize the detector), "pose" (full pipeline). Loaded == enabled: a mode
change loads/frees models synchronously so the response is truthful, and
"off" returns the GPU memory (minus torch's CUDA context, which lives until
the process exits). ~enable (SetBool) is the two-state alias: true = the
fullest mode the config supports, false = off. ~set_config takes an object
name or YAML path and, once loaded, swaps on a worker thread while the old
model keeps serving. Status: ~enabled (Bool) + ~mode (String), latched, 1 Hz.

VitposeNodeBase — the ~enable/~set_mode/~set_config shell — is shared with
sim_vitpose_node, which is why torch is imported lazily here.
"""

import os

# Must run before torch import (PyTorch ApproximateClock workaround,
# https://github.com/pytorch/pytorch/issues/91516).
os.environ.setdefault("KINETO_DISABLED", "1")

import gc
import sys
import threading
import time

import numpy as np
import rospy

from cv_bridge import CvBridge

import cv2

from auv_msgs.msg import Keypoint, VitposeResult
from auv_msgs.srv import SetString, SetStringResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
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
    runtime_detect_rate,
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
    class_id, tracker (CropTracker kwargs), detect_columns ([x0, x1] source
    px, or {camera: [x0, x1]}: the DETECTOR only sees these columns — hides
    a fisheye housing's side arcs it otherwise locks onto; boxes come back
    in full-frame px, the pose model and tracker still see the whole frame).
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
        cols = params.get("detect_columns")
        if isinstance(cols, dict):
            cols = cols.get(params.get("_camera"))
        self._detect_columns = (int(cols[0]), int(cols[1])) if cols else None

    def _detect(self, img_rgb, now):
        """Run the detector; a miss holds off retries for one search period."""
        if self._detect_columns is None:
            bbox, self._score = self._detector.predict(img_rgb)
        else:
            x0, x1 = self._detect_columns
            bbox, self._score = self._detector.predict(
                np.ascontiguousarray(img_rgb[:, x0:x1])
            )
            if bbox is not None:
                bbox = (bbox[0] + x0, bbox[1], bbox[2], bbox[3])
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

    Detect-only behavior arises two ways: a config with no `model:` section
    (permanent — `detection.rate` is the cost control), or load_pose=False
    (mode 'detect' on a pose config — the joint model is simply not loaded,
    the `tracker:` block is dropped because it propagates from joint output,
    and the frame gate falls back to runtime_detect_rate). Either way: no
    VitposeResult, the only output is the provider's Detection2DArray.
    """

    def __init__(self, config, image_cb, load_pose=True):
        self.config = config
        self.object_name = config["object"]
        detection_cfg = config["detection"]
        static_detect_only = is_detect_only(config)
        self.detect_only = static_detect_only or not load_pose

        provider_cfg = dict(
            detection_cfg.get("bbox_provider") or {"type": "full_frame"}
        )
        provider_cfg["_camera"] = config.get("camera")  # per-camera knobs
        rate = detection_cfg.get("rate")

        if static_detect_only:
            validate_detect_only(self.object_name, provider_cfg)
        elif not load_pose:
            provider_cfg.pop("tracker", None)
            provider_cfg["search_rate"] = None  # the frame gate below is the cap
            rate = runtime_detect_rate(detection_cfg)
            if not provider_cfg.get("publish_topic"):
                rospy.logwarn(
                    f"{self.object_name}: mode 'detect' will publish nothing "
                    "(bbox_provider has no publish_topic)"
                )

        if self.detect_only:
            self.model = None
            self.keypoint_names = []
            self.mask_classes = []
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

        self._rate = RateGate(rate)

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


MODES = ("off", "detect", "pose")


class VitposeNodeBase:
    """The ~enable / ~set_mode / ~set_config shell, shared with
    sim_vitpose_node so the two interfaces cannot drift.

    Modes: "off" (nothing loaded), "detect" (bbox provider only, boxes out),
    "pose" (provider + joint model, VitposeResult too). Loaded == enabled:
    leaving a mode disposes its pipeline (GPU memory included, see
    _dispose_pipeline). ~enable is the two-state alias: true = the fullest
    mode the config supports, false = off. Subclasses provide:

        log_name                        log-line prefix
        _build_pipeline(name, load_pose) construct the pipeline flavour
        _swap_config(name)              ~set_config once loaded; called
                                        HOLDING _pipeline_lock
        _image_cb(msg, pipeline)        per-frame work; must start with
                                        _pipeline_active(pipeline)
        _dispose_pipeline(pipeline)     teardown (default: just shutdown())

    _pipeline_lock guards the installed-pipeline reference (held briefly);
    _load_lock serializes slow builds so concurrent mode calls cannot each
    build a pipeline and leak the loser's live subscribers.
    """

    log_name = "vitpose_node"

    def _init_shell(self):
        """Call at the END of the subclass __init__: registers services, then
        honours ~enabled (which may synchronously load the pipeline)."""
        self.bridge = CvBridge()
        self.mode = "off"
        self._pipeline_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._config_name = rospy.get_param("~config", "gate")
        self._pipeline = None

        rospy.Service("~enable", SetBool, self._handle_enable)
        rospy.Service("~set_mode", SetString, self._handle_set_mode)
        rospy.Service("~set_config", SetString, self._handle_set_config)
        self._enabled_pub = rospy.Publisher("~enabled", Bool, queue_size=1, latch=True)
        self._mode_pub = rospy.Publisher("~mode", String, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(1.0), self._publish_status)

        if bool(rospy.get_param("~enabled", False)):
            ok, message = self._enable_full()
            if not ok:
                rospy.logerr(
                    f"{self.log_name}: ~enabled was true but {message}. "
                    "Staying up in mode 'off' — fix it and call ~enable."
                )
        self._publish_status(None)

        rospy.loginfo(
            f"{self.log_name} ready as '{rospy.get_name()}' "
            f"(config={self._config_name}, mode={self.mode})"
        )

    # ------------------------------------------------------------- hooks

    def _build_pipeline(self, name_or_path, load_pose=True):
        raise NotImplementedError

    def _swap_config(self, name_or_path):
        raise NotImplementedError

    def _dispose_pipeline(self, pipeline):
        pipeline.shutdown()

    # ------------------------------------------------------------- pipeline

    def _pipeline_active(self, pipeline) -> bool:
        """True iff not off AND `pipeline` is the installed one. Subscribers
        exist before installation and until shutdown; this drops frames from
        both windows."""
        if self.mode == "off":
            return False
        with self._pipeline_lock:
            return pipeline is self._pipeline

    def _transition(self, target) -> tuple:
        """Move to `target` mode; returns (success, message). Builds before
        installing, so a failed load leaves the current mode serving.
        Synchronous on purpose: the caller gets a truthful answer."""
        with self._load_lock:
            if target == self.mode:
                return True, f"already in mode '{self.mode}'"
            if target == "off":
                with self._pipeline_lock:
                    old, self._pipeline = self._pipeline, None
                    self.mode = "off"
                if old is not None:
                    self._dispose_pipeline(old)
                self._publish_status(None)
                return True, "mode 'off' (unloaded)"
            try:
                config = load_object_config(self._config_name)
                if target == "pose" and is_detect_only(config):
                    return False, (
                        f"'{config['object']}' is detect-only (no `model:` "
                        "section); mode 'pose' unavailable"
                    )
                pipeline = self._build_pipeline(
                    self._config_name, load_pose=(target == "pose")
                )
            except Exception as exc:
                return False, f"failed to load '{self._config_name}': {exc}"
            with self._pipeline_lock:
                old, self._pipeline = self._pipeline, pipeline
                self.mode = target
            if old is not None:
                self._dispose_pipeline(old)
            self._publish_status(None)
            return True, f"mode '{target}' (object={pipeline.object_name})"

    def _enable_full(self) -> tuple:
        """The fullest mode the current config supports."""
        try:
            config = load_object_config(self._config_name)
        except Exception as exc:
            return False, f"cannot read config '{self._config_name}': {exc}"
        return self._transition("detect" if is_detect_only(config) else "pose")

    # ------------------------------------------------------------- services

    def _publish_status(self, _event):
        self._enabled_pub.publish(Bool(data=self.mode != "off"))
        self._mode_pub.publish(String(data=self.mode))

    def _handle_enable(self, req):
        ok, message = self._enable_full() if req.data else self._transition("off")
        (rospy.loginfo if ok else rospy.logerr)(f"{self.log_name}: {message}")
        return SetBoolResponse(success=ok, message=message)

    def _handle_set_mode(self, req):
        target = req.data.strip().lower()
        if target not in MODES:
            return SetStringResponse(
                success=False,
                message=f"unknown mode '{req.data}' (modes: {', '.join(MODES)})",
            )
        ok, message = self._transition(target)
        (rospy.loginfo if ok else rospy.logerr)(
            f"{self.log_name}: set_mode('{target}'): {message}"
        )
        return SetStringResponse(success=ok, message=message)

    def _handle_set_config(self, req):
        with self._pipeline_lock:
            if self._pipeline is None:
                # Cold: just record the name; loading waits for a mode call.
                self._config_name = req.data
                return SetStringResponse(
                    success=True,
                    message=f"config set to '{req.data}' "
                    "(loads on ~enable/~set_mode)",
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

    def _build_pipeline(self, name_or_path, load_pose=True):
        ns = rospy.get_namespace().strip("/") or "taluy"
        config = load_object_config(name_or_path, ns)
        camera = rospy.get_param("~camera", "")
        if camera:
            apply_camera(config, camera, ns)
            rospy.logwarn(
                f"~camera override: '{config['object']}' running on cam_{camera}"
            )
        return _Pipeline(config, self._image_cb, load_pose=load_pose)

    def _dispose_pipeline(self, pipeline):
        """Teardown that actually returns the GPU memory: drop the model refs
        once in-flight callbacks have drained (they may hold the old pipeline
        for one more frame) and flush torch's allocator cache back to the
        driver. Torch's CUDA context itself lives until the process exits."""
        pipeline.shutdown()
        time.sleep(0.5)  # wall time: sim time may be paused
        pipeline.model = None
        pipeline.provider = None
        gc.collect()
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        with self._load_lock:
            try:
                new_pipeline = self._build_pipeline(
                    name_or_path, load_pose=(self.mode == "pose")
                )
            except Exception as exc:
                rospy.logerr(f"set_config('{name_or_path}') failed: {exc}")
                return
            with self._pipeline_lock:
                old, self._pipeline = self._pipeline, new_pipeline
                self._config_name = name_or_path
                if new_pipeline.detect_only and self.mode == "pose":
                    # A detect-only config caps the mode.
                    self.mode = "detect"
                    rospy.logwarn(
                        f"'{new_pipeline.object_name}' is detect-only: "
                        "mode clamped to 'detect'"
                    )
        self._dispose_pipeline(old)
        self._publish_status(None)
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
