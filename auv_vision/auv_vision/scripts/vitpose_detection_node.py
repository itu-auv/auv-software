#!/usr/bin/env python3
"""ViTPose detection node: image -> (bbox source) -> joint model -> VitposeResult.

Deliberately thin producer (auv_vision/VITPOSE_PLAN.md §4): the model runs on
a crop chosen by a pluggable BboxProvider and every inference result — all K
keypoints with raw confidences plus per-class mask probability maps warped to
source resolution — is published as one auv_msgs/VitposeResult. All judgment
(confidence gates, pose solving, mask consumption) belongs downstream in
vitpose_process_node.

Bbox providers (the objectness seam):
    full_frame  one bbox covering the whole image (default; "assume the
                target is already framed").
    topic       latest vision_msgs/Detection2DArray from a configured topic,
                optional class-id filter and staleness gate. A future
                objectness/tracker node only needs to publish that topic.

Runtime model switching: ~set_config (auv_msgs/SetString) takes an object
name ("gate") or a YAML path; the new config + checkpoint load on a worker
thread and swap in atomically — frames keep flowing through the old model
until the swap.

Enable convention (mirrors the YOLO tracker nodes): ~enabled param,
~enable SetBool service, ~enabled Bool status republished at 1 Hz.
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

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.vitpose_config import (  # noqa: E402
    load_object_config,
    model_kwargs,
    resolve_checkpoint_path,
)
from utils.vitpose_inference import load_vitpose  # noqa: E402


# ─────────────────────────────────────────────── bbox providers


class FullFrameProvider:
    """One bbox covering the whole image."""

    def __init__(self, params):
        pass

    def get_bboxes(self, img_bgr, header):
        h, w = img_bgr.shape[:2]
        return [(0.0, 0.0, float(w), float(h))]

    def shutdown(self):
        pass


class TopicProvider:
    """Bboxes from a vision_msgs/Detection2DArray topic (future objectness).

    params:
        topic:     Detection2DArray topic to subscribe.
        class_id:  optional int; keep only detections whose top hypothesis
                   id matches.
        max_age:   seconds; detections older than this yield no bboxes.
    """

    def __init__(self, params):
        from vision_msgs.msg import Detection2DArray

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

    def get_bboxes(self, img_bgr, header):
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
    "topic": TopicProvider,
}


# ─────────────────────────────────────────────── active pipeline (swappable)


class _Pipeline:
    """Everything derived from one object config: model, provider, topics.

    Built off the ROS callback thread, installed atomically, torn down as a
    unit on switch — so a config swap can never mix (say) the old model with
    the new mask class names.
    """

    def __init__(self, config, image_cb):
        self.config = config
        self.object_name = config["object"]
        model_cfg = config["model"]
        detection_cfg = config["detection"]

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

        provider_cfg = dict(
            detection_cfg.get("bbox_provider") or {"type": "full_frame"}
        )
        provider_type = provider_cfg.pop("type", "full_frame")
        if provider_type not in _BBOX_PROVIDERS:
            raise ValueError(
                f"unknown bbox_provider '{provider_type}' "
                f"(available: {sorted(_BBOX_PROVIDERS)})"
            )
        self.provider = _BBOX_PROVIDERS[provider_type](provider_cfg)

        self.result_pub = rospy.Publisher(
            detection_cfg["result_topic"], VitposeResult, queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            detection_cfg["image_topic"],
            Image,
            image_cb,
            queue_size=1,
            buff_size=2**24,
        )

    def shutdown(self):
        self.image_sub.unregister()
        self.provider.shutdown()
        self.result_pub.unregister()


# ─────────────────────────────────────────────── node


class VitposeDetectionNode:
    def __init__(self):
        rospy.init_node("vitpose_detection_node")

        self.bridge = CvBridge()
        self.enabled = bool(rospy.get_param("~enabled", True))

        # Guards pipeline swaps; the image callback only takes it briefly to
        # snapshot the current pipeline reference.
        self._pipeline_lock = threading.Lock()
        self._swap_thread = None

        initial = rospy.get_param("~config", "gate")
        self._pipeline = self._build_pipeline(initial)

        rospy.Service("~enable", SetBool, self._handle_enable)
        rospy.Service("~set_config", SetString, self._handle_set_config)
        self._enabled_pub = rospy.Publisher("~enabled", Bool, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(1.0), self._publish_enabled)

        rospy.loginfo(
            f"vitpose_detection_node ready (object={self._pipeline.object_name}, "
            f"enabled={self.enabled})"
        )

    # ------------------------------------------------------------- pipeline

    def _build_pipeline(self, name_or_path):
        config = load_object_config(name_or_path)
        return _Pipeline(config, self._image_cb)

    # ------------------------------------------------------------- services

    def _publish_enabled(self, _event):
        self._enabled_pub.publish(Bool(data=self.enabled))

    def _handle_enable(self, req):
        self.enabled = bool(req.data)
        message = f"vitpose_detection_node enabled set to: {self.enabled}"
        rospy.loginfo(message)
        self._publish_enabled(None)
        return SetBoolResponse(success=True, message=message)

    def _handle_set_config(self, req):
        with self._pipeline_lock:
            if self._swap_thread is not None and self._swap_thread.is_alive():
                return SetStringResponse(
                    success=False, message="a config swap is already in progress"
                )
            self._swap_thread = threading.Thread(
                target=self._swap_worker, args=(req.data,), daemon=True
            )
            self._swap_thread.start()
        return SetStringResponse(
            success=True,
            message=f"loading '{req.data}' in the background "
            "(old config keeps serving until the swap)",
        )

    def _swap_worker(self, name_or_path):
        try:
            new_pipeline = self._build_pipeline(name_or_path)
        except Exception as exc:
            rospy.logerr(f"set_config('{name_or_path}') failed: {exc}")
            return
        with self._pipeline_lock:
            old, self._pipeline = self._pipeline, new_pipeline
        old.shutdown()
        rospy.loginfo(
            f"vitpose_detection_node switched to object "
            f"'{new_pipeline.object_name}' ({new_pipeline.config['_path']})"
        )

    # ------------------------------------------------------------- inference

    def _image_cb(self, msg):
        if not self.enabled:
            return
        with self._pipeline_lock:
            pipeline = self._pipeline
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logerr_throttle(5.0, f"image decode failed: {exc}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_bgr.shape[:2]
        full_frame = (0.0, 0.0, float(w_img), float(h_img))

        for bbox in pipeline.provider.get_bboxes(img_bgr, msg.header):
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

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        VitposeDetectionNode().run()
    except rospy.ROSInterruptException:
        pass
