#!/usr/bin/env python3

"""Valve keypoint producer: image → YOLO bbox → ViTPose → KeypointResult."""

import os

# Must run before torch/ultralytics import (PyTorch ApproximateClock workaround,
# https://github.com/pytorch/pytorch/issues/91516).
if os.environ.get("DEVICE", "cpu") == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("KINETO_DISABLED", "1")

import sys
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospkg
import rospy
import yaml

from cv_bridge import CvBridge
from ultralytics import YOLO

from auv_msgs.msg import Keypoint, KeypointResult
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse

# vitpose_inference sits next to this file but isn't an installable module yet.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.vitpose_inference import SKELETON_10, load_valve_pose  # noqa: E402


SKELETON_COLOR = (0, 255, 0)
KP_COLOR = (0, 80, 255)
LOW_CONF_COLOR = (100, 100, 100)
BBOX_COLOR = (0, 200, 255)


def _load_id_mapping(yaml_path: str, object_name: str) -> Tuple[int, str]:
    """Return (id_start, kp_object_name) for object_name from keypoint_objects.yaml."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    for entry in cfg.get("objects", []):
        if entry.get("name") == object_name:
            return int(entry["id_start"]), entry["object_names"][0]
    raise RuntimeError(f"object '{object_name}' not found in {yaml_path}")


class ValveKeypointNode:
    def __init__(self):
        rospy.init_node("valve_keypoint_node")

        self.image_topic = rospy.get_param(
            "~image_topic", "/taluy/cameras/cam_front/image_raw"
        )
        self.result_topic = rospy.get_param("~result_topic", "/keypoint_result_front")
        self.image_out_topic = rospy.get_param(
            "~image_out_topic", "/keypoint_image_front"
        )

        det_pkg = rospkg.RosPack().get_path("auv_detection")
        self.vitpose_model_path = rospy.get_param(
            "~vitpose_model_path", os.path.join(det_pkg, "models", "valve_vitpose.pth")
        )
        self.yolo_model_path = rospy.get_param(
            "~yolo_model_path", os.path.join(det_pkg, "models", "valve_yolo.pt")
        )
        self.device = rospy.get_param("~device", "cpu")
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.5))
        self.bbox_pad = float(rospy.get_param("~bbox_pad", 0.10))

        self.yolo_conf_thres = float(rospy.get_param("~yolo_conf_thres", 0.25))
        self.yolo_iou_thres = float(rospy.get_param("~yolo_iou_thres", 0.45))
        self.yolo_max_det = int(rospy.get_param("~yolo_max_det", 300))

        default_yaml = os.path.join(
            rospkg.RosPack().get_path("auv_vision"),
            "config",
            "keypoint_objects.yaml",
        )
        self.keypoint_objects_yaml = rospy.get_param(
            "~keypoint_objects_yaml", default_yaml
        )
        self.object_name = rospy.get_param("~object_name", "valve")
        self.id_start, self.kp_object = _load_id_mapping(
            self.keypoint_objects_yaml, self.object_name
        )

        rospy.loginfo(f"Loading YOLO from {self.yolo_model_path}")
        self.yolo = YOLO(self.yolo_model_path)
        self.yolo.fuse()

        rospy.loginfo(
            f"Loading ViTPose from {self.vitpose_model_path} (device={self.device})"
        )
        self.vp = load_valve_pose(self.vitpose_model_path, device=self.device)
        rospy.loginfo(f"ViTPose ready ({self.vp.num_kps} keypoints).")

        self.bridge = CvBridge()
        self._debug_thread: Optional[threading.Thread] = None
        self._debug_lock = threading.Lock()

        self.enabled = bool(rospy.get_param("~enabled", True))
        self.set_enabled_service = rospy.Service(
            "set_valve_keypoint_enabled", SetBool, self._handle_set_enabled
        )

        self.result_pub = rospy.Publisher(
            self.result_topic, KeypointResult, queue_size=1
        )
        self.image_pub = rospy.Publisher(self.image_out_topic, Image, queue_size=1)

        rospy.Subscriber(
            self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(
            f"valve_keypoint_node ready (image={self.image_topic}, "
            f"result={self.result_topic}, image_out={self.image_out_topic})"
        )

    def _handle_set_enabled(self, req):
        self.enabled = bool(req.data)
        message = f"valve_keypoint_node enabled set to: {self.enabled}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _image_cb(self, msg: Image):
        if not self.enabled:
            return
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"image decode failed: {e}")
            return

        bbox_xywh = self._run_yolo(img_bgr)

        kps, scores = None, None
        if bbox_xywh is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            kps, scores = self.vp.predict(img_rgb, bbox_xywh)

        keypoints = self._build_keypoints(kps, scores) if kps is not None else []
        result = KeypointResult()
        result.header = msg.header
        result.keypoints = keypoints
        self.result_pub.publish(result)

        if self.image_pub.get_num_connections() > 0:
            self._start_debug_publish(img_bgr, bbox_xywh, kps, scores, msg.header)

    def _run_yolo(
        self, img_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Run YOLO and return the highest-confidence bbox as padded (x, y, w, h)."""
        results = self.yolo.predict(
            source=img_bgr,
            conf=self.yolo_conf_thres,
            iou=self.yolo_iou_thres,
            max_det=self.yolo_max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        cx, cy, w, h = boxes.xywh[best].cpu().numpy().tolist()

        img_h, img_w = img_bgr.shape[:2]
        x = cx - w * 0.5
        y = cy - h * 0.5
        pad_x = w * self.bbox_pad
        pad_y = h * self.bbox_pad
        x = max(0.0, x - pad_x)
        y = max(0.0, y - pad_y)
        w = min(img_w - x, w + 2 * pad_x)
        h = min(img_h - y, h + 2 * pad_y)
        return (x, y, w, h)

    def _build_keypoints(self, kps: np.ndarray, scores: np.ndarray) -> List[Keypoint]:
        out: List[Keypoint] = []
        for i in range(len(kps)):
            conf = float(scores[i, 0])
            if conf < self.conf_threshold:
                continue
            msg = Keypoint()
            msg.id = i + self.id_start
            msg.x = float(kps[i, 0])
            msg.y = float(kps[i, 1])
            msg.confidence = conf
            msg.object = self.kp_object
            out.append(msg)
        return out

    def _start_debug_publish(self, img_bgr, bbox_xywh, kps, scores, header):
        with self._debug_lock:
            if self._debug_thread is not None and self._debug_thread.is_alive():
                return
            self._debug_thread = threading.Thread(
                target=self._publish_debug,
                args=(img_bgr.copy(), bbox_xywh, kps, scores, header),
                daemon=True,
            )
            self._debug_thread.start()

    def _publish_debug(self, vis, bbox_xywh, kps, scores, header):
        try:
            if bbox_xywh is not None:
                x, y, bw, bh = bbox_xywh
                cv2.rectangle(
                    vis,
                    (int(x), int(y)),
                    (int(x + bw), int(y + bh)),
                    BBOX_COLOR,
                    2,
                )
                cv2.putText(
                    vis,
                    "YOLO",
                    (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    BBOX_COLOR,
                    1,
                )

            if kps is not None and scores is not None:
                n_kps = len(kps)
                skeleton = (
                    SKELETON_10
                    if n_kps >= 10
                    else [(i, (i + 1) % 8) for i in range(min(n_kps, 8))]
                )
                for a, b in skeleton:
                    if (
                        a < n_kps
                        and b < n_kps
                        and scores[a, 0] > self.conf_threshold
                        and scores[b, 0] > self.conf_threshold
                    ):
                        cv2.line(
                            vis,
                            (int(kps[a, 0]), int(kps[a, 1])),
                            (int(kps[b, 0]), int(kps[b, 1])),
                            SKELETON_COLOR,
                            2,
                        )

                for i in range(n_kps):
                    pt = (int(kps[i, 0]), int(kps[i, 1]))
                    conf = float(scores[i, 0])
                    if i == 8:
                        label = f"O ({conf:.2f})"
                    elif i == 9:
                        label = f"H ({conf:.2f})"
                    else:
                        label = f"{i+1} ({conf:.2f})"
                    if conf > self.conf_threshold:
                        cv2.circle(vis, pt, 5, KP_COLOR, -1)
                        cv2.putText(
                            vis,
                            label,
                            (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            KP_COLOR,
                            1,
                        )
                    else:
                        cv2.circle(vis, pt, 4, LOW_CONF_COLOR, 1)
                        cv2.putText(
                            vis,
                            label,
                            (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            LOW_CONF_COLOR,
                            1,
                        )

            out = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            out.header = header
            self.image_pub.publish(out)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"debug image publish failed: {e}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        ValveKeypointNode().run()
    except rospy.ROSInterruptException:
        pass
