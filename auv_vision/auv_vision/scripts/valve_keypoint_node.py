#!/usr/bin/env python3

"""
Valve keypoint producer node.

Real-camera analogue of sim_keypoint_node.py:

    Image → HSV bbox → ViTPose → KeypointResult

The HSV bbox step is a STOPGAP until the YOLO valve class is trained — at
that point only the bbox source changes; the ViTPose + KeypointResult half
stays identical (see the comment block on _detect_bbox_hsv).

Output topology matches sim_keypoint_node, so the same keypoint_pose_node
consumer (configured via keypoint_objects.yaml) runs identically against
either producer.

Topics:
  ~image_topic         (sensor_msgs/Image or CompressedImage, auto-detected)
  ~result_topic        (auv_msgs/KeypointResult)
  ~image_out_topic     (sensor_msgs/Image, only published when subscribed)

Keypoint ID convention (matches keypoint_objects.yaml valve entry):
  id 1..8  → bolts -135° .. 180°  (object = "valve")
  id 9     → valve face center    (object = "valve")
  id 10    → handle tip           (object = "valve",  excluded from PnP via
                                    `model` having only 9 points; consumed by
                                    keypoint_pose_node's orientation_refinement
                                    to anchor the rotation around the face
                                    normal to the handle direction)
"""

import os
import sys
import threading
from typing import List, Optional

import cv2
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge

from auv_msgs.msg import Keypoint, KeypointResult
from sensor_msgs.msg import CompressedImage, Image

# Add the source scripts directory to sys.path so `vitpose_inference` resolves
# to the real module file rather than the catkin_install_python relay stub
# (the stub exec's the source into a private context dict, so module-level
# symbols like `ValvePose` aren't visible to a normal `from … import …`).
# Same pattern keypoint_pose_node uses for its utils/ imports.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from vitpose_inference import SKELETON_10, ValvePose  # noqa: E402


# ---------------------------------------------------------------------------
# Visualization constants — match valve_pnp_bag_node so debug images look
# the same regardless of which valve node is running.
# ---------------------------------------------------------------------------

SKELETON_COLOR = (0, 255, 0)
KP_COLOR = (0, 80, 255)
LOW_CONF_COLOR = (100, 100, 100)
BBOX_COLOR = (0, 200, 255)


# ---------------------------------------------------------------------------
# HSV bbox detector
# ---------------------------------------------------------------------------
#
# STOPGAP — replace this whole block with a YOLO detection subscription once
# the valve class is trained.  The contract for the rest of the node is just
# `(x, y, w, h)` in pixel coordinates, so swapping in a different bbox source
# requires no other changes downstream.
#
# Lifted verbatim from valve_pnp_bag_node.detect_valve_bbox_hsv so behaviour
# matches the bag-replay tooling we already validated against.


def _detect_bbox_hsv(
    img_bgr: np.ndarray,
    hsv_low: np.ndarray,
    hsv_high: np.ndarray,
    min_area: int,
    bbox_pad: float,
):
    """Find the orange/bronze valve via HSV thresholding.

    Returns ((x, y, w, h), mask) or (None, mask).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None, mask

    rx, ry, rw, rh = cv2.boundingRect(best)

    img_h, img_w = img_bgr.shape[:2]
    pad_x = rw * bbox_pad
    pad_y = rh * bbox_pad
    rx = max(0, rx - pad_x)
    ry = max(0, ry - pad_y)
    rw = min(img_w - rx, rw + 2 * pad_x)
    rh = min(img_h - ry, rh + 2 * pad_y)

    return (rx, ry, rw, rh), mask


# ---------------------------------------------------------------------------
# ID / object-name mapping
# ---------------------------------------------------------------------------
#
# Maps a 0-based ViTPose channel index to the (id, object) tuple emitted on
# KeypointResult.  ViTPose models trained for the valve carry either 8
# channels (bolts only) or 10 channels (bolts + face origin + handle tip);
# both cases are handled.  The yaml-side id_start=1 means our ids run 1..9
# for the PnP set and 10 for the handle.


def _id_and_object_for_channel(idx: int):
    if 0 <= idx <= 7:  # bolts: model channels 0..7 → ids 1..8
        return idx + 1, "valve"
    if idx == 8:  # face center (origin)
        return 9, "valve"
    if idx == 9:  # handle tip — same object name; PnP excludes
        return 10, "valve"  #   it via the 9-point model, refinement uses it
    return None  # unexpected extra channels


# ---------------------------------------------------------------------------
# Image-topic auto-detect
# ---------------------------------------------------------------------------


def _resolve_image_msg_type(topic: str, wait_secs: float = 2.0):
    """Return (msg_class, type_name) for the given image topic.

    Polls rospy.get_published_topics for up to wait_secs.  If the topic isn't
    published yet, falls back to the URL convention (`/compressed` suffix
    means CompressedImage, otherwise plain Image).
    """
    deadline = rospy.Time.now() + rospy.Duration(wait_secs)
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        for name, ttype in rospy.get_published_topics():
            if name == topic:
                if ttype == "sensor_msgs/CompressedImage":
                    return CompressedImage, ttype
                return Image, ttype
        rospy.sleep(0.1)

    if topic.endswith("/compressed"):
        return CompressedImage, "sensor_msgs/CompressedImage (assumed)"
    return Image, "sensor_msgs/Image (assumed)"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class ValveKeypointNode:
    def __init__(self):
        rospy.init_node("valve_keypoint_node")

        # ----- topics & frames -----
        self.image_topic = rospy.get_param(
            "~image_topic", "/taluy/cameras/cam_front/image_raw"
        )
        self.result_topic = rospy.get_param("~result_topic", "/keypoint_result_front")
        self.image_out_topic = rospy.get_param(
            "~image_out_topic", "/keypoint_image_front"
        )

        # ----- ViTPose -----
        default_model = os.path.join(
            rospkg.RosPack().get_path("auv_detection"),
            "models",
            "valve_vitpose.pth",
        )
        self.model_path = rospy.get_param("~vitpose_model_path", default_model)
        self.device = rospy.get_param("~device", "cpu")
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.5))

        # ----- HSV bbox detector (stopgap) -----
        self.bbox_pad = float(rospy.get_param("~bbox_pad", 0.15))
        self.min_contour_area = int(rospy.get_param("~min_contour_area", 800))
        self.hsv_low = np.array(
            rospy.get_param("~hsv_low", [0, 20, 20]), dtype=np.uint8
        )
        self.hsv_high = np.array(
            rospy.get_param("~hsv_high", [45, 255, 255]), dtype=np.uint8
        )

        # ----- ViTPose load (slow — do it before subscribers go live) -----
        rospy.loginfo(f"Loading ViTPose from {self.model_path} (device={self.device})…")
        self.vp = ValvePose(self.model_path, device=self.device)
        rospy.loginfo(f"ViTPose ready ({self.vp.num_kps} keypoints).")

        # ----- runtime -----
        self.bridge = CvBridge()
        self._debug_thread: Optional[threading.Thread] = None
        self._debug_lock = threading.Lock()

        self.result_pub = rospy.Publisher(
            self.result_topic, KeypointResult, queue_size=1
        )
        self.image_pub = rospy.Publisher(self.image_out_topic, Image, queue_size=1)

        # Auto-detect Image vs CompressedImage on the configured topic.
        msg_class, ttype = _resolve_image_msg_type(self.image_topic)
        cb = self._compressed_cb if msg_class is CompressedImage else self._image_cb
        rospy.Subscriber(self.image_topic, msg_class, cb, queue_size=1, buff_size=2**24)
        rospy.loginfo(
            f"valve_keypoint_node ready  "
            f"(image_topic={self.image_topic} [{ttype}], "
            f"result_topic={self.result_topic}, "
            f"image_out_topic={self.image_out_topic})"
        )

    # ----- subscribers -----

    def _image_cb(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"image decode failed: {e}")
            return
        self._process(img_bgr, msg.header)

    def _compressed_cb(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return
        except Exception as e:
            rospy.logerr_throttle(5.0, f"compressed image decode failed: {e}")
            return
        self._process(img_bgr, msg.header)

    # ----- pipeline -----

    def _process(self, img_bgr: np.ndarray, header):
        bbox_xywh, _ = _detect_bbox_hsv(
            img_bgr,
            self.hsv_low,
            self.hsv_high,
            self.min_contour_area,
            self.bbox_pad,
        )

        kps = scores = None
        if bbox_xywh is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            kps, scores = self.vp.predict(img_rgb, bbox_xywh)

        # Always publish a KeypointResult — empty when nothing is detected —
        # so downstream consumers can rely on stamps to detect dropouts.
        keypoints = self._build_keypoints(kps, scores) if kps is not None else []
        result = KeypointResult()
        result.header = header
        result.keypoints = keypoints
        self.result_pub.publish(result)

        if self.image_pub.get_num_connections() > 0:
            self._start_debug_publish(img_bgr, bbox_xywh, kps, scores, header)

    def _build_keypoints(self, kps: np.ndarray, scores: np.ndarray) -> List[Keypoint]:
        """Filter by confidence and tag with (id, object) per channel."""
        out: List[Keypoint] = []
        n = len(kps)
        for i in range(n):
            conf = float(scores[i, 0])
            if conf < self.conf_threshold:
                continue
            id_obj = _id_and_object_for_channel(i)
            if id_obj is None:
                continue
            kp_id, obj_name = id_obj

            msg = Keypoint()
            msg.id = kp_id
            msg.x = float(kps[i, 0])
            msg.y = float(kps[i, 1])
            msg.confidence = conf
            msg.object = obj_name
            out.append(msg)
        return out

    # ----- debug image (threaded so callbacks stay snappy) -----

    def _start_debug_publish(self, img_bgr, bbox_xywh, kps, scores, header):
        with self._debug_lock:
            if self._debug_thread is not None and self._debug_thread.is_alive():
                return  # last debug frame still rendering — skip this one
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
                    "HSV",
                    (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    BBOX_COLOR,
                    1,
                )

            if kps is not None and scores is not None:
                n_kps = len(kps)
                # Skeleton: 8-bolt ring + (optional) origin→handle line.
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
                        label = f"S ({conf:.2f})"
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
