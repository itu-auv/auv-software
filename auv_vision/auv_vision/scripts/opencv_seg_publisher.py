#!/usr/bin/env python3
"""Classic-OpenCV yellow-pipe segmentation publisher.

Drop-in fallback for the YOLO segment model: it consumes the same bottom-camera
image topic and publishes the same mono8 mask topic that pipe_follower_legacy
consumes, but derives the mask from a colour pipeline instead of a YOLO model.
Use it on the robot Jetson when the segmentation network is unavailable.

All thresholds are live-tunable via dynamic_reconfigure (rqt_reconfigure), and a
tuned parameter set produced offline by seg_tuning/seg_optimize.py can be pushed
with ``rosrun dynamic_reconfigure dynparam load <node> best_params.yaml``.

The actual segmentation lives in utils/pipe_segmentation.py (ROS-free), shared
with the offline tuning tools so the live mask matches what was optimised.
"""

import os
import sys

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Trigger, TriggerResponse
from dynamic_reconfigure.server import Server
from auv_vision.cfg import OpenCVSegConfig

# Add scripts directory to path so we can import the shared core (same pattern as
# camera_detection_node.py).
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.pipe_segmentation import (  # noqa: E402
    segment,
    make_debug_overlay,
    params_from_dict,
)


class OpenCVSegPublisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.use_compressed = bool(rospy.get_param("~use_compressed", False))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.is_enabled = bool(rospy.get_param("~start_enabled", True))

        # dynamic_reconfigure owns the SegParams; the server callback fires once
        # immediately on construction and populates self.params.
        self.params = None
        self.reconfigure_server = Server(OpenCVSegConfig, self.cb_reconfigure)

        # Output: the mask topic pipe_follower_legacy subscribes to. The local
        # name is remapped by launch to match the YOLO segment mask topic.
        self.pub_mask = rospy.Publisher("bottle_mask", Image, queue_size=1)
        self.pub_debug = (
            rospy.Publisher("~debug_image/compressed", CompressedImage, queue_size=1)
            if self.publish_debug
            else None
        )

        # Input image (relative "image_raw"; remapped to the bottom camera in the
        # launch file). Raw or compressed transport.
        if self.use_compressed:
            self.sub = rospy.Subscriber(
                "image_raw/compressed", CompressedImage, self.cb_image,
                queue_size=1, buff_size=2 ** 24)
        else:
            self.sub = rospy.Subscriber(
                "image_raw", Image, self.cb_image,
                queue_size=1, buff_size=2 ** 24)

        # Optional enable/disable to silence the publisher (pipe_follower pattern).
        self.srv_enable = rospy.Service("~enable", Trigger, self.cb_enable)
        self.srv_disable = rospy.Service("~disable", Trigger, self.cb_disable)

        rospy.loginfo(
            "opencv_seg_publisher started (use_compressed=%s, enabled=%s)",
            self.use_compressed, self.is_enabled)

    def cb_reconfigure(self, config, level):
        self.params = params_from_dict(dict(config))
        return config

    def cb_enable(self, _req):
        self.is_enabled = True
        return TriggerResponse(success=True, message="opencv_seg_publisher enabled")

    def cb_disable(self, _req):
        self.is_enabled = False
        return TriggerResponse(success=True, message="opencv_seg_publisher disabled")

    def _to_bgr(self, msg):
        if self.use_compressed:
            arr = np.frombuffer(msg.data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def cb_image(self, msg):
        if not self.is_enabled or self.params is None:
            return
        try:
            bgr = self._to_bgr(msg)
        except Exception as e:  # noqa: BLE001
            rospy.logerr_throttle(2.0, "image conversion failed: %s" % e)
            return
        if bgr is None or bgr.size == 0:
            return

        mask = segment(bgr, self.params)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        if self.pub_debug is not None and self.pub_debug.get_num_connections() > 0:
            overlay = make_debug_overlay(bgr, mask)
            ok, enc = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                dbg = CompressedImage()
                dbg.header = msg.header
                dbg.format = "jpeg"
                dbg.data = np.array(enc).tobytes()
                self.pub_debug.publish(dbg)


def main():
    rospy.init_node("opencv_seg_publisher")
    OpenCVSegPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
