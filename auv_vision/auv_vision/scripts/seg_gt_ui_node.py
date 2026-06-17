#!/usr/bin/env python3
"""Ground-truth drawing UI as a ROS node (runs on the operator laptop).

Subscribes to the robot camera over the ROS network and serves a web canvas
(see seg_tuning/gt_web/server.py) at http://localhost:<port>/. The "Grab live
frame" button pulls the most recent frame the robot actually sees, you paint the
yellow pipe, and Save writes image.png + gt.png into the session directory that
seg_tuning/seg_optimize.py reads.

This is the laptop-side counterpart of opencv_seg_publisher.py (which runs on the
Jetson). Both share one ROS master; set ROS_MASTER_URI to the robot. Use
~use_compressed:=true to pull compressed frames over the network cheaply.
"""

import os
import sys
import threading

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from seg_tuning.gt_web.server import (  # noqa: E402
    make_server,
    serve_forever_in_thread,
)


class SegGtUiNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.use_compressed = bool(rospy.get_param("~use_compressed", False))
        self.host = rospy.get_param("~host", "0.0.0.0")
        self.port = int(rospy.get_param("~port", 8088))
        self.save_dir = rospy.get_param("~session_dir", "/tmp/seg_session")
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))

        self._lock = threading.Lock()
        self._latest_jpeg = None  # most recent frame, pre-encoded as JPEG bytes

        if self.use_compressed:
            self.sub = rospy.Subscriber(
                "image_raw/compressed", CompressedImage, self.cb_compressed,
                queue_size=1, buff_size=2 ** 24)
        else:
            self.sub = rospy.Subscriber(
                "image_raw", Image, self.cb_image,
                queue_size=1, buff_size=2 ** 24)

        self.server = make_server(
            self.host, self.port, self._frame_provider, self.save_dir,
            on_save=self._on_save)
        serve_forever_in_thread(self.server)
        rospy.loginfo(
            "seg_gt_ui serving http://%s:%d/  (session_dir=%s, compressed=%s)",
            "localhost" if self.host == "0.0.0.0" else self.host,
            self.port, self.save_dir, self.use_compressed)
        rospy.on_shutdown(self._shutdown)

    def _store(self, bgr):
        ok, enc = cv2.imencode(".jpg", bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._lock:
                self._latest_jpeg = enc.tobytes()

    def cb_image(self, msg):
        try:
            self._store(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        except Exception as e:  # noqa: BLE001
            rospy.logerr_throttle(2.0, "image conversion failed: %s" % e)

    def cb_compressed(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                self._store(bgr)
        except Exception as e:  # noqa: BLE001
            rospy.logerr_throttle(2.0, "image decode failed: %s" % e)

    def _frame_provider(self):
        with self._lock:
            return self._latest_jpeg

    def _on_save(self, sample_dir, image_path, mask_path):
        rospy.loginfo("GT sample saved: %s", sample_dir)

    def _shutdown(self):
        try:
            self.server.shutdown()
        except Exception:  # noqa: BLE001
            pass


def main():
    rospy.init_node("seg_gt_ui")
    SegGtUiNode()
    rospy.spin()


if __name__ == "__main__":
    main()
