#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class CvImageProcNode:
    def __init__(self):
        rospy.init_node("cv_image_proc")

        self.jpeg_quality = rospy.get_param("~jpeg_quality", 80)
        self.bridge = CvBridge()

        self.pub_raw_compressed = rospy.Publisher(
            "image_raw/compressed", CompressedImage, queue_size=1
        )
        self.pub_rect = rospy.Publisher("image_rect_color", Image, queue_size=1)
        self.pub_rect_compressed = rospy.Publisher(
            "image_rect_color/compressed", CompressedImage, queue_size=1
        )

        self.map1 = None
        self.map2 = None
        self.map_shape = None
        self.latest_info = None

        self.info_sub = rospy.Subscriber(
            "camera_info", CameraInfo, self._info_callback, queue_size=1
        )
        self.raw_sub = rospy.Subscriber(
            "image_raw", Image, self._image_callback, queue_size=1, buff_size=2**24
        )

        rospy.loginfo("cv_image_proc node started")

    def _info_callback(self, info_msg):
        self.latest_info = info_msg

    def _make_compressed(self, frame, header):
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return None
        msg.data = buf.tobytes()
        return msg

    def _update_maps(self, info_msg, width, height):
        if self.map_shape != None:
            return True
        if len(info_msg.K) != 9 or len(info_msg.D) == 0:
            return False

        shape = (width, height, tuple(info_msg.K), tuple(info_msg.D), tuple(info_msg.R))
        if self.map_shape == shape and self.map1 is not None and self.map2 is not None:
            return True

        K = np.array(info_msg.K, dtype=np.float64).reshape(3, 3)
        D = np.array(info_msg.D, dtype=np.float64)
        R = np.array(info_msg.R, dtype=np.float64).reshape(3, 3)

        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha=0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, R, new_K, (width, height), cv2.CV_16SC2
        )
        self.map_shape = shape
        return True

    def _image_callback(self, raw_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(raw_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, f"cv_bridge conversion failed: {exc}")
            return

        raw_compressed = self._make_compressed(frame, raw_msg.header)
        if raw_compressed is not None:
            self.pub_raw_compressed.publish(raw_compressed)


        info_msg = self.latest_info
        if info_msg is None:
            return

        if not self._update_maps(info_msg, raw_msg.width, raw_msg.height):
            return

        rectified = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        rect_msg = self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8")
        rect_msg.header = raw_msg.header
        self.pub_rect.publish(rect_msg)

        rect_compressed = self._make_compressed(rectified, raw_msg.header)
        if rect_compressed is not None:
            self.pub_rect_compressed.publish(rect_compressed)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = CvImageProcNode()
    node.spin()