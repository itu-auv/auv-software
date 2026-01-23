#!/usr/bin/env python3
import math
import numpy as np
import cv2

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image


class FakeYolo(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_fake_yolo = rospy.Publisher("yolo_fake_image", Image, queue_size=1)
        self.sub_camera = rospy.Subscriber(
            "taluy/cameras/cam_bottom/image_raw", Image, self.cb_image, queue_size=1
        )

    def cb_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        out_msg = self.bridge.cv2_to_imgmsg(mask_bgr, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_fake_yolo.publish(out_msg)


def main():
    rospy.init_node("fake_yolo")
    FakeYolo()
    rospy.spin()


if __name__ == "__main__":
    main()
