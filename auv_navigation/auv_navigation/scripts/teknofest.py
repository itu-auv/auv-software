#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from dynamic_reconfigure.server import Server
from auv_navigation.cfg import CableFollowerConfig  # Paket ismine göre uyarlanmalı


class CableFollower:
    def __init__(self):
        rospy.init_node("cable_follower_node")

        self.bridge = CvBridge()
        self.twist_pub = rospy.Publisher("/taluy/cmd_vel", Twist, queue_size=10)
        self.image_sub = rospy.Subscriber(
            "/taluy/cameras/cam_front/image_rect_color/compressed",
            Image,
            self.image_callback,
        )

        self.twist_msg = Twist()
        self.lower_hsv = np.array([20, 100, 100])
        self.upper_hsv = np.array([30, 255, 255])

        self.reconfig_server = Server(CableFollowerConfig, self.reconfigure_cb)
        rospy.loginfo("Cable follower with dynamic HSV started.")
        rospy.spin()

    def reconfigure_cb(self, config, level):
        self.lower_hsv = np.array([config.lower_h, config.lower_s, config.lower_v])
        self.upper_hsv = np.array([config.upper_h, config.upper_s, config.upper_v])
        return config

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge Error: %s", e)
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        h, w, _ = cv_image.shape
        mask[0 : int(h * 0.6), :] = 0
        mask[int(h * 0.8) : h, :] = 0

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - w // 2
            self.twist_msg.linear.x = 0.3
            self.twist_msg.angular.z = -error / 400.0
        else:
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = 0.3

        self.twist_pub.publish(self.twist_msg)


if __name__ == "__main__":
    try:
        CableFollower()
    except rospy.ROSInterruptException:
        pass
