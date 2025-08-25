#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class UnderwaterEffectNode:
    def __init__(self):
        self.bridge = CvBridge()
        # Alt kamera
        self.sub_bottom = rospy.Subscriber(
            "/taluy/cameras/cam_bottom/image_raw",
            Image,
            self.callback_bottom,
            queue_size=1,
        )
        self.pub_bottom = rospy.Publisher(
            "/taluy/cameras/cam_bottom/image_underwater", Image, queue_size=1
        )
        # Ön kamera
        self.sub_front = rospy.Subscriber(
            "/taluy/cameras/cam_front/image_raw",
            Image,
            self.callback_front,
            queue_size=1,
        )
        self.pub_front = rospy.Publisher(
            "/taluy/cameras/cam_front/image_underwater", Image, queue_size=1
        )

    def apply_underwater_effect(self, cv_img):
        b, g, r = cv2.split(cv_img)
        b = cv2.addWeighted(b, 1.25, g, 0.10, 0)
        g = cv2.addWeighted(g, 1.15, b, 0.08, 0)
        r = cv2.multiply(r, 0.65)
        img_turquoise = cv2.merge((b, g, r.astype(np.uint8)))
        blurred = cv2.GaussianBlur(img_turquoise, (11, 11), 0)
        overlay = np.full(
            blurred.shape, (40, 120, 160), dtype=np.uint8
        )  # Daha koyu turkuaz
        underwater = cv2.addWeighted(blurred, 0.7, overlay, 0.3, 0)
        # Daha güçlü karanlıklaştırma
        darkened = cv2.addWeighted(underwater, 0.6, np.zeros_like(underwater), 0.4, 0)
        return darkened

    def apply_fog_effect(self, cv_img):
        h, w = cv_img.shape[:2]
        mask = np.linspace(0, 1.2, h).reshape(-1, 1)  # 1.2 ile daha yoğun sis
        mask = np.repeat(mask, w, axis=1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=40, sigmaY=40)
        mask = np.clip(mask, 0, 1)
        foggy = cv2.GaussianBlur(cv_img, (31, 31), 0)
        foggy = cv2.addWeighted(foggy, 0.5, np.full_like(foggy, 210), 0.5, 0)
        mask3 = np.stack([mask] * 3, axis=2)
        result = cv_img * (1 - mask3) + foggy * mask3
        return result.astype(np.uint8)

    def callback_bottom(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        underwater = self.apply_underwater_effect(cv_img)
        foggy = self.apply_fog_effect(underwater)
        out_msg = self.bridge.cv2_to_imgmsg(foggy.astype(np.uint8), encoding="bgr8")
        out_msg.header = msg.header
        self.pub_bottom.publish(out_msg)

    def callback_front(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Önce su altı efekti, sonra sis efekti uygula
        underwater = self.apply_underwater_effect(cv_img)
        foggy = self.apply_fog_effect(underwater)
        out_msg = self.bridge.cv2_to_imgmsg(foggy.astype(np.uint8), encoding="bgr8")
        out_msg.header = msg.header
        self.pub_front.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("underwater_effect_node")
    UnderwaterEffectNode()
    rospy.spin()
