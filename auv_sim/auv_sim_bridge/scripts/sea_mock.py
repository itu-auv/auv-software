#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class UnderwaterEffectNode:
    def __init__(self):
        self.bridge = CvBridge()
        # Bottom camera subscriber/publisher
        self.sub_bottom = rospy.Subscriber(
            "cam_bottom/image_raw",
            Image,
            self.callback_bottom,
            queue_size=1,
        )
        self.pub_bottom = rospy.Publisher(
            "cam_bottom/image_underwater", Image, queue_size=1
        )
        # Front camera subscriber/publisher
        self.sub_front = rospy.Subscriber(
            "cam_front/image_raw",
            Image,
            self.callback_front,
            queue_size=1,
        )
        self.pub_front = rospy.Publisher(
            "cam_front/image_underwater", Image, queue_size=1
        )

    # -------- Core pipeline --------
    def process_image(self, cv_img, camera_type):
        # Step 1: Apply underwater color absorption and tint
        uw = self.apply_underwater_effect(cv_img)
        # Step 2: Apply slight gamma correction
        uw = self.apply_gamma(uw, gamma=0.9)
        # Step 3: Apply fog/backscatter depending on camera type
        foggy = self.apply_fog_effect(uw, camera_type=camera_type)
        return foggy

    def apply_underwater_effect(self, cv_img):
        img = cv_img.astype(np.float32)
        b, g, r = cv2.split(img)

        # Blue enhanced, Green slightly boosted, Red strongly attenuated
        b = cv2.addWeighted(b, 1.25, g, 0.06, 0.0)
        g = cv2.addWeighted(g, 1.10, b, 0.05, 0.0)
        r = r * 0.45

        img_turquoise = cv2.merge((b, g, r))

        # Stronger Gaussian blur for underwater softness
        blurred = cv2.GaussianBlur(img_turquoise, (19, 19), 5)

        # Turquoise overlay
        overlay = np.full_like(blurred, (32, 110, 150), dtype=np.float32)
        underwater = cv2.addWeighted(blurred, 0.65, overlay, 0.35, 0.0)

        # Darkening more aggressively (simulate 5 m depth)
        darkened = cv2.addWeighted(
            underwater, 0.55, np.zeros_like(underwater), 0.45, 0.0
        )

        return np.clip(darkened, 0, 255).astype(np.uint8)

    def apply_gamma(self, img, gamma=0.9):
        if gamma <= 0:
            return img
        inv = 1.0 / gamma
        # Lookup table for gamma correction
        table = (np.linspace(0, 1, 256) ** inv) * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(img, table)

    def apply_fog_effect(self, cv_img, camera_type="front"):
        h, w = cv_img.shape[:2]

        # Different fog intensity for front vs bottom cameras
        if camera_type == "front":
            mask_max = 1.0
            sigmaX = 75
            sigmaY = 75
            fog_mix_img = 0.45  # less of original image
            fog_mix_grey = 0.55  # more grey overlay
        else:  # bottom camera
            mask_max = 0.65
            sigmaX = 60
            sigmaY = 60
            fog_mix_img = 0.40
            fog_mix_grey = 0.60

        # Vertical gradient mask
        mask = np.linspace(0.0, mask_max, h, dtype=np.float32).reshape(-1, 1)
        mask = np.repeat(mask, w, axis=1)
        mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigmaX, sigmaY=sigmaY)
        mask = np.clip(mask, 0.0, 1.0)

        # Create foggy version: blur + grey overlay
        foggy = cv2.GaussianBlur(cv_img, (29, 29), 7)
        grey = np.full_like(foggy, 210, dtype=np.uint8)
        foggy = cv2.addWeighted(foggy, fog_mix_img, grey, fog_mix_grey, 0.0)

        # Blend original and foggy
        mask3 = np.stack([mask] * 3, axis=2)
        base = cv_img.astype(np.float32)
        foggy = foggy.astype(np.float32)
        result = base * (1.0 - mask3) + foggy * mask3

        return np.clip(result, 0, 255).astype(np.uint8)

    # -------- Callbacks --------
    def callback_bottom(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        out = self.process_image(cv_img, camera_type="bottom")
        out_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_bottom.publish(out_msg)

    def callback_front(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        out = self.process_image(cv_img, camera_type="front")
        out_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_front.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("underwater_effect_node")
    UnderwaterEffectNode()
    rospy.spin()
