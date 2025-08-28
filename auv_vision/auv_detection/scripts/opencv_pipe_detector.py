#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class PipeDetectorSimple:
    def __init__(self):
        # ---- Params ----
        self.use_percentile = rospy.get_param(
            "~use_percentile", True
        )  # True -> dinamik eşik
        self.black_percentile = rospy.get_param(
            "~black_percentile", 5
        )  # 0-100 arası; 5 iyi bir başlangıç
        self.fixed_thresh = rospy.get_param(
            "~fixed_thresh", 30
        )  # use_percentile=False ise kullanılır (0-255)
        self.blur_ksize = rospy.get_param(
            "~blur_ksize", 3
        )  # 0/1 => kapalı; >=3 tek sayı
        self.morph_close = rospy.get_param("~morph_close", 5)  # tek sayı; 0/1 => kapalı
        self.morph_open = rospy.get_param("~morph_open", 3)  # tek sayı; 0/1 => kapalı
        self.min_area_px = rospy.get_param("~min_area_px", 1000)  # küçük gürültüyü ele
        self.publish_thresh = rospy.get_param("~publish_thresh", True)  # debug

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(
            "/taluy/cameras/cam_bottom/image_underwater",
            Image,
            self.cb,
            queue_size=1,
            buff_size=2**24,
        )
        self.pub_mask = rospy.Publisher("pipe_mask", Image, queue_size=1)
        self.pub_gray = rospy.Publisher("gray_image", Image, queue_size=1)
        self.pub_thr = rospy.Publisher("thresh_image", Image, queue_size=1)

    @staticmethod
    def _odd(k):
        k = int(k)
        return k if k % 2 == 1 else (k + 1)

    def cb(self, msg):
        print("Received image")
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # İsteğe bağlı hafif blur (gürültüyü yumuşatır)
        if self.blur_ksize and self.blur_ksize >= 3:
            k = self._odd(self.blur_ksize)
            gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            gray_blur = gray

        # --- Eşik belirleme ---
        if self.use_percentile:
            p = np.clip(self.black_percentile, 0, 100)
            thr_val = int(np.percentile(gray_blur, p))
            # Aşırı düşük çıkarsa az kaldır (zemin tamamen siyaha yakınsa):
            thr_val = max(thr_val, 10)
        else:
            thr_val = int(np.clip(self.fixed_thresh, 0, 255))

        # “Siyaha yakın” pikseller (<= thr_val) beyaz, diğerleri siyah
        _, thr = cv2.threshold(gray_blur, thr_val, 255, cv2.THRESH_BINARY_INV)

        # --- Morfoloji (opsiyonel ve hafif) ---
        if self.morph_close and self.morph_close >= 3:
            k = self._odd(self.morph_close)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        if self.morph_open and self.morph_open >= 3:
            k = self._odd(self.morph_open)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

        # --- (İsteğe bağlı) en büyük koyu bileşeni seç ---
        # Küçük adacıkları elemek için bağlantılı bileşen analizi
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
        mask = np.zeros_like(thr)
        if num > 1:
            # 0 index arka plan; en büyük foreground’u bul
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = np.argmax(areas) + 1
            # Minimum alan eşiği uygula
            if stats[max_idx, cv2.CC_STAT_AREA] >= self.min_area_px:
                mask[labels == max_idx] = 255
            # değilse maske boş kalır (debug için daha net)
        # hiç bileşen yoksa maske boş (siyah) kalır

        # Publish
        self.pub_gray.publish(self.bridge.cv2_to_imgmsg(gray, encoding="mono8"))
        if self.publish_thresh:
            self.pub_thr.publish(self.bridge.cv2_to_imgmsg(thr, encoding="mono8"))
        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))


def main():
    rospy.init_node("pipe_detector_simple")
    PipeDetectorSimple()
    rospy.spin()


if __name__ == "__main__":
    main()
