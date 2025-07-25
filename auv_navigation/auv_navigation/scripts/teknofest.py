#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from dynamic_reconfigure.server import Server
from auv_navigation.cfg import PipeFollower as ConfigType


class PipeFollower:
    def __init__(self):
        rospy.init_node("pipe_follower_node")

        # ROS iletişim kurulumu
        self.bridge = CvBridge()
        self.twist_pub = rospy.Publisher("/taluy/cmd_vel", Twist, queue_size=10)
        self.debug_pub = rospy.Publisher(
            "/pipe_follower/debug_image", Image, queue_size=10
        )

        # Görüntü aboneliği (compressed image için)
        self.image_sub = rospy.Subscriber(
            "/taluy/cameras/cam_front/image_rect_color/compressed",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        # Kontrol değişkenleri
        self.twist_msg = Twist()
        self.min_contour_area = 100  # Gürültü filtreleme için minimum kontur alanı

        # PID benzeri kontrol katsayıları
        self.kp_angle = 0.01
        self.kp_position = 0.005
        self.base_speed = 0.3

        # HSV Thresholds
        self.lower_h = 20
        self.lower_s = 100
        self.lower_v = 100
        self.upper_h = 30
        self.upper_s = 255
        self.upper_v = 255

        # Dinamik reconfig ayarları
        self.reconfig_server = Server(ConfigType, self.reconfigure_cb)
        rospy.loginfo("Pipe Follower node başlatıldı")

    def reconfigure_cb(self, config, level):
        self.lower_h = config.lower_h
        self.lower_s = config.lower_s
        self.lower_v = config.lower_v
        self.upper_h = config.upper_h
        self.upper_s = config.upper_s
        self.upper_v = config.upper_v
        return config

    def estimate_pipe_direction(self, contour):
        """Borunun yön açısını tahmin et"""
        rect = cv2.minAreaRect(contour)
        angle = rect[2]

        # minAreaRect açıları 0-90 arasında döner, düzeltme yapıyoruz
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        return angle

    def calculate_control(self, angle_error, position_error, img_width):
        """Hata değerlerine göre kontrol komutlarını hesapla"""
        # Normalize hatalar
        norm_angle_error = angle_error / 90.0  # -1 ile 1 arasında
        norm_position_error = position_error / (img_width / 2.0)  # -1 ile 1 arasında

        # PID benzeri kontrol
        angular_z = -(
            self.kp_position * norm_position_error + self.kp_angle * norm_angle_error
        )
        linear_x = self.base_speed * (
            1.0 - min(abs(norm_angle_error), 0.5)
        )  # Büyük açı hatasında yavaşla

        return linear_x, angular_z

    def image_callback(self, msg):
        try:
            # Compressed image decode
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            h, w = cv_image.shape[:2]

            # ROI (Region of Interest) belirle - Görüntünün alt %60'ını al, üst kısmı gözardı et
            roi = cv_image[int(h * 0.4) : h, :]
            roi_h, roi_w = roi.shape[:2]

            # HSV'ye çevir ve threshold uygula
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([self.lower_h, self.lower_s, self.lower_v])
            upper_bound = np.array([self.upper_h, self.upper_s, self.upper_v])
            thresh = cv2.inRange(hsv, lower_bound, upper_bound)

            # Morfolojik işlemler (gürültüyü azaltmak için)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Kontur bulma
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Debug görüntüsü oluştur
            debug_img = roi.copy()

            if len(contours) > 0:
                # En büyük konturu seç
                main_contour = max(contours, key=cv2.contourArea)

                if cv2.contourArea(main_contour) > self.min_contour_area:
                    # Boru yönünü tahmin et
                    angle = self.estimate_pipe_direction(main_contour)

                    # Boru merkezini bul
                    M = cv2.moments(main_contour)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else roi_w // 2
                    cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

                    # Hataları hesapla
                    position_error = cx - roi_w // 2
                    angle_error = angle

                    # Kontrol komutlarını hesapla
                    linear_x, angular_z = self.calculate_control(
                        angle_error, position_error, roi_w
                    )

                    # Twist mesajını güncelle
                    self.twist_msg.linear.x = linear_x
                    self.twist_msg.angular.z = angular_z

                    # Debug çizimleri
                    cv2.drawContours(debug_img, [main_contour], -1, (0, 255, 0), 2)
                    cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(
                        debug_img,
                        f"Angle: {angle:.1f}deg",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        debug_img,
                        f"PosErr: {position_error}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                else:
                    # Boru çok küçük, arama modu
                    self.twist_msg.linear.x = 0.1
                    self.twist_msg.angular.z = 0.3
                    cv2.putText(
                        debug_img,
                        "SEARCHING...",
                        (roi_w // 2 - 100, roi_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
            else:
                # Boru bulunamadı, arama modu
                self.twist_msg.linear.x = 0.1
                self.twist_msg.angular.z = 0.3
                cv2.putText(
                    debug_img,
                    "NO PIPE DETECTED",
                    (roi_w // 2 - 150, roi_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Debug görüntüsünü yayınla
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"Görüntü işleme hatası: {str(e)}")
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = 0.0

        # Kontrol komutlarını yayınla
        self.twist_pub.publish(self.twist_msg)


if __name__ == "__main__":
    try:
        PipeFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
