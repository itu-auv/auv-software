#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PipeHeadingTF
- pipe_mask'tan boru eksen açısını (image düzleminde) cv2.fitLine ile çıkarır
- base_link'e göre yalnızca yaw dönmüş bir child frame yayınlar (konum aynıdır)
- Hedef: child frame'in +x ekseni borunun İLERİ yönünü göstersin

Önemli:
- base_link +x görüntüde SOL ise (senin kurulumun), fitLine vektörü sağa bakıyorsa ters çevrilir.
- Böylece yön belirsizliği çözülür ve +pi eklemeye gerek kalmaz.
"""

import math
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler


def wrap_pi(a: float) -> float:
    """Açıyı [-pi, +pi] aralığına sar."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class PipeHeadingTF:
    def __init__(self):
        self.bridge = CvBridge()

        # --- Parametreler ---
        self.mask_topic   = rospy.get_param("~mask_topic", "pipe_mask")
        self.parent_frame = rospy.get_param("~parent_frame", "taluy/base_link")
        self.child_frame  = rospy.get_param("~child_frame",  "pipe_x")

        # Görüntü oryantasyonu (gerekirse kullan)
        self.rotate_cw = int(rospy.get_param("~rotate_cw", 0))   # 0,1,2,3 (×90° saat yönü)
        self.flip_x    = bool(rospy.get_param("~flip_x", False))
        self.flip_y    = bool(rospy.get_param("~flip_y", False))

        # Eşik
        self.min_pipe_area_px = int(rospy.get_param("~min_pipe_area_px", 1500))

        # Yön/işaret ayarları
        # Senaryon: base_link +x görüntüde SOL → True bırak.
        self.base_x_is_image_left = bool(rospy.get_param("~base_x_is_image_left", True))
        self.invert_angle         = bool(rospy.get_param("~invert_angle", False))
        self.extra_yaw_offset_rad = float(rospy.get_param("~extra_yaw_offset_rad", 0.0))
        # İstersen sabit 180° çevirme için:
        self.flip_180             = bool(rospy.get_param("~flip_180", False))

        # TF yayıncı ve abone
        self.br  = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber(self.mask_topic, Image, self.cb_mask, queue_size=1)

        rospy.loginfo("[pipe_heading_tf] parent=%s child=%s base_x_is_image_left=%s",
                      self.parent_frame, self.child_frame, self.base_x_is_image_left)

    # ---- Yardımcılar ----
    def _orient_img(self, m: np.ndarray) -> np.ndarray:
        r = self.rotate_cw % 4
        if r == 1:
            m = np.ascontiguousarray(np.rot90(m, k=3))
        elif r == 2:
            m = np.ascontiguousarray(np.rot90(m, k=2))
        elif r == 3:
            m = np.ascontiguousarray(np.rot90(m, k=1))
        if self.flip_x:
            m = np.ascontiguousarray(np.fliplr(m))
        if self.flip_y:
            m = np.ascontiguousarray(np.flipud(m))
        return m

    def _angle_from_mask(self, mask: np.ndarray) -> float or None:
        """
        1) En büyük konturu bul
        2) fitLine ile eksen vektörü (vx,vy)
        3) Vektörü 'ileri'ye zorla:
           - base_link +x görüntüde SOL ise → vx <= 0 olmalı (sağa bakıyorsa çevir)
           - base_link +x görüntüde SAĞ ise → vx >= 0 olmalı (sola bakıyorsa çevir)
        4) Açıyı [-pi/2, +pi/2] aralığına katla
        """
        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = _ret[1] if len(_ret) == 3 else _ret[0]
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 2:
            return None

        [vx, vy, x0, y0] = cv2.fitLine(cnt.reshape(-1, 2), cv2.DIST_L2, 0, 0.01, 0.01)

        # --- YÖNÜ İLERİYE SABİTLE ---
        if self.base_x_is_image_left:
            if vx > 0:      # sağa bakıyorsa çevir → sola baksın (ileri)
                vx, vy = -vx, -vy
        else:
            if vx < 0:      # sola bakıyorsa çevir → sağa baksın (ileri)
                vx, vy = -vx, -vy

        angle = math.atan2(float(vy), float(vx))  # 0 ~ ileri (görüntüde base_x yönü)

        # --- Doğru katlama: [-pi/2, +pi/2] ---
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi

        return angle

    # ---- Callback ----
    def cb_mask(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("cv_bridge: %s", e)
            return

        mask = self._orient_img(mask)

        # Basit geçerlilik
        if int(np.count_nonzero(mask)) < self.min_pipe_area_px:
            return

        angle_img = self._angle_from_mask(mask)
        if angle_img is None:
            return

        # Yaw hesabı:
        # Yön sabitlendiği için +pi EKLEMEYE GEREK YOK; angle_img zaten 'ileri' sapması.
        yaw_rel = -angle_img

        if self.flip_180:
            yaw_rel = wrap_pi(yaw_rel + math.pi)
        if self.invert_angle:
            yaw_rel = -yaw_rel
        yaw_rel = wrap_pi(yaw_rel + self.extra_yaw_offset_rad)

        # TF: parent=base_link, child=pipe_x; sadece yaw, translasyon = 0
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame
        t.child_frame_id  = self.child_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_rel)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.br.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("pipe_heading_tf")
    PipeHeadingTF()
    rospy.spin()
