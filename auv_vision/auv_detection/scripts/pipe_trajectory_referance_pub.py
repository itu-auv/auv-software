#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

def wrap_pi(a):
    """[-pi, +pi] aralığına sar."""
    return (a + math.pi) % (2*math.pi) - math.pi

class PipeHeadingTF:
    """
    - pipe_mask'tan boru eksen açısını (image düzleminde) hesaplar
    - base_link'e göre sadece yaw döndürülmüş bir frame yayınlar:
        parent: base_link
        child:  pipe_x
        T = [0,0,0], Rz = yaw_rel
    Not: Odom'a göre pozisyonu otomatik olarak base_link ile aynı olur (tf zinciri).
    """

    def __init__(self):
        self.bridge = CvBridge()

        # Giriş/çıkış
        self.mask_topic = rospy.get_param("~mask_topic", "pipe_mask")
        self.parent_frame = rospy.get_param("~parent_frame", "taluy/base_link")
        self.child_frame  = rospy.get_param("~child_frame",  "pipe_x")

        # Görüntü oryantasyonu (gerekirse kullan)
        self.rotate_cw = rospy.get_param("~rotate_cw", 0)    # 0,1,2,3 (×90° saat yönü)
        self.flip_x    = rospy.get_param("~flip_x", False)
        self.flip_y    = rospy.get_param("~flip_y", False)

        # Eşikler
        self.min_pipe_area_px = int(rospy.get_param("~min_pipe_area_px", 1500))

        # İşaret/offset ayarları
        # Kullanıcının verdiği bilgi: base_link +x, görüntüde "sol" tarafa denk geliyor.
        # Bu durumda, image +x (sağ) eksenine göre ölçülen açıya PI eklemek gerekir.
        self.base_x_is_image_left = rospy.get_param("~base_x_is_image_left", True)
        self.extra_yaw_offset_rad = float(rospy.get_param("~extra_yaw_offset_rad", 0.0))
        self.invert_angle = bool(rospy.get_param("~invert_angle", False))  # boru yönünü ters almak istersen

        # Yayıncılar / abonelik
        self.br = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber(self.mask_topic, Image, self.cb_mask, queue_size=1)

        rospy.loginfo("[pipe_heading_tf] parent=%s child=%s base_x_is_image_left=%s",
                      self.parent_frame, self.child_frame, self.base_x_is_image_left)

    def _orient_img(self, m):
        if self.rotate_cw % 4 == 1:
            m = np.ascontiguousarray(np.rot90(m, k=3))
        elif self.rotate_cw % 4 == 2:
            m = np.ascontiguousarray(np.rot90(m, k=2))
        elif self.rotate_cw % 4 == 3:
            m = np.ascontiguousarray(np.rot90(m, k=1))
        if self.flip_x:
            m = np.ascontiguousarray(np.fliplr(m))
        if self.flip_y:
            m = np.ascontiguousarray(np.flipud(m))
        return m

    def _angle_from_mask(self, mask):
        """
        cv2.fitLine ile boru eksen vektörünü bul.
        angle: image düzleminde +x (sağ) eksenine göre açı (radyan).
        """
        # En büyük konturu al
        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(_ret) == 3:
            _img_out, contours, _hier = _ret
        else:
            contours, _hier = _ret
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 2:
            return None

        [vx, vy, x0, y0] = cv2.fitLine(cnt.reshape(-1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
        angle = math.atan2(float(vy), float(vx))  # 0 ~ yatay (sağa doğru)
        # [-pi/2, +pi/2] aralığına katla (flip sıçramalarını önlemek için)
        if angle > math.pi/2: angle -= math.pi
        if angle < -math.pi/2: angle += math.pi
        return angle

    def cb_mask(self, msg):
        # Maskeyi al
        rospy.loginfo_once("Received mask message")
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("cv_bridge: %s", e)
            return

        mask = self._orient_img(mask)

        # Geçerlilik
        if int(np.count_nonzero(mask)) < self.min_pipe_area_px:
            return

        angle_img = self._angle_from_mask(mask)
        if angle_img is None:
            return

        # base_link +x görüntüde "sol" ise, image +x (sağ) referanslı açıya PI ekle
        yaw_rel = angle_img + (math.pi if self.base_x_is_image_left else 0.0)

        # İsteğe bağlı tersleme/offset
        if self.invert_angle:
            yaw_rel = -yaw_rel
        yaw_rel = wrap_pi(yaw_rel + self.extra_yaw_offset_rad)

        # TF: parent=base_link, child=pipe_x; ÇEVİRİ: sadece yaw, translasyon yok
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
        print(t)
        self.br.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("pipe_heading_tf")
    PipeHeadingTF()
    rospy.spin()
