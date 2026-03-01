#!/usr/bin/env python3
"""
Valve Trajectory Publisher
--------------------------
valve_stand_link TF'ini okur ve üç yeni frame oluşturur:
  1. valve_coarse_approach_frame → Robot→valve yönünde, coarse_approach_offset metre
     uzakta (oryantasyonsuz, sadece yaklaşma)
  2. valve_approach_frame → Vananın yüzey normali yönünde, approach_offset metre
     uzakta (oryantasyonlu yaklaşma)
  3. valve_contact_frame → Vananın yüzey normali yönünde, contact_offset metre
     uzakta (temas noktası)

3 Aşamalı Yaklaşım:
  Phase 1: Coarse approach — robot→valve yönünden kaba yaklaşma (~2.5m)
  Phase 2: Oriented approach — yüzey normaline dik yaklaşma (~1.5m)
  Phase 3: Contact — yüzey normaline dik temas (~0.625m)

Pattern: torpedo_frame_publisher.py ile aynı mimari.
  - SetBool servisleri ile enable/disable
  - Dynamic reconfigure ile offset ayarı
  - 20Hz döngüde TF güncelleme
"""

import numpy as np
import tf.transformations
from tf.transformations import quaternion_matrix
import rospy
import tf2_ros
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from dynamic_reconfigure.server import Server

from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
)
from auv_mapping.cfg import ValveTrajectoryConfig


class ValveTrajectoryPublisherNode:
    def __init__(self):
        rospy.init_node("valve_trajectory_publisher_node")

        self.enable_coarse_approach = False
        self.enable_approach = False
        self.enable_contact = False

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Object transform service (object_map_tf_server'a yazar)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        # Frame isimleri
        self.odom_frame = "odom"
        self.robot_frame = "taluy/base_link"

        # Kaynak frame (valve_detector.py tarafından yayınlanır)
        self.valve_frame = rospy.get_param(
            "~valve_frame", "valve_stand_link"
        )

        # Hedef frame isimleri
        self.coarse_approach_frame = rospy.get_param(
            "~coarse_approach_frame", "valve_coarse_approach_frame"
        )
        self.approach_frame = rospy.get_param(
            "~approach_frame", "valve_approach_frame"
        )
        self.contact_frame = rospy.get_param(
            "~contact_frame", "valve_contact_frame"
        )

        # Default offset değerleri (dynamic reconfigure ile değiştirilir)
        self.coarse_approach_offset = 1.75  # metre
        self.approach_offset = 1.25         # metre
        self.contact_offset = 0.625        # metre

        # Dynamic reconfigure server
        self.reconfigure_server = Server(
            ValveTrajectoryConfig, self.reconfigure_callback
        )

        # Enable/disable servisleri
        self.set_enable_coarse_approach_service = rospy.Service(
            "set_transform_valve_coarse_approach_frame",
            SetBool,
            self.handle_enable_coarse_approach_service,
        )
        self.set_enable_approach_service = rospy.Service(
            "set_transform_valve_approach_frame",
            SetBool,
            self.handle_enable_approach_service,
        )
        self.set_enable_contact_service = rospy.Service(
            "set_transform_valve_contact_frame",
            SetBool,
            self.handle_enable_contact_service,
        )

        rospy.loginfo("Valve trajectory publisher node started (3-phase)")

    # =====================================================================
    #  YARDIMCI METODLAR
    # =====================================================================
    def get_pose(self, transform: TransformStamped) -> Pose:
        """TransformStamped -> Pose dönüşümü."""
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        """Pose'dan TransformStamped mesajı oluştur."""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform):
        """Object map TF server'a transform gönder."""
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def get_valve_surface_normal_2d(self, valve_tf):
        """
        Valve TF'in oryantasyonundan yüzey normalinin XY bileşenini çıkar.
        valve_detection.py, valve_stand_link'in x-eksenini yüzey normali
        yönünde ayarlar. Bu fonksiyon o yönü odom frame'de döndürür.

        Returns: (normal_2d, yaw) veya None
            normal_2d = np.array([nx, ny]) birim vektör
            yaw = yüzey normaline bakan yaw açısı (ters yön)
        """
        q = [
            valve_tf.transform.rotation.x,
            valve_tf.transform.rotation.y,
            valve_tf.transform.rotation.z,
            valve_tf.transform.rotation.w,
        ]

        # Rotation matrisinden x-ekseni (yüzey normali) yönünü al
        rot_matrix = quaternion_matrix(q)
        surface_normal_3d = rot_matrix[:3, 0]  # x-ekseni = yüzey normali

        # XY düzlemine projekte et
        normal_2d = surface_normal_3d[:2]
        norm = np.linalg.norm(normal_2d)

        if norm < 1e-6:
            rospy.logwarn_throttle(5.0, "Valve surface normal has no XY component!")
            return None

        normal_2d = normal_2d / norm

        # Yüzey normalinin TERSİ yönüne bakan yaw (vanaya doğru)
        # Normal kameraya doğru bakıyor, biz vanaya doğru bakmak istiyoruz
        yaw = np.arctan2(-normal_2d[1], -normal_2d[0])

        return normal_2d, yaw

    # =====================================================================
    #  PHASE 1: COARSE APPROACH (oryantasyonsuz, robot→valve yönü)
    # =====================================================================
    def create_coarse_approach_frame(self):
        """
        Robot→valve yönünde coarse_approach_offset metre uzakta,
        vanaya bakan bir frame oluştur.
        Oryantasyon bilgisi KULLANILMAZ — sadece kaba yaklaşma.
        """
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame,
                rospy.Time(0), rospy.Duration(4.0)
            )
            valve_tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.valve_frame,
                rospy.Time(0), rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Valve TF lookup failed: {e}")
            return

        robot_pose = self.get_pose(robot_tf)
        valve_pose = self.get_pose(valve_tf)

        robot_pos = np.array([
            robot_pose.position.x, robot_pose.position.y, robot_pose.position.z
        ])
        valve_pos = np.array([
            valve_pose.position.x, valve_pose.position.y, valve_pose.position.z
        ])

        # Robot → Valve yön vektörü (XY düzleminde)
        direction_vector_2d = valve_pos[:2] - robot_pos[:2]
        total_distance_2d = np.linalg.norm(direction_vector_2d)

        if total_distance_2d == 0:
            rospy.logwarn_throttle(
                5.0, "Robot and valve at same XY position!"
            )
            return

        direction_unit_2d = direction_vector_2d / total_distance_2d

        # Vanaya bakan yaw açısı
        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        # Coarse approach pozisyonu: vanadan geriye
        coarse_pos_2d = valve_pos[:2] - (direction_unit_2d * self.coarse_approach_offset)

        coarse_pose = Pose()
        coarse_pose.position.x = coarse_pos_2d[0]
        coarse_pose.position.y = coarse_pos_2d[1]
        coarse_pose.position.z = valve_pos[2]  # Vana ile aynı derinlik
        coarse_pose.orientation.x = q[0]
        coarse_pose.orientation.y = q[1]
        coarse_pose.orientation.z = q[2]
        coarse_pose.orientation.w = q[3]

        coarse_transform = self.build_transform_message(
            self.coarse_approach_frame, coarse_pose
        )
        self.send_transform(coarse_transform)

    # =====================================================================
    #  PHASE 2: ORIENTED APPROACH (yüzey normaline dik)
    # =====================================================================
    def create_approach_frame(self):
        """
        valve_stand_link'in yüzey normali yönünde approach_offset metre
        geriye çekilmiş, vanaya dik bakan bir frame oluştur.
        """
        try:
            valve_tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.valve_frame,
                rospy.Time(0), rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Valve TF lookup failed: {e}")
            return

        valve_pose = self.get_pose(valve_tf)
        valve_pos = np.array([
            valve_pose.position.x, valve_pose.position.y, valve_pose.position.z
        ])

        # Valve yüzey normalinden approach yönünü hesapla
        result = self.get_valve_surface_normal_2d(valve_tf)
        if result is None:
            return
        normal_2d, yaw = result

        # Approach pozisyonu: vanadan yüzey normali yönünde approach_offset metre
        approach_pos_2d = valve_pos[:2] + (normal_2d * self.approach_offset)

        approach_pose = Pose()
        approach_pose.position.x = approach_pos_2d[0]
        approach_pose.position.y = approach_pos_2d[1]
        approach_pose.position.z = valve_pos[2]  # Vana ile aynı derinlik

        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        approach_pose.orientation.x = q[0]
        approach_pose.orientation.y = q[1]
        approach_pose.orientation.z = q[2]
        approach_pose.orientation.w = q[3]

        approach_transform = self.build_transform_message(
            self.approach_frame, approach_pose
        )
        self.send_transform(approach_transform)

    # =====================================================================
    #  PHASE 3: CONTACT (yüzey normaline dik, temas mesafesi)
    # =====================================================================
    def create_contact_frame(self):
        """
        valve_stand_link'in yüzey normali yönünde contact_offset metre
        geriye çekilmiş, vanaya dik bakan bir frame oluştur. Kavrama mesafesi.
        """
        try:
            valve_tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.valve_frame,
                rospy.Time(0), rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return

        valve_pose = self.get_pose(valve_tf)
        valve_pos = np.array([
            valve_pose.position.x, valve_pose.position.y, valve_pose.position.z
        ])

        # Valve yüzey normalinden contact yönünü hesapla
        result = self.get_valve_surface_normal_2d(valve_tf)
        if result is None:
            return
        normal_2d, yaw = result

        # Contact pozisyonu: vanadan yüzey normali yönünde contact_offset metre
        contact_pos_2d = valve_pos[:2] + (normal_2d * self.contact_offset)

        contact_pose = Pose()
        contact_pose.position.x = contact_pos_2d[0]
        contact_pose.position.y = contact_pos_2d[1]
        contact_pose.position.z = valve_pos[2]  # Vana ile aynı derinlik

        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        contact_pose.orientation.x = q[0]
        contact_pose.orientation.y = q[1]
        contact_pose.orientation.z = q[2]
        contact_pose.orientation.w = q[3]

        contact_transform = self.build_transform_message(
            self.contact_frame, contact_pose
        )
        self.send_transform(contact_transform)

    # =====================================================================
    #  SERVİS HANDLER'LAR
    # =====================================================================
    def handle_enable_coarse_approach_service(self, req):
        self.enable_coarse_approach = req.data
        message = f"Valve coarse approach frame publish is set to: {self.enable_coarse_approach}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_enable_approach_service(self, req):
        self.enable_approach = req.data
        message = f"Valve approach frame publish is set to: {self.enable_approach}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_enable_contact_service(self, req):
        self.enable_contact = req.data
        message = f"Valve contact frame publish is set to: {self.enable_contact}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    # =====================================================================
    #  DYNAMIC RECONFIGURE
    # =====================================================================
    def reconfigure_callback(self, config, level):
        """Dynamic reconfigure ile offset değerlerini güncelle."""
        self.coarse_approach_offset = config.coarse_approach_offset
        self.approach_offset = config.approach_offset
        self.contact_offset = config.contact_offset
        rospy.loginfo(
            f"Valve trajectory params updated: "
            f"coarse={self.coarse_approach_offset:.2f}m, "
            f"approach={self.approach_offset:.2f}m, "
            f"contact={self.contact_offset:.2f}m"
        )
        return config

    # =====================================================================
    #  ANA DÖNGÜ
    # =====================================================================
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.enable_coarse_approach:
                self.create_coarse_approach_frame()
            if self.enable_approach:
                self.create_approach_frame()
            if self.enable_contact:
                self.create_contact_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ValveTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
