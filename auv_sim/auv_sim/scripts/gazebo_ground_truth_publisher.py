#!/usr/bin/env python3
"""
Ground Truth Object Pose Publisher
-----------------------------------
Gazebo simülasyonundaki herhangi bir modelin gerçek pozisyon ve oryantasyonunu
alıp TF frame olarak ve object_map_tf_server'a yayınlar.

Kullanım:
  rosrun auv_sim gazebo_ground_truth_publisher.py \
      _models:="tac_valve_alone::valve_stand_link"

  Ya da launch file'da:
    <param name="models" value="tac_valve_alone::valve_stand_link" />
    <param name="yaw_offset" value="1.5708" />   (90° döndür)

  Format: "model_name::tf_frame_name, model_name2::tf_frame_name2, ..."
  Kısa:   "model_name" → frame_name otomatik "model_name_gt_link" olur

Parametreler:
  ~models          : model_name::frame_name listesi
  ~yaw_offset      : Tüm modellere uygulanacak yaw offset (radyan)
  ~robot_name      : Robot model adı (default: taluy)
  ~odom_frame      : Odom frame adı (default: odom)
  ~rate            : Yayın hızı Hz (default: 20)

Gazebo model pozisyonu → odom frame'e dönüştürülür ve TF olarak yayınlanır.
Dönüşüm: tf_gazebo_debug.py ile aynı mantık.
"""

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.srv import GetModelState
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class GroundTruthPublisher:
    def __init__(self):
        rospy.init_node("gazebo_ground_truth_publisher")

        # Parametreler
        self.robot_name = rospy.get_param("~robot_name", "taluy")
        self.robot_base_frame = rospy.get_param(
            "~robot_base_frame", f"{self.robot_name}/base_link"
        )
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.rate_hz = rospy.get_param("~rate", 20)

        # Oryantasyon offset (yaw, radyan)
        self.yaw_offset = rospy.get_param("~yaw_offset", 0.0)

        # Model listesi: {model_name: frame_name}
        self.models = {}
        models_param = rospy.get_param("~models", "")
        if models_param:
            self._parse_models_param(models_param)

        # Gazebo servisi
        rospy.loginfo("Waiting for /gazebo/get_model_state...")
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Object map TF server (opsiyonel)
        self.set_object_transform = None
        try:
            rospy.wait_for_service("set_object_transform", timeout=5.0)
            self.set_object_transform = rospy.ServiceProxy(
                "set_object_transform", SetObjectTransform
            )
            rospy.loginfo("Connected to set_object_transform service")
        except rospy.ROSException:
            rospy.logwarn(
                "set_object_transform service not found, "
                "will only publish TF (no object map)"
            )

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Cached odom→world transform
        self._M_odom_in_world = None
        self._last_odom_update = rospy.Time(0)

        if self.yaw_offset != 0.0:
            rospy.loginfo(f"Yaw offset: {self.yaw_offset:.3f} rad ({np.degrees(self.yaw_offset):.1f} deg)")

        rospy.loginfo(
            f"Ground truth publisher started. "
            f"Models: {self.models if self.models else '(none)'}"
        )

    def _parse_models_param(self, param_str):
        """
        "model_name::frame_name, model2::frame2" formatını parse et.
        Kısa format: "model_name" → frame_name = "model_name_gt_link"
        """
        for item in param_str.split(","):
            item = item.strip()
            if not item:
                continue
            if "::" in item:
                model_name, frame_name = item.split("::", 1)
            else:
                model_name = item
                frame_name = f"{item}_gt_link"
            self.models[model_name.strip()] = frame_name.strip()
            rospy.loginfo(f"  Tracking: {model_name} -> {frame_name}")

    # =====================================================================
    #  GAZEBO → ODOM DÖNÜŞÜM
    # =====================================================================
    def _pose_to_matrix(self, position, orientation):
        """Pose → 4x4 homogeneous matrix."""
        T = tft.translation_matrix([position.x, position.y, position.z])
        R = tft.quaternion_matrix([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        return np.dot(T, R)

    def _update_odom_world_transform(self):
        """
        odom→world dönüşümünü güncelle.
        Robot'un Gazebo'daki (world frame) ve ROS'taki (odom frame)
        pozisyonlarının farkından hesaplanır.
        """
        now = rospy.Time.now()
        if (now - self._last_odom_update).to_sec() < 0.5 and self._M_odom_in_world is not None:
            return True

        try:
            gazebo_robot = self.get_model_state(
                model_name=self.robot_name, relative_entity_name="world"
            )
            if not gazebo_robot.success:
                return False

            ros_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_base_frame,
                rospy.Time(0), rospy.Duration(1.0)
            )

            M_robot_in_world = self._pose_to_matrix(
                gazebo_robot.pose.position, gazebo_robot.pose.orientation
            )
            M_robot_in_odom = self._pose_to_matrix(
                ros_robot.transform.translation, ros_robot.transform.rotation
            )

            self._M_odom_in_world = np.dot(
                M_robot_in_world, np.linalg.inv(M_robot_in_odom)
            )
            self._last_odom_update = now
            return True

        except Exception as e:
            rospy.logwarn_throttle(5.0, f"odom->world transform update failed: {e}")
            return False

    def _gazebo_pose_to_odom(self, position, orientation):
        """
        Gazebo world frame'deki pose'u odom frame'e dönüştür.
        Yaw offset uygular.
        """
        if not self._update_odom_world_transform():
            return None

        M_object_in_world = self._pose_to_matrix(position, orientation)

        # object_in_odom = inv(odom_in_world) * object_in_world
        M_object_in_odom = np.dot(
            np.linalg.inv(self._M_odom_in_world), M_object_in_world
        )

        # Yaw offset uygula
        if self.yaw_offset != 0.0:
            M_yaw = tft.euler_matrix(0, 0, self.yaw_offset)
            M_object_in_odom = np.dot(M_object_in_odom, M_yaw)

        trans = tft.translation_from_matrix(M_object_in_odom)
        quat = tft.quaternion_from_matrix(M_object_in_odom)

        return trans, quat

    # =====================================================================
    #  YAYINLA
    # =====================================================================
    def publish_model_pose(self, model_name, frame_name):
        """Tek bir modelin ground truth pozisyonunu yayınla."""
        try:
            state = self.get_model_state(
                model_name=model_name, relative_entity_name="world"
            )
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(10.0, f"GetModelState failed for {model_name}: {e}")
            return

        if not state.success:
            rospy.logwarn_throttle(
                10.0, f"Model '{model_name}' not found in Gazebo"
            )
            return

        result = self._gazebo_pose_to_odom(state.pose.position, state.pose.orientation)
        if result is None:
            return

        trans, quat = result

        # TF olarak yayınla
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = frame_name
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

        # Object map TF server'a da gönder
        if self.set_object_transform is not None:
            try:
                req = SetObjectTransformRequest()
                req.transform = t
                self.set_object_transform.call(req)
            except rospy.ServiceException:
                pass

    # =====================================================================
    #  ANA DÖNGÜ
    # =====================================================================
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            for model_name, frame_name in self.models.items():
                self.publish_model_pose(model_name, frame_name)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = GroundTruthPublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
