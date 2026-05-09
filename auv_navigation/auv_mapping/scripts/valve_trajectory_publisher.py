#!/usr/bin/env python3
import numpy as np
import rospy
import tf.transformations
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_matrix

from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
)

from auv_mapping.cfg import ValveTrajectoryConfig


class ValveTrajectoryPublisherNode:
    def __init__(self):
        rospy.init_node("valve_trajectory_publisher_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.valve_frame = rospy.get_param("~valve_frame", "tac/valve")
        self.approach_frame = rospy.get_param("~approach_frame", "valve_approach_frame")
        self.contact_frame = rospy.get_param("~contact_frame", "valve_contact_frame")

        self.enable_approach = False
        self.enable_contact = False

        self.approach_offset = 2.00
        self.contact_offset = 0.625

        self.reconfigure_server = Server(
            ValveTrajectoryConfig, self.reconfigure_callback
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

    def get_pose(self, transform: TransformStamped) -> Pose:
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def get_valve_surface_normal_3d(self, valve_tf):
        q = [
            valve_tf.transform.rotation.x,
            valve_tf.transform.rotation.y,
            valve_tf.transform.rotation.z,
            valve_tf.transform.rotation.w,
        ]

        rot_matrix = quaternion_matrix(q)
        normal_3d = rot_matrix[:3, 0].copy()

        norm = np.linalg.norm(normal_3d)
        if norm < 1e-6:
            rospy.logwarn_throttle(5.0, "Valve surface normal is degenerate!")
            return None

        normal_3d = normal_3d / norm
        facing_dir = -normal_3d

        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(facing_dir, up_hint)) > 0.99:
            up_hint = np.array([0.0, 1.0, 0.0])

        y_axis = np.cross(up_hint, facing_dir)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(facing_dir, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        rot = np.eye(4)
        rot[:3, 0] = facing_dir
        rot[:3, 1] = y_axis
        rot[:3, 2] = z_axis

        orientation_quat = tf.transformations.quaternion_from_matrix(rot)

        return normal_3d, orientation_quat

    def send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def _create_offset_frame(self, child_frame_id: str, offset_distance: float):
        try:
            valve_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.valve_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Valve TF lookup failed: {e}")
            return

        valve_pose = self.get_pose(valve_tf)
        valve_pos = np.array(
            [valve_pose.position.x, valve_pose.position.y, valve_pose.position.z]
        )

        result = self.get_valve_surface_normal_3d(valve_tf)
        if result is None:
            return
        normal_3d, orientation_quat = result

        target_pos = valve_pos + (normal_3d * offset_distance)

        pose = Pose()
        pose.position.x = target_pos[0]
        pose.position.y = target_pos[1]
        pose.position.z = target_pos[2]

        pose.orientation.x = orientation_quat[0]
        pose.orientation.y = orientation_quat[1]
        pose.orientation.z = orientation_quat[2]
        pose.orientation.w = orientation_quat[3]

        transform = self.build_transform_message(child_frame_id, pose)
        self.send_transform(transform)

    def create_approach_frame(self):
        self._create_offset_frame(self.approach_frame, self.approach_offset)

    def create_contact_frame(self):
        self._create_offset_frame(self.contact_frame, self.contact_offset)

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

    def reconfigure_callback(self, config, level):
        self.approach_offset = config.approach_offset
        self.contact_offset = config.contact_offset
        return config

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
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
