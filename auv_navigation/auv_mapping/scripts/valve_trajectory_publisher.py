#!/usr/bin/env python3
"""Publishes a target TF frame for valve alignment.

Simple geometry: position is offset along the valve's normal (+X of the
valve frame), orientation faces the valve with roll=0 and pitch=0 in
odom — i.e. the AUV is perfectly level, yaw points at the valve.

Since all three Euler angles are fully determined (roll=0, pitch=0,
yaw=atan2 of the valve normal), there is no leftover degree of freedom
and no need for shortest-arc rotations or snapshot tricks.
"""
import numpy as np
import rospy
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_from_euler, quaternion_matrix

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
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
        self.target_frame = rospy.get_param("~target_frame", "valve_target")

        self.enable_publishing = False

        # Distance from the valve along its flange-normal (+X of the valve
        # TF). Tuned per-phase via dynamic reconfigure.
        self.approach_offset = rospy.get_param("~approach_offset", 2.0)

        self.reconfigure_server = Server(
            ValveTrajectoryConfig, self.reconfigure_callback
        )

        self.set_enable_publishing_service = rospy.Service(
            "set_publishing",
            SetBool,
            self.handle_enable_publishing_service,
        )

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

    def send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: "
                f"{resp.message}"
            )

    def _compute_target(self, valve_tf):
        """Position offset along valve normal, orientation = level + facing valve."""
        # Valve position and normal (+X of valve frame) in odom.
        valve_pos = np.array(
            [
                valve_tf.transform.translation.x,
                valve_tf.transform.translation.y,
                valve_tf.transform.translation.z,
            ]
        )
        q_valve = [
            valve_tf.transform.rotation.x,
            valve_tf.transform.rotation.y,
            valve_tf.transform.rotation.z,
            valve_tf.transform.rotation.w,
        ]
        R_valve = quaternion_matrix(q_valve)[:3, :3]
        valve_normal = R_valve[:, 0]  # +X axis of valve frame

        # Target position: offset along the valve's outward normal.
        target_pos = valve_pos + valve_normal * self.approach_offset

        # Yaw: face the valve. The direction from target to valve is
        # -valve_normal (projected onto XY). So yaw = atan2(-ny, -nx).
        yaw = np.arctan2(-valve_normal[1], -valve_normal[0])

        # Perfectly level: roll=0, pitch=0.
        q_target = quaternion_from_euler(0.0, 0.0, yaw)

        pose = Pose()
        pose.position.x = float(target_pos[0])
        pose.position.y = float(target_pos[1])
        pose.position.z = float(target_pos[2])
        pose.orientation.x = float(q_target[0])
        pose.orientation.y = float(q_target[1])
        pose.orientation.z = float(q_target[2])
        pose.orientation.w = float(q_target[3])
        return pose

    def publish_target_frame(self):
        try:
            valve_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.valve_frame,
                rospy.Time(0),
                rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Valve TF lookup failed: {e}")
            return

        pose = self._compute_target(valve_tf)
        transform = self.build_transform_message(self.target_frame, pose)
        self.send_transform(transform)

    def handle_enable_publishing_service(self, req):
        self.enable_publishing = req.data
        message = (
            f"Valve frame publishing ({self.target_frame}) set to: "
            f"{self.enable_publishing}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def reconfigure_callback(self, config, level):
        self.approach_offset = config.approach_offset
        return config

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.enable_publishing:
                self.publish_target_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ValveTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
