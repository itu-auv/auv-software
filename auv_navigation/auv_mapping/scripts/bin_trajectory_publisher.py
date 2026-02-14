#!/usr/bin/env python3

import numpy as np

from typing import Tuple
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse


class BinTransformServiceNode:
    def __init__(self):
        self.enable = False
        rospy.init_node("create_bin_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_pub = rospy.Publisher(
            "set_object_transform", TransformStamped, queue_size=10
        )

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.bin_frame = rospy.get_param("~bin_frame", "bin_whole_link")
        self.bin_further_frame = rospy.get_param("~bin_further_frame", "bin_far_trial")
        self.bin_closer_frame = rospy.get_param(
            "~bin_closer_frame", "bin_close_approach"
        )
        self.second_trial_frame = rospy.get_param(
            "~second_trial_frame", "bin_second_trial"
        )

        self.closer_frame_distance = rospy.get_param("~closer_frame_distance", 1.0)
        self.further_frame_distance = rospy.get_param("~further_frame_distance", 2.5)
        self.second_trial_distance = rospy.get_param("~second_trial_distance", 2.5)

        self.bin_whole_estimated_frame = rospy.get_param(
            "~bin_whole_estimated_frame", "bin_whole_estimated"
        )
        self.bin_exit_frame = rospy.get_param("~bin_exit_frame", "bin_exit")

        self.set_enable_service = rospy.Service(
            "toggle_bin_trajectory", SetBool, self.handle_enable_service
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

    def send_transform(self, transform):
        try:
            self.set_object_transform_pub.publish(transform)
        except Exception as e:
            rospy.logerr(f"Failed to publish transform: {e}")

    def create_bin_frames(self):
        try:
            transform_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(4.0)
            )
            transform_bin = self.tf_buffer.lookup_transform(
                self.odom_frame, self.bin_frame, rospy.Time.now(), rospy.Duration(4.0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        # Try to get gate_exit transform, if not available use default values
        gate_exit_pose = None
        try:
            transform_gate_exit = self.tf_buffer.lookup_transform(
                self.odom_frame,
                "torpedo_target_realsense",
                rospy.Time.now(),
                rospy.Duration(1),
            )
            gate_exit_pose = self.get_pose(transform_gate_exit)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                10.0,
                f"gate_exit frame not found: {e}. Using default values for bin_exit frame.",
            )

        robot_pose = self.get_pose(transform_robot)
        bin_pose = self.get_pose(transform_bin)

        robot_pos = np.array(
            [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
        )
        bin_pos = np.array(
            [bin_pose.position.x, bin_pose.position.y, bin_pose.position.z]
        )

        direction_vector_2d = bin_pos[:2] - robot_pos[:2]
        total_distance_2d = np.linalg.norm(direction_vector_2d)

        if total_distance_2d == 0:
            rospy.logwarn(
                "Robot and bin are at the same XY position! Cannot create frames."
            )
            return

        direction_unit_2d = direction_vector_2d / total_distance_2d

        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        orientation = Pose().orientation
        orientation.x = q[0]
        orientation.y = q[1]
        orientation.z = q[2]
        orientation.w = q[3]

        closer_pos_2d = bin_pos[:2] - (direction_unit_2d * self.closer_frame_distance)
        further_pos_2d = bin_pos[:2] + (direction_unit_2d * self.further_frame_distance)

        closer_pos = np.append(closer_pos_2d, robot_pos[2])
        further_pos = np.append(further_pos_2d, robot_pos[2])

        closer_pose = Pose()
        closer_pose.position.x, closer_pose.position.y, closer_pose.position.z = (
            closer_pos
        )
        closer_pose.orientation = orientation

        further_pose = Pose()
        further_pose.position.x, further_pose.position.y, further_pose.position.z = (
            further_pos
        )
        further_pose.orientation = orientation

        further_transform = self.build_transform_message(
            self.bin_further_frame, further_pose
        )
        closer_transform = self.build_transform_message(
            self.bin_closer_frame, closer_pose
        )

        self.send_transform(further_transform)
        self.send_transform(closer_transform)

        bin_whole_estimated_pose = Pose()
        bin_whole_estimated_pose.position.x = bin_pose.position.x
        bin_whole_estimated_pose.position.y = bin_pose.position.y
        bin_whole_estimated_pose.position.z = robot_pose.position.z
        bin_whole_estimated_pose.orientation = orientation
        bin_whole_estimated_transform = self.build_transform_message(
            self.bin_whole_estimated_frame, bin_whole_estimated_pose
        )
        self.send_transform(bin_whole_estimated_transform)

        bin_exit_pose = Pose()
        bin_exit_pose.position.x = bin_whole_estimated_pose.position.x
        bin_exit_pose.position.y = bin_whole_estimated_pose.position.y

        # If gate_exit frame is available, use its Z position and orientation
        # Otherwise, use robot's Z position and default (odom frame) orientation
        if gate_exit_pose is not None:
            bin_exit_pose.position.z = gate_exit_pose.position.z
            bin_exit_pose.orientation = gate_exit_pose.orientation
        else:
            bin_exit_pose.position.z = robot_pose.position.z
            # Default orientation (same as odom frame - no rotation)
            bin_exit_pose.orientation.x = 0.0
            bin_exit_pose.orientation.y = 0.0
            bin_exit_pose.orientation.z = 0.0
            bin_exit_pose.orientation.w = 1.0

        bin_exit_transform = self.build_transform_message(
            self.bin_exit_frame, bin_exit_pose
        )
        self.send_transform(bin_exit_transform)

        perp_direction_unit_2d = np.array([-direction_unit_2d[1], direction_unit_2d[0]])

        second_trial_pos_2d = bin_pos[:2] - (
            perp_direction_unit_2d * self.second_trial_distance
        )
        second_trial_pos = np.append(second_trial_pos_2d, robot_pos[2])

        second_trial_yaw = yaw + np.pi / 2.0
        second_trial_q = tf.transformations.quaternion_from_euler(
            0, 0, second_trial_yaw
        )

        second_trial_pose = Pose()
        (
            second_trial_pose.position.x,
            second_trial_pose.position.y,
            second_trial_pose.position.z,
        ) = second_trial_pos
        second_trial_pose.orientation.x = second_trial_q[0]
        second_trial_pose.orientation.y = second_trial_q[1]
        second_trial_pose.orientation.z = second_trial_q[2]
        second_trial_pose.orientation.w = second_trial_q[3]
        self.send_transform(
            self.build_transform_message("bin_second_trial", second_trial_pose)
        )

    def handle_enable_service(self, req):
        self.enable = req.data
        message = f"Bin target frames transform publish is set to: {self.enable}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if self.enable:
                self.create_bin_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = BinTransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
