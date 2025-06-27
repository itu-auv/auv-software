#!/usr/bin/env python3

import numpy as np

from typing import Tuple
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class BinTransformServiceNode:
    def __init__(self):
        self.enable = False
        rospy.init_node("create_bin_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.robot_frame = "taluy/base_link"
        self.bin_frame = "bin_whole_link"

        self.bin_further_frame = "bin_far_trial"
        self.bin_closer_frame = "bin_close_trial"

        self.further_distance = rospy.get_param("~further_distance", 2.5)
        self.closer_distance = rospy.get_param("~closer_distance", 1.0)
        self.second_trial_distance = rospy.get_param("~second_trial_distance", 2.5)

        self.set_enable_service = rospy.Service(
            "set_transform_bin_frames", SetBool, self.handle_enable_service
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
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logwarn(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def create_bin_frames(self):
        try:
            transform_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(1)
            )
            transform_bin = self.tf_buffer.lookup_transform(
                self.odom_frame, self.bin_frame, rospy.Time(0), rospy.Duration(1)
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

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

        # Calculate yaw from the direction vector (robot to bin)
        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        orientation = Pose().orientation
        orientation.x = q[0]
        orientation.y = q[1]
        orientation.z = q[2]
        orientation.w = q[3]

        closer_pos_2d = bin_pos[:2] - (direction_unit_2d * self.closer_distance)
        further_pos_2d = bin_pos[:2] + (direction_unit_2d * self.further_distance)

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

        # Calculate position for bin_second_trial frame
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
        rate = rospy.Rate(2.0)
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
