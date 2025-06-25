#!/usr/bin/env python3

import math
import numpy as np
from tf.transformations import (
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_matrix,
)
from typing import Tuple
import rospy
import tf2_ros
from geometry_msgs.msg import Pose, Quaternion, TransformStamped, Vector3
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

        self.bin_further_frame = "bin_further"
        self.bin_closer_frame = "bin_closer"

        self.further_distance = rospy.get_param("~further_distance", 1.5)
        self.closer_distance = rospy.get_param("~closer_distance", 1.0)

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
            rospy.logerr(
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

        direction_vector = bin_pos - robot_pos
        total_distance = np.linalg.norm(direction_vector)

        if total_distance == 0:
            rospy.logwarn(
                "Robot and bin are at the same position! Cannot create frames."
            )
            return

        direction_unit = direction_vector / total_distance

        closer_pos = bin_pos - (direction_unit * self.closer_distance)
        further_pos = bin_pos + (direction_unit * self.further_distance)

        closer_pose = Pose()
        closer_pose.position.x, closer_pose.position.y, closer_pose.position.z = (
            closer_pos
        )
        closer_pose.orientation = bin_pose.orientation

        further_pose = Pose()
        further_pose.position.x, further_pose.position.y, further_pose.position.z = (
            further_pos
        )
        further_pose.orientation = bin_pose.orientation

        further_transform = self.build_transform_message(
            self.bin_further_frame, further_pose
        )
        closer_transform = self.build_transform_message(
            self.bin_closer_frame, closer_pose
        )

        self.send_transform(further_transform)
        self.send_transform(closer_transform)

        rospy.logdebug(
            f"Published bin frames relative to bin: {self.bin_closer_frame} (-{self.closer_distance}m) and "
            f"{self.bin_further_frame} (+{self.further_distance}m)"
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
