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
import tf_conversions
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class TorpedoTransformServiceNode:
    def __init__(self):
        self.enable = False
        rospy.init_node("create_torpedo_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.target_frame = "torpedo_target"
        self.torpedo_frame = rospy.get_param("~torpedo_frame", "torpedo_map_link")

        self.offset_x = rospy.get_param("~offset_x", 0.0)
        self.offset_y = rospy.get_param("~offset_y", 0.5)
        self.offset_z = rospy.get_param("~offset_z", 0.0)
        self.offset_yaw = rospy.get_param("~offset_yaw", -90.0)  # in degrees

        self.set_enable_service = rospy.Service(
            "set_transform_torpedo_frames", SetBool, self.handle_enable_service
        )

    def get_pose(self, transform: TransformStamped) -> Pose:
        """
        Convert a TransformStamped to a Pose.
        """
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

    def apply_offsets(
        self,
        pose: Pose,
        translation_offsets: Tuple[float, float, float],
        yaw_offset: float,
    ) -> Pose:
        """
        Apply position and yaw offsets to a pose. The position offset is applied in the local frame of the pose.
        The yaw offset is applied around the local Z-axis.
        """
        # Extract the orientation quaternion
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

        # Create rotation matrix from quaternion
        rot_matrix = quaternion_matrix(q)  # 4x4 transformation matrix
        rot_matrix = rot_matrix[:3, :3]  # Extract 3x3 rotation part

        # Rotate the offset vector into world frame
        offset_vector = np.array(translation_offsets)
        rotated_offset = rot_matrix.dot(offset_vector)

        # Apply translated position in world frame
        pose.position.x += rotated_offset[0]
        pose.position.y += rotated_offset[1]
        pose.position.z += rotated_offset[2]

        # Apply yaw rotation (in local frame)
        yaw_rad = math.radians(yaw_offset)
        yaw_q = quaternion_from_euler(0, 0, yaw_rad)
        new_orientation = quaternion_multiply(q, yaw_q)

        pose.orientation = Quaternion(*new_orientation)
        return pose

    def create_torpedo_frames(self):
        """
        Look up the current transforms, compute target transforms, and broadcast them
        """
        try:
            transform_torpedo = self.tf_buffer.lookup_transform(
                self.odom_frame, self.torpedo_frame, rospy.Time(0), rospy.Duration(1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        torpedo_pose = self.get_pose(transform_torpedo)

        target_pose = self.apply_offsets(
            torpedo_pose, [self.offset_x, self.offset_y, self.offset_z], self.offset_yaw
        )

        target_transform = self.build_transform_message(self.target_frame, target_pose)
        self.send_transform(target_transform)

    def handle_enable_service(self, req):
        self.enable = req.data
        message = f"Torpido target frames transform publish is set to: {self.enable}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.enable:
                self.create_torpedo_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TorpedoTransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
