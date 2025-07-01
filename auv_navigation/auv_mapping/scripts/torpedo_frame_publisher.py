#!/usr/bin/env python3
import numpy as np
from tf.transformations import (
    quaternion_matrix,
)
import rospy
import tf2_ros
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse

from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
)


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
        self.realsense_target_frame = "torpedo_target_realsense"
        self.torpedo_frame = rospy.get_param("~torpedo_frame", "torpedo_map_link")
        self.torpedo_realsense_frame = rospy.get_param(
            "~torpedo_realsense_frame", "torpedo_map_link_realsense"
        )

        self.offset_x = rospy.get_param("~offset_x", 0.0)
        self.offset_y = rospy.get_param("~offset_y", 0.5)
        self.offset_z = rospy.get_param("~offset_z", 0.0)
        self.realsense_offset_x = rospy.get_param("~realsense_offset_x", 0.0)
        self.realsense_offset_y = rospy.get_param("~realsense_offset_y", 0.5)
        self.realsense_offset_z = rospy.get_param("~realsense_offset_z", 0.0)

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

    def apply_offsets(self, pose: Pose, offsets: list) -> Pose:
        """
        Apply offsets to a pose in its own frame.
        """
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

        rotation_matrix = quaternion_matrix(q)[:3, :3]

        offset_vector = np.array(offsets)
        rotated_offset = np.dot(rotation_matrix, offset_vector)

        new_pose = Pose()
        new_pose.position.x = pose.position.x + rotated_offset[0]
        new_pose.position.y = pose.position.y + rotated_offset[1]
        new_pose.position.z = pose.position.z + rotated_offset[2]
        new_pose.orientation = pose.orientation

        return new_pose

    def create_torpedo_frames(self):
        """
        Look up the current transforms, compute target transforms, and broadcast them
        """
        try:
            transform_torpedo = self.tf_buffer.lookup_transform(
                self.odom_frame, self.torpedo_frame, rospy.Time(0), rospy.Duration(1)
            )
            torpedo_pose = self.get_pose(transform_torpedo)
            target_pose = self.apply_offsets(
                torpedo_pose, [self.offset_x, self.offset_y, self.offset_z]
            )
            target_transform = self.build_transform_message(
                self.target_frame, target_pose
            )
            self.send_transform(target_transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup for {self.torpedo_frame} failed: {e}")

        try:
            transform_realsense = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_realsense_frame,
                rospy.Time(0),
                rospy.Duration(1),
            )
            realsense_pose = self.get_pose(transform_realsense)
            realsense_target_pose = self.apply_offsets(
                realsense_pose,
                [
                    self.realsense_offset_x,
                    self.realsense_offset_y,
                    self.realsense_offset_z,
                ],
            )
            realsense_target_transform = self.build_transform_message(
                self.realsense_target_frame, realsense_target_pose
            )
            self.send_transform(realsense_target_transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

    def handle_enable_service(self, req):
        self.enable = req.data
        message = f"Torpido target frames transform publish is set to: {self.enable}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(20)
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
