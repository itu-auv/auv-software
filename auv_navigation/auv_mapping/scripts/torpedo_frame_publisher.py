#!/usr/bin/env python3
import numpy as np
import tf.transformations
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
        rospy.init_node("create_torpedo_frames_node")

        self.enable_target = False
        self.enable_realsense_target = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.robot_frame = "taluy/base_link"
        self.target_frame = "torpedo_target"
        self.realsense_target_frame = "torpedo_target_realsense"
        self.torpedo_frame = rospy.get_param("~torpedo_frame", "torpedo_map_link")
        self.torpedo_realsense_frame = rospy.get_param(
            "~torpedo_realsense_frame", "torpedo_map_link_realsense"
        )

        self.initial_offset = rospy.get_param("~initial_offset", 3.0)
        self.realsense_offset = rospy.get_param("~realsense_offset", 1.5)

        self.set_enable_service = rospy.Service(
            "set_transform_torpedo_target_frame",
            SetBool,
            self.handle_enable_target_service,
        )
        self.set_enable_realsense_service = rospy.Service(
            "set_transform_torpedo_realsense_target_frame",
            SetBool,
            self.handle_enable_realsense_target_service,
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

    def create_target_frame(self):
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(1)
            )
            torpedo_tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.torpedo_frame, rospy.Time(0), rospy.Duration(1)
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        robot_pose = self.get_pose(robot_tf)
        torpedo_pose = self.get_pose(torpedo_tf)

        robot_pos = np.array(
            [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
        )
        torpedo_pos = np.array(
            [torpedo_pose.position.x, torpedo_pose.position.y, torpedo_pose.position.z]
        )

        direction_vector_2d = torpedo_pos[:2] - robot_pos[:2]
        total_distance_2d = np.linalg.norm(direction_vector_2d)

        if total_distance_2d == 0:
            rospy.logwarn(
                "Robot and torpedo are at the same XY position! Cannot create frame."
            )
            return

        direction_unit_2d = direction_vector_2d / total_distance_2d

        # Calculate yaw from the direction vector (robot to torpedo)
        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        orientation = Pose().orientation
        orientation.x = q[0]
        orientation.y = q[1]
        orientation.z = q[2]
        orientation.w = q[3]

        # Calculate position for closer frame
        closer_pos_2d = torpedo_pos[:2] - (direction_unit_2d * self.initial_offset)
        closer_pos = np.append(closer_pos_2d, robot_pos[2])

        # Create closer frame
        closer_pose = Pose()
        closer_pose.position.x, closer_pose.position.y, closer_pose.position.z = (
            closer_pos
        )
        closer_pose.orientation = orientation

        # Send the transform
        closer_transform = self.build_transform_message(self.target_frame, closer_pose)
        self.send_transform(closer_transform)

    def create_realsense_target_frame(self):
        try:
            torpedo_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_realsense_frame,
                rospy.Time(0),
                rospy.Duration(1),
            )
            torpedo_pose = self.get_pose(torpedo_tf)
            realsense_target_pose = self.apply_offsets(
                torpedo_pose,
                [
                    0.0,
                    self.realsense_offset,
                    0.0,
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

    def handle_enable_target_service(self, req):
        self.enable_target = req.data
        message = (
            f"Torpido target frame transform publish is set to: {self.enable_target}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_enable_realsense_target_service(self, req):
        self.enable_realsense_target = req.data
        message = f"Torpido realsense target frame transform publish is set to: {self.enable_realsense_target}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.enable_target:
                self.create_target_frame()
            if self.enable_realsense_target:
                self.create_realsense_target_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TorpedoTransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
