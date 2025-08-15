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
from dynamic_reconfigure.server import Server

from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
)
from auv_mapping.cfg import TorpedoTrajectoryConfig


class TorpedoTransformServiceNode:
    def __init__(self):
        rospy.init_node("create_torpedo_frames_node")

        self.enable_target = False
        self.enable_realsense_target = False
        self.enable_torpedo_hole_target = False

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
        self.torpedo_fire_frame = "torpedo_fire_frame"

        self.torpedo_frame = rospy.get_param("~torpedo_frame", "torpedo_map_link")
        self.torpedo_realsense_frame = rospy.get_param(
            "~torpedo_realsense_frame", "torpedo_map_link_realsense"
        )
        self.torpedo_hole_shark_frame = rospy.get_param(
            "~torpedo_hole_shark_frame", "torpedo_hole_shark_link"
        )
        self.torpedo_hole_sawfish_frame = rospy.get_param(
            "~torpedo_hole_sawfish_frame", "torpedo_hole_sawfish_link"
        )
        self.torpedo_shark_fire_frame = rospy.get_param(
            "~torpedo_shark_fire_frame", "torpedo_shark_fire_frame"
        )
        self.torpedo_sawfish_fire_frame = rospy.get_param(
            "~torpedo_sawfish_fire_frame", "torpedo_sawfish_fire_frame"
        )

        # Initialize default values for dynamic reconfigure parameters
        self.initial_offset = 2.5
        self.realsense_offset = 1.4
        self.fire_offset = -0.4
        self.shark_fire_y_offset = 0.0
        self.sawfish_fire_y_offset = 0.0

        # Dynamic reconfigure server
        self.reconfigure_server = Server(
            TorpedoTrajectoryConfig, self.reconfigure_callback
        )

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
        self.set_enable_torpedo_hole_service = rospy.Service(
            "set_transform_torpedo_hole_target_frame",
            SetBool,
            self.handle_enable_torpedo_hole_target_service,
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
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(4.0)
            )
            torpedo_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
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
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            torpedo_pose = self.get_pose(torpedo_tf)

            # Set the pitch and roll to zero
            q = [
                torpedo_pose.orientation.x,
                torpedo_pose.orientation.y,
                torpedo_pose.orientation.z,
                torpedo_pose.orientation.w,
            ]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(q)
            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            torpedo_pose.orientation.x = q[0]
            torpedo_pose.orientation.y = q[1]
            torpedo_pose.orientation.z = q[2]
            torpedo_pose.orientation.w = q[3]

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

    def create_torpedo_hole_target_frame(self):
        # Shark fire frame
        try:
            torpedo_hole_shark_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_hole_shark_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            realsense_target_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.realsense_target_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            torpedo_hole_shark_pose = self.get_pose(torpedo_hole_shark_tf)
            realsense_target_pose = self.get_pose(realsense_target_tf)

            # Get the yaw from realsense target and add 90 degrees (pi/2 radians)
            q = [
                realsense_target_pose.orientation.x,
                realsense_target_pose.orientation.y,
                realsense_target_pose.orientation.z,
                realsense_target_pose.orientation.w,
            ]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(q)
            # Rotate 90 degrees to the right
            rotated_yaw = yaw - np.pi / 2
            rotated_q = tf.transformations.quaternion_from_euler(0, 0, rotated_yaw)

            torpedo_hole_shark_pose.orientation.x = rotated_q[0]
            torpedo_hole_shark_pose.orientation.y = rotated_q[1]
            torpedo_hole_shark_pose.orientation.z = rotated_q[2]
            torpedo_hole_shark_pose.orientation.w = rotated_q[3]

            shark_fire_pose = self.apply_offsets(
                torpedo_hole_shark_pose,
                [self.fire_offset, self.shark_fire_y_offset, 0.0],
            )
            shark_fire_transform = self.build_transform_message(
                self.torpedo_shark_fire_frame, shark_fire_pose
            )
            self.send_transform(shark_fire_transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass
        # Sawfish fire frame
        try:
            torpedo_hole_sawfish_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_hole_sawfish_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            realsense_target_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.realsense_target_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            torpedo_hole_sawfish_pose = self.get_pose(torpedo_hole_sawfish_tf)
            realsense_target_pose = self.get_pose(realsense_target_tf)

            # Get the yaw from realsense target and add 90 degrees (pi/2 radians)
            q = [
                realsense_target_pose.orientation.x,
                realsense_target_pose.orientation.y,
                realsense_target_pose.orientation.z,
                realsense_target_pose.orientation.w,
            ]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(q)
            # Rotate 90 degrees to the right
            rotated_yaw = yaw - np.pi / 2
            rotated_q = tf.transformations.quaternion_from_euler(0, 0, rotated_yaw)

            torpedo_hole_sawfish_pose.orientation.x = rotated_q[0]
            torpedo_hole_sawfish_pose.orientation.y = rotated_q[1]
            torpedo_hole_sawfish_pose.orientation.z = rotated_q[2]
            torpedo_hole_sawfish_pose.orientation.w = rotated_q[3]

            sawfish_fire_pose = self.apply_offsets(
                torpedo_hole_sawfish_pose,
                [self.fire_offset, self.sawfish_fire_y_offset, 0.0],
            )
            sawfish_fire_transform = self.build_transform_message(
                self.torpedo_sawfish_fire_frame, sawfish_fire_pose
            )
            self.send_transform(sawfish_fire_transform)
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

    def handle_enable_torpedo_hole_target_service(self, req):
        self.enable_torpedo_hole_target = req.data
        message = f"Torpido hole target frame transform publish is set to: {self.enable_torpedo_hole_target}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def reconfigure_callback(self, config, level):
        """Callback for dynamic reconfigure parameters"""
        self.initial_offset = config.initial_offset
        self.realsense_offset = config.realsense_offset
        self.fire_offset = config.fire_offset
        self.shark_fire_y_offset = config.shark_fire_y_offset
        self.sawfish_fire_y_offset = config.sawfish_fire_y_offset
        rospy.loginfo("Torpedo trajectory parameters updated via dynamic reconfigure")
        return config

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.enable_target:
                self.create_target_frame()
            if self.enable_realsense_target:
                self.create_realsense_target_frame()
            if self.enable_torpedo_hole_target:
                self.create_torpedo_hole_target_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TorpedoTransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
