#!/usr/bin/env python3
import rospy
import tf2_ros
import tf
import numpy as np

# ROS message imports
from nav_msgs.msg import Path
from geometry_msgs.msg import (
    PoseStamped,
    Point,
    Quaternion,
    Vector3,
)
from std_msgs.msg import Header
from typing import Tuple, List

TWO_PI = 2 * np.pi


class PathPlanningHelper:
    """Helper class containing static methods for path creation."""

    @staticmethod
    def lookup_transform(
        tf_buffer: tf2_ros.Buffer, frame: str, fixed_frame: str = "odom"
    ) -> Tuple[Vector3, List[float]]:
        """
        Looks up the transform for the given frame and returns its position and quaternion.
        """
        transform = tf_buffer.lookup_transform(
            fixed_frame, frame, rospy.Time(0), rospy.Duration(1.0)
        )
        position = transform.transform.translation
        quaternion = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        return position, quaternion

    @staticmethod
    def apply_angle_offset(quaternion: List[float], angle_offset: float) -> List[float]:
        """
        Applies an angle offset (in radians) to the given quaternion.

        Returns:
            List[float]: The resulting 4-element quaternion after applying the offset.
        """
        offset_quat = tf.transformations.quaternion_from_euler(0, 0, angle_offset)
        return tf.transformations.quaternion_multiply(offset_quat, quaternion)

    @staticmethod
    def compute_angular_difference(
        source_euler: List[float], target_euler: List[float], n_turns: int
    ) -> float:
        """
        Computes the yaw angular difference between the source and target orientations,
        adding extra full 360° turns if requested.

        Args:
            source_euler (List[float]): [roll, pitch, yaw] of the source.
            target_euler (List[float]): [roll, pitch, yaw] of the target.
            n_turns (int): Number of extra full 360° rotations to add to the difference.

        Returns:
            float: The yaw difference (including extra turns) in radians.
        """
        raw_diff = target_euler[2] - source_euler[2]
        normalized_diff = (raw_diff + np.pi) % (2 * np.pi) - np.pi
        angular_diff = normalized_diff + (TWO_PI * n_turns)
        return angular_diff

    @staticmethod
    def interpolate_position(
        source_pos: Point,
        target_pos: Point,
        t: float,
        interpolate_xy: bool = True,
        interpolate_z: bool = True,
    ) -> Point:
        """
        Linearly interpolates between two positions given a factor t [0,1]

        Args:
            source_pos: The starting position.
            target_pos: The ending position.
            t: Interpolation factor (0.0 to 1.0).
            interpolate_xy: If True, interpolate x and y; otherwise, keep source x and y.
            interpolate_z: If True, interpolate z; otherwise, keep source z.

        Returns:
            A new position with interpolated values for the selected components.
        """
        interp_pos = type(source_pos)()  # Create a new instance of the position type.
        if interpolate_xy:
            interp_pos.x = source_pos.x + t * (target_pos.x - source_pos.x)
            interp_pos.y = source_pos.y + t * (target_pos.y - source_pos.y)
        else:
            interp_pos.x = source_pos.x
            interp_pos.y = source_pos.y

        if interpolate_z:
            interp_pos.z = source_pos.z + t * (target_pos.z - source_pos.z)
        else:
            interp_pos.z = source_pos.z

        return interp_pos

    @staticmethod
    def interpolate_orientation(
        source_euler: List[float],
        angular_diff: float,
        t: float,
        interpolate_yaw: bool = True,
    ) -> List[float]:
        """
        Interpolates the yaw component between the source and target orientations based on the flag.

        Roll and pitch are taken from the source.

        Args:
            source_euler: Euler angles of source.
            angular_diff: The yaw difference between target and source (with extra turns).
            t: Interpolation factor (0.0 to 1.0).
            interpolate_yaw: If True, interpolate yaw; otherwise, keep source yaw.

        Returns:
            A quaternion representing the interpolated orientation.
        """
        roll = source_euler[0]
        pitch = source_euler[1]
        if interpolate_yaw:
            new_yaw = source_euler[2] + t * angular_diff
        else:
            new_yaw = source_euler[2]
        return tf.transformations.quaternion_from_euler(roll, pitch, new_yaw)

    @staticmethod
    def generate_waypoints(
        header: Header,
        source_position: Point,
        target_position: Point,
        source_euler: List[float],
        angular_diff: float,
        num_waypoints: int,
        interpolate_xy: bool,
        interpolate_z: bool,
        interpolate_yaw: bool,
    ) -> List[PoseStamped]:
        """
        Generates a list of PoseStamped waypoints interpolated between the source and target positions/orientations.

        Args:
            header (Header): The ROS Header to attach to each PoseStamped.
            source_position (Point): Starting position (x,y,z).
            target_position (Point): Target position (x,y,z).
            source_euler (List[float]): [roll, pitch, yaw] of the source orientation.
            angular_diff (float): The yaw difference (with possible extra turns).
            num_waypoints (int): Number of waypoints to generate.
            interpolate_xy (bool): If True, interpolate x and y.
            interpolate_z (bool): If True, interpolate z.
            interpolate_yaw (bool): If True, interpolate yaw.

        Returns:
            List[PoseStamped]: A list of interpolated PoseStamped messages.
        """
        poses = []
        for t in np.linspace(0, 1, num_waypoints):
            pose = PoseStamped()
            pose.header = header

            # Interpolate position based on flags.
            interp_pos = PathPlanningHelper.interpolate_position(
                source_position, target_position, t, interpolate_xy, interpolate_z
            )
            pose.pose.position.x = interp_pos.x
            pose.pose.position.y = interp_pos.y
            pose.pose.position.z = interp_pos.z

            # Interpolate orientation (yaw) based on flag.
            interp_quat = PathPlanningHelper.interpolate_orientation(
                source_euler, angular_diff, t, interpolate_yaw
            )
            pose.pose.orientation.x = interp_quat[0]
            pose.pose.orientation.y = interp_quat[1]
            pose.pose.orientation.z = interp_quat[2]
            pose.pose.orientation.w = interp_quat[3]

            poses.append(pose)
        return poses

    @staticmethod
    def create_path_header(frame_id: str) -> rospy.Header:
        """
        Creates header for the Path message.
        """
        header = rospy.Header()
        header.frame_id = frame_id
        header.stamp = rospy.Time.now()
        return header

    @staticmethod
    def create_path_from_poses(header, poses):
        """
        Creates a Path message given a header and list of PoseStamped poses.
        """
        path = Path()
        path.header = header
        path.poses = poses
        return path
