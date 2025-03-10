#!/usr/bin/env python3
import rospy
import tf2_ros
import tf
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from typing import Optional

DEFAULT_FIXED_FRAME: str = "odom"
DEFAULT_FRAME_ID: str = "odom"
TIME_ZERO = rospy.Time(0)
TF_LOOKUP_TIMEOUT = rospy.Duration(1.0)
TWO_PI: float = 2 * np.pi


class PathPlanningHelper:
    """Helper class containing static methods for path creation."""

    @staticmethod
    def lookup_transform(
        tf_buffer: tf2_ros.Buffer, frame: str, fixed_frame: str = DEFAULT_FIXED_FRAME
    ):
        """
        Looks up the transform for the given frame and returns its position and quaternion.

        Returns:
            (position, quaternion)
        """
        transform = tf_buffer.lookup_transform(
            fixed_frame, frame, TIME_ZERO, TF_LOOKUP_TIMEOUT
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
    def apply_angle_offset(quaternion, angle_offset: float):
        """
        Applies an angle offset (in radians) to the given quaternion.

        Returns:
            The resulting quaternion after applying the offset.
        """
        offset_quat = tf.transformations.quaternion_from_euler(0, 0, angle_offset)
        return tf.transformations.quaternion_multiply(offset_quat, quaternion)

    @staticmethod
    def compute_angular_difference(source_euler, target_euler, n_turns: int):
        """
        Computes the yaw angular difference between the source and target orientations,
        adding extra full 360° turns if requested.

        Returns:
            angular_diff: The yaw difference (including extra turns) between target and source in radians.
        """
        angular_diff = (target_euler[2] - source_euler[2]) + (TWO_PI * n_turns)
        return angular_diff

    @staticmethod
    def interpolate_position(
        source_pos,
        target_pos,
        t: float,
        interpolate_xy: bool = True,
        interpolate_z: bool = True,
    ):
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
        source_euler, angular_diff: float, t: float, interpolate_yaw: bool = True
    ):
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
        header,
        source_position,
        target_position,
        source_euler,
        angular_diff: float,
        num_waypoints: int,
        interpolate_xy: bool,
        interpolate_z: bool,
        interpolate_yaw: bool,
    ):
        """
        Generates a list of PoseStamped waypoints interpolated between the source and target.

        Returns:
            A list of PoseStamped messages.
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
