#!/usr/bin/env python3
import rospy
import tf2_ros
import tf
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from typing import Optional, List
from .path_planning_helpers import PathPlanningHelper


class PathPlanners:
    def __init__(self, tf_buffer: tf2_ros.Buffer):
        self.tf_buffer = tf_buffer
        self.base_link_frame: str = rospy.get_param("~base_link", "taluy/base_link")
        self.header_frame: str = rospy.get_param("~header_frame", "odom")
        self.gate_entrance_frame: str = rospy.get_param(
            "~gate_entrance_frame", "gate_entrance"
        )
        self.gate_exit_frame: str = rospy.get_param("~gate_exit_frame", "gate_exit")
        self.bin_closer_frame: str = rospy.get_param("~bin_closer_frame", "bin_closer")
        self.bin_further_frame: str = rospy.get_param(
            "~bin_further_frame", "bin_further"
        )
        self.path_creation_timeout: float = rospy.get_param(
            "~path_creation_timeout", 20.0
        )

    def straight_path_to_frame(
        self,
        source_frame: str,
        target_frame: str,
        angle_offset: float = 0.0,
        num_waypoints: int = 50,
        n_turns: int = 0,
        interpolate_xy: bool = True,
        interpolate_z: bool = True,
        interpolate_yaw: bool = True,
    ) -> Optional[Path]:
        """
        Creates a straight path from source frame to target frame.

        Args:
            source_frame: The starting frame.
            target_frame: The destination frame.
            angle_offset: Additional angle offset (in radians) to add to the target orientation.
            num_waypoints: Number of waypoints to generate.
            n_turns: Number of full 360-degree turns to add to the interpolation.
            interpolate_xy: If True, interpolate the x and y positions.
            interpolate_z: If True, interpolate the z position.
            interpolate_yaw: If True, interpolate the yaw orientation.

        Returns:
            A Path message if successful, otherwise None.
        """
        try:
            # Lookup transforms and extract positions and quaternions.
            source_position, source_quaternion = PathPlanningHelper.lookup_transform(
                self.tf_buffer, source_frame
            )
            target_position, target_quaternion = PathPlanningHelper.lookup_transform(
                self.tf_buffer, target_frame
            )

            # Apply angle_offset to the target quaternion.
            final_target_quat = PathPlanningHelper.apply_angle_offset(
                target_quaternion, angle_offset
            )

            # Get euler of source and target.
            source_euler = tf.transformations.euler_from_quaternion(source_quaternion)
            final_target_euler = tf.transformations.euler_from_quaternion(
                final_target_quat
            )

            # Compute the angular difference (yaw only).
            angular_diff = PathPlanningHelper.compute_angular_difference(
                source_euler, final_target_euler, n_turns
            )

            # Create a header for the path.
            header = PathPlanningHelper.create_path_header(self.header_frame)

            # Generate waypoints using the provided interpolation flags.
            poses = PathPlanningHelper.generate_waypoints(
                header,
                source_position,
                target_position,
                source_euler,
                angular_diff,
                num_waypoints,
                interpolate_xy,
                interpolate_z,
                interpolate_yaw,
            )

            # Create and return the final Path message.
            return PathPlanningHelper.create_path_from_poses(header, poses)

        except Exception as e:
            rospy.logerr(f"Error in straight_path_to_frame: {e}")
            return None

    def path_for_gate(self) -> Optional[List[Path]]:
        """
        Plans paths for the gate task, which includes the two paths:
        1. path to gate entrance
        2. Gate entrance to gate exit (with 1 360 degrees turn)
        """
        try:
            rospy.logdebug("[GatePathPlanner] Planning paths for gate task...")
            start_time = rospy.Time.now()
            # Plan path to gate entrance
            entrance_path = None
            exit_path = None

            while (rospy.Time.now() - start_time).to_sec() < self.path_creation_timeout:
                try:
                    # create the first segment
                    if entrance_path is None:
                        entrance_path = self.straight_path_to_frame(
                            source_frame=self.base_link_frame,
                            target_frame=self.gate_entrance_frame,
                        )
                    # create the second segment
                    if exit_path is None:
                        exit_path = self.straight_path_to_frame(
                            source_frame=self.gate_entrance_frame,
                            target_frame=self.gate_exit_frame,
                            n_turns=1,
                        )
                    if entrance_path is not None and exit_path is not None:
                        return [entrance_path, exit_path]
                    rospy.logwarn(
                        "[GatePathPlanner] Failed to plan paths, retrying... Time elapsed: %.1f seconds",
                        (rospy.Time.now() - start_time).to_sec(),
                    )
                    rospy.sleep(0.5)

                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logwarn(
                        "[GatePathPlanner] TF error while planning paths: %s. Retrying...",
                        str(e),
                    )
                    rospy.sleep(0.5)

            # If we get here, we timed out
            rospy.logwarn(
                "[GatePathPlanner] Failed to plan paths after %.1f seconds",
                self.path_creation_timeout,
            )
            return None

        except Exception as e:
            rospy.logwarn("[GatePathPlanner] Error in gate path planning: %s", str(e))
            return None

    def path_for_bin(self) -> Optional[List[Path]]:
        try:
            rospy.logdebug("[BinPathPlanner] Planning paths for bin task...")
            start_time = rospy.Time.now()
            # Plan path to gate entrance
            entrance_path = None
            exit_path = None

            while (rospy.Time.now() - start_time).to_sec() < self.path_creation_timeout:
                try:
                    # create the first segment
                    if entrance_path is None:
                        entrance_path = self.straight_path_to_frame(
                            source_frame=self.base_link_frame,
                            target_frame=self.bin_closer_frame,
                        )
                    # create the second segment
                    if exit_path is None:
                        exit_path = self.straight_path_to_frame(
                            source_frame=self.bin_closer_frame,
                            target_frame=self.bin_further_frame,
                        )
                    if entrance_path is not None and exit_path is not None:
                        return [entrance_path, exit_path]
                    rospy.logwarn(
                        "[BinPathPlanner] Failed to plan paths, retrying... Time elapsed: %.1f seconds",
                        (rospy.Time.now() - start_time).to_sec(),
                    )
                    rospy.sleep(0.5)

                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logwarn(
                        "[GatePathPlanner] TF error while planning paths: %s. Retrying...",
                        str(e),
                    )
                    rospy.sleep(0.5)

            # If we get here, we timed out
            rospy.logwarn(
                "[GatePathPlanner] Failed to plan paths after %.1f seconds",
                self.path_creation_timeout,
            )
            return None

        except Exception as e:
            rospy.logwarn("[GatePathPlanner] Error in gate path planning: %s", str(e))
            return None
