#!/usr/bin/env python3
import rospy
import tf2_ros
import tf
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from typing import Optional, List
from path_planning_helpers import PathPlanningHelper

DEFAULT_HEADER_FRAME: str = "odom"
BASE_LINK_FRAME: str = "taluy/base_link"
GATE_ENTRANCE_FRAME: str = "gate_entrance"
GATE_EXIT_FRAME: str = "gate_exit"


class PathPlanners:
    def __init__(self, tf_buffer: tf2_ros.Buffer):
        self.tf_buffer = tf_buffer

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
            header = PathPlanningHelper.create_path_header(DEFAULT_HEADER_FRAME)

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

    def path_for_gate(
        self, path_creation_timeout: float = 20.0
    ) -> Optional[List[Path]]:
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

            while (rospy.Time.now() - start_time).to_sec() < path_creation_timeout:
                try:
                    # create the first segment
                    if entrance_path is None:
                        entrance_path = self.straight_path_to_frame(
                            source_frame=BASE_LINK_FRAME,
                            target_frame=GATE_ENTRANCE_FRAME,
                        )
                    # create the second segment
                    if exit_path is None:
                        exit_path = self.straight_path_to_frame(
                            source_frame=GATE_ENTRANCE_FRAME,
                            target_frame=GATE_EXIT_FRAME,
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
                path_creation_timeout,
            )
            return None

        except Exception as e:
            rospy.logwarn("[GatePathPlanner] Error in gate path planning: %s", str(e))
            return None
