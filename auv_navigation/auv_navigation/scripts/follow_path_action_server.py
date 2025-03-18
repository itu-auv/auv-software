#!/usr/bin/env python3
import rospy
import actionlib
import tf2_ros
import math
from typing import List, Optional
from geometry_msgs.msg import PoseStamped
import tf.transformations
from nav_msgs.msg import Path
from auv_msgs.msg import (
    FollowPathAction,
    FollowPathFeedback,
    FollowPathResult,
    FollowPathActionGoal,
)
from auv_navigation.follow_path_action import follow_path_helpers


class FollowPathActionServer:
    def __init__(self) -> None:
        # tf2 objects
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # load parameters
        self.dynamic_target_lookahead_distance: float = rospy.get_param(
            "~dynamic_target_lookahead_distance", 1.0
        )
        self.source_frame: str = rospy.get_param("~source_frame", "taluy/base_link")
        self.loop_rate = rospy.Rate(rospy.get_param("~loop_rate", 20))
        self.dynamic_target_yaw_threshold: float = rospy.get_param(
            "~dynamic_target_yaw_threshold", math.pi / 6
        )
        self.last_dynamic_target: PoseStamped = None

        self.path_pub = rospy.Publisher("target_path", Path, queue_size=1)

        self.server = actionlib.SimpleActionServer(
            "follow_path", FollowPathAction, self.execute, auto_start=False
        )
        self.server.start()
        rospy.logdebug("[follow_path server] Action server started")

    def do_path_following(self, path: Path, segment_endpoints: List[int]) -> bool:
        """
        Performs dynamic target following while tracking path progress and segment completion.
            The function continuously:
            - computes the dynamic target ahead of the vehicle.
            - tracks whether each segment of the path (that is, the individual paths before they were combined)
            has been completed.
            - tracks the overall path completion.

        Args:
            path (Path): The combined path to be followed.
            segment_endpoints (List[int]): Indices marking the endpoints of individual path segments.

        Returns:
            bool: True if the entire path is completed successfully, False if interrupted or failed.
        """
        try:
            num_segments = len(segment_endpoints)
            current_segment_index = 0

            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.server.set_preempted()
                    rospy.logdebug("Path following preempted")
                    return False

                # Publish the target path for visualization
                self.path_pub.publish(path)

                # get current pose of robot
                robot_pose = follow_path_helpers.get_robot_pose(
                    self.tf_buffer, self.source_frame
                )
                if robot_pose is None:
                    rospy.logwarn("Failed to get current pose. Retrying...")
                    self.loop_rate.sleep()
                    continue

                candidate_dynamic_target = follow_path_helpers.calculate_dynamic_target(
                    path, robot_pose, self.dynamic_target_lookahead_distance
                )
                if candidate_dynamic_target is None:
                    rospy.logwarn("Failed to calculate dynamic target. Retrying...")
                    self.loop_rate.sleep()
                    continue

                # Check if the yaw of the dynamic target is within the threshold
                # If not, freeze the dynamic target till is.

                # Calculate yaw difference between robot and candidate dynamic target yaw
                robot_orientation = robot_pose.pose.orientation
                _, _, robot_yaw = tf.transformations.euler_from_quaternion(
                    robot_orientation
                )
                candidate_orientation = candidate_dynamic_target.pose.orientation
                _, _, candidate_yaw = tf.transformations.euler_from_quaternion(
                    candidate_orientation
                )

                yaw_diff = abs(tf2_ros.wrap_to_pi(candidate_yaw - robot_yaw))

                if (
                    yaw_diff > self.dynamic_target_yaw_threshold
                    and self.last_dynamic_target is not None
                ):  # only freeze if we have a last dynamic target
                    dynamic_target = self.last_dynamic_target
                    rospy.logdebug(
                        "Yaw diff {:.2f} rad exceeds threshold {:.2f} rad. Freezing dynamic target.".format(
                            yaw_diff, self.dynamic_target_yaw_threshold
                        )
                    )
                else:
                    dynamic_target = candidate_dynamic_target
                    self.last_dynamic_target = candidate_dynamic_target

                # Broadcast the dynamic target frame so that controllers can use it
                follow_path_helpers.broadcast_dynamic_target_frame(
                    self.tf_broadcaster,
                    self.tf_buffer,
                    self.source_frame,
                    dynamic_target,
                )

                # Check progress along the current segment and overall path
                current_segment_progress, overall_progress = (
                    follow_path_helpers.check_segment_progress(
                        path, robot_pose, current_segment_index, segment_endpoints
                    )
                )
                feedback = FollowPathFeedback()
                feedback.current_segment_progress = current_segment_progress
                feedback.overall_progress = overall_progress
                feedback.current_segment_index = current_segment_index
                self.server.publish_feedback(feedback)

                # Check if current segment is completed
                segment_end_index = segment_endpoints[current_segment_index]
                if follow_path_helpers.is_segment_completed(
                    robot_pose, path, segment_end_index
                ):
                    if current_segment_index < num_segments - 1:
                        current_segment_index += 1
                    else:  # was on the last path and it's completed
                        rospy.logdebug("All paths completed")
                        return True
                self.loop_rate.sleep()

            return False

        except Exception as e:
            rospy.logerr(f"Error during path following: {e}")
            return False

    def execute(self, goal: FollowPathActionGoal) -> None:
        rospy.logdebug("FollowPathActionServer: Received a new path following goal.")

        # Check if the goal contains valid paths
        # 1. Abort if paths list is empty or none
        # 2. Abort if all paths are empty
        if goal.paths is None or not goal.paths or all(not p.poses for p in goal.paths):
            rospy.logerr("Received empty paths list or paths with no poses")
            self.server.set_aborted(FollowPathResult(success=False))
            return

        # Combine all paths and get endpoints
        combined_path, segment_endpoints = follow_path_helpers.combine_segments(
            goal.paths
        )

        # Perform path following
        success = self.do_path_following(combined_path, segment_endpoints)
        result = FollowPathResult(success=success)

        if success:
            rospy.logdebug(f"Path following succeeded.")
            self.server.set_succeeded(result)
        else:
            rospy.logdebug(f"Path following did not succeed.")
            if not self.server.is_preempt_requested():
                self.server.set_aborted(result)


if __name__ == "__main__":
    rospy.init_node("follow_path_action_server")
    FollowPathActionServer()
    rospy.spin()
