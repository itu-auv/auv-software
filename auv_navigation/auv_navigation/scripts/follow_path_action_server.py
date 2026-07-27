#!/usr/bin/env python3
import math
import rospy
import actionlib
import tf2_ros
from typing import List, Optional

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
        self.yaw_gate_enter_threshold_rad: float = rospy.get_param(
            "~yaw_gate_enter_threshold_rad", 0.4
        )
        self.yaw_gate_exit_threshold_rad: float = rospy.get_param(
            "~yaw_gate_exit_threshold_rad", 0.1
        )
        self.yaw_gate_position_lookahead_m: float = rospy.get_param(
            "~yaw_gate_position_lookahead_m", 0.15
        )
        self.completion_distance_threshold: float = rospy.get_param(
            "~completion_distance_threshold", 0.2
        )
        self.completion_yaw_threshold: float = rospy.get_param(
            "~completion_yaw_threshold", 0.18
        )
        self.source_frame: str = rospy.get_param("~source_frame", "taluy/base_link")
        self.loop_rate = rospy.Rate(rospy.get_param("~loop_rate", 20))

        self.path_pub = rospy.Publisher("target_path", Path, queue_size=1)
        self.path_sub = rospy.Subscriber(
            "/planned_path", Path, self.path_cb, queue_size=1
        )
        self.current_path = None
        self.yaw_gate_active = False

        self.server = actionlib.SimpleActionServer(
            "follow_path", FollowPathAction, self.execute, auto_start=False
        )
        self.server.start()
        rospy.logdebug("[follow_path server] Action server started")

    def path_cb(self, msg: Path) -> None:
        self.current_path = msg

    def do_path_following(self) -> bool:
        """
        Performs dynamic target following while tracking path progress and segment completion.
            The function continuously:
            - computes the dynamic target ahead of the vehicle.
            - tracks the overall path completion.

        Returns:
            bool: True if the entire path is completed successfully, False if interrupted or failed.
        """
        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.server.set_preempted()
                    rospy.logdebug("Path following preempted")
                    return False

                if self.current_path is None:
                    rospy.logdebug("No path received yet. Waiting for path...")
                    self.loop_rate.sleep()
                    continue

                # Publish the target path for visualization
                self.path_pub.publish(self.current_path)

                # get current pose of robot
                robot_pose = follow_path_helpers.get_robot_pose(
                    self.tf_buffer, self.source_frame
                )
                if robot_pose is None:
                    rospy.logwarn("Failed to get current pose. Retrying...")
                    self.loop_rate.sleep()
                    continue

                dynamic_target_pose = follow_path_helpers.calculate_dynamic_target(
                    self.current_path,
                    robot_pose,
                    self.dynamic_target_lookahead_distance,
                )
                if dynamic_target_pose is None:
                    rospy.logwarn("Failed to calculate dynamic target. Retrying...")
                    self.loop_rate.sleep()
                    continue

                target_yaw = follow_path_helpers.get_pose_yaw(dynamic_target_pose)
                robot_yaw = follow_path_helpers.get_pose_yaw(robot_pose)
                yaw_error = abs(
                    follow_path_helpers.normalize_angle(target_yaw - robot_yaw)
                )
                was_yaw_gate_active = self.yaw_gate_active

                rospy.logdebug(
                    "[follow_path server] yaw_gate error: %.3f rad (%.1f deg), active=%s",
                    yaw_error,
                    math.degrees(yaw_error),
                    self.yaw_gate_active,
                )

                (
                    dynamic_target_pose,
                    self.yaw_gate_active,
                ) = follow_path_helpers.apply_yaw_gate(
                    dynamic_target_pose,
                    robot_pose,
                    self.yaw_gate_active,
                    self.yaw_gate_enter_threshold_rad,
                    self.yaw_gate_exit_threshold_rad,
                    self.yaw_gate_position_lookahead_m,
                )
                if self.yaw_gate_active and not was_yaw_gate_active:
                    rospy.logdebug(
                        "[follow_path server] yaw_gate entered: yaw_error=%.3f rad (%.1f deg)",
                        yaw_error,
                        math.degrees(yaw_error),
                    )
                elif was_yaw_gate_active and not self.yaw_gate_active:
                    rospy.logdebug(
                        "[follow_path server] yaw_gate exited: yaw_error=%.3f rad (%.1f deg)",
                        yaw_error,
                        math.degrees(yaw_error),
                    )

                # Broadcast the dynamic target frame so that controllers can use it
                follow_path_helpers.broadcast_dynamic_target_frame(
                    self.tf_broadcaster,
                    self.tf_buffer,
                    self.source_frame,
                    dynamic_target_pose,
                )

                if follow_path_helpers.is_path_completed(
                    robot_pose,
                    self.current_path,
                    self.completion_distance_threshold,
                    self.completion_yaw_threshold,
                ):
                    rospy.loginfo(" [FollowPathActionServer] Path completed!")
                    return True

                feedback = FollowPathFeedback()
                self.server.publish_feedback(feedback)

                self.loop_rate.sleep()

            return False

        except Exception as e:
            rospy.logerr(f"Error during path following: {e}")
            return False

    def execute(self, goal: FollowPathActionGoal) -> None:
        rospy.logdebug("FollowPathActionServer: Received a new path following goal.")
        self.current_path = None  # Clear the path on new goal
        self.yaw_gate_active = False

        success = self.do_path_following()
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
