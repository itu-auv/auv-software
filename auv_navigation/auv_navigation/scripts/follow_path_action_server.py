#!/usr/bin/env python3
import math
import rospy
import actionlib
import tf2_ros
from typing import List, Optional
from tf.transformations import euler_from_quaternion

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
        self.completion_distance_threshold: float = rospy.get_param(
            "~completion_distance_threshold", 0.5
        )
        self.completion_yaw_threshold: float = rospy.get_param(
            "~completion_yaw_threshold", 0.4
        )
        self.source_frame: str = rospy.get_param("~source_frame", "taluy/base_link")
        self.loop_rate = rospy.Rate(rospy.get_param("~loop_rate", 20))

        self.path_pub = rospy.Publisher("target_path", Path, queue_size=1)
        self.path_sub = rospy.Subscriber(
            "/planned_path", Path, self.path_cb, queue_size=1
        )
        self.current_path = None
        self._last_dynamic_target_stamp = None

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
                    rospy.logwarn_throttle(
                        2.0, "[FollowPath] No path received yet. Waiting..."
                    )
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

                # Broadcast the dynamic target frame so that controllers can use it.
                # Guard against repeated timestamps under use_sim_time (same fix as waypoint_gui).
                _now = rospy.Time.now()
                if (
                    self._last_dynamic_target_stamp is None
                    or _now > self._last_dynamic_target_stamp
                ):
                    self._last_dynamic_target_stamp = _now
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

                # --- Diagnostic: log progress ---
                last_pose = (
                    self.current_path.poses[-1].pose
                    if self.current_path.poses
                    else None
                )
                if last_pose is not None:
                    dx = robot_pose.pose.position.x - last_pose.position.x
                    dy = robot_pose.pose.position.y - last_pose.position.y
                    dist_to_end = (dx**2 + dy**2) ** 0.5
                    _, _, robot_yaw = euler_from_quaternion(
                        [
                            robot_pose.pose.orientation.x,
                            robot_pose.pose.orientation.y,
                            robot_pose.pose.orientation.z,
                            robot_pose.pose.orientation.w,
                        ]
                    )
                    _, _, last_yaw = euler_from_quaternion(
                        [
                            last_pose.orientation.x,
                            last_pose.orientation.y,
                            last_pose.orientation.z,
                            last_pose.orientation.w,
                        ]
                    )
                    yaw_diff = abs(robot_yaw - last_yaw)
                    if yaw_diff > math.pi:
                        yaw_diff = 2 * math.pi - yaw_diff
                    rospy.loginfo_throttle(
                        3.0,
                        f"[FollowPath] dist_to_end={dist_to_end:.2f}m "
                        f"(thr={self.completion_distance_threshold}m) "
                        f"yaw_diff={math.degrees(yaw_diff):.1f}deg "
                        f"path_len={len(self.current_path.poses)}",
                    )

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
