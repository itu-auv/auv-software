#!/usr/bin/env python3
import threading
import rclpy
from auv_common_lib.ros import get_param, Rate
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from auv_common_lib import ros
import tf2_ros
from typing import List, Optional

from nav_msgs.msg import Path
from auv_msgs.action import FollowPath
from auv_navigation.follow_path_action import follow_path_helpers


class FollowPathActionServer(Node):
    def __init__(self) -> None:
        super().__init__("follow_path_action_server")
        # tf2 objects
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # load parameters
        self.dynamic_target_lookahead_distance: float = get_param(
            self, "~dynamic_target_lookahead_distance", 1.0
        )
        self.completion_distance_threshold: float = get_param(
            self, "~completion_distance_threshold", 0.5
        )
        self.completion_yaw_threshold: float = get_param(
            self, "~completion_yaw_threshold", 0.4
        )
        self.source_frame: str = get_param(self, "~source_frame", "taluy/base_link")
        self.loop_rate = Rate(self, get_param(self, "~loop_rate", 20))

        self.path_pub = self.create_publisher(Path, "target_path", 1)
        self.path_sub = self.create_subscription(Path, "/planned_path", self.path_cb, 1)
        self.current_path = None

        self.goal_handle = None
        self.goal_lock = threading.Lock()
        self.server = ActionServer(
            self,
            FollowPath,
            "follow_path",
            execute_callback=self.execute,
            handle_accepted_callback=self.handle_accepted,
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().debug("[follow_path server] Action server started")

    def handle_accepted(self, goal_handle) -> None:
        # SimpleActionServer semantics: a new goal preempts the running one.
        with self.goal_lock:
            if self.goal_handle is not None and self.goal_handle.is_active:
                self.goal_handle.abort()
            self.goal_handle = goal_handle
        goal_handle.execute()

    def is_preempt_requested(self, goal_handle) -> bool:
        return goal_handle.is_cancel_requested or not goal_handle.is_active

    def path_cb(self, msg: Path) -> None:
        self.current_path = msg

    def do_path_following(self, goal_handle) -> bool:
        """
        Performs dynamic target following while tracking path progress and segment completion.
            The function continuously:
            - computes the dynamic target ahead of the vehicle.
            - tracks the overall path completion.

        Returns:
            bool: True if the entire path is completed successfully, False if interrupted or failed.
        """
        try:
            while rclpy.ok():
                if self.is_preempt_requested(goal_handle):
                    self.get_logger().debug("Path following preempted")
                    return False

                if self.current_path is None:
                    self.get_logger().debug("No path received yet. Waiting for path...")
                    self.loop_rate.sleep()
                    continue

                # Publish the target path for visualization
                self.path_pub.publish(self.current_path)

                # get current pose of robot
                robot_pose = follow_path_helpers.get_robot_pose(
                    self, self.tf_buffer, self.source_frame
                )
                if robot_pose is None:
                    self.get_logger().warning("Failed to get current pose. Retrying...")
                    self.loop_rate.sleep()
                    continue

                dynamic_target_pose = follow_path_helpers.calculate_dynamic_target(
                    self.current_path,
                    robot_pose,
                    self.dynamic_target_lookahead_distance,
                )
                if dynamic_target_pose is None:
                    self.get_logger().warning(
                        "Failed to calculate dynamic target. Retrying..."
                    )
                    self.loop_rate.sleep()
                    continue

                # Broadcast the dynamic target frame so that controllers can use it
                follow_path_helpers.broadcast_dynamic_target_frame(
                    self,
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
                    self.get_logger().info(" [FollowPathActionServer] Path completed!")
                    return True

                feedback = FollowPath.Feedback()
                goal_handle.publish_feedback(feedback)

                self.loop_rate.sleep()

            return False

        except Exception as e:
            self.get_logger().error(f"Error during path following: {e}")
            return False

    def execute(self, goal_handle) -> FollowPath.Result:
        self.get_logger().debug(
            "FollowPathActionServer: Received a new path following goal."
        )
        self.current_path = None  # Clear the path on new goal

        success = self.do_path_following(goal_handle)
        result = FollowPath.Result(success=success)

        if success:
            self.get_logger().debug(f"Path following succeeded.")
            goal_handle.succeed()
        else:
            self.get_logger().debug(f"Path following did not succeed.")
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            elif goal_handle.is_active:
                goal_handle.abort()
        return result


def main():
    ros.init()
    ros.spin(FollowPathActionServer())


if __name__ == "__main__":
    main()
