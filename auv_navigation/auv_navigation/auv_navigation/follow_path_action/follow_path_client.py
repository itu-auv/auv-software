#!/usr/bin/env python3
from rclpy.action import ActionClient
from auv_msgs.action import FollowPath
from typing import List
from nav_msgs.msg import Path


class FollowPathActionClient:
    def __init__(self, node):
        self.node = node

        self.client = ActionClient(node, FollowPath, "follow_path")

        self.node.get_logger().debug(
            "[follow_path client] waiting for action server..."
        )
        self.client.wait_for_server()
        self.node.get_logger().debug("[follow_path client] action server is up!")

    def execute_paths(self, paths: List[Path]) -> bool:
        """
        Sends a lists of Path messages to the follow_path action server for execution.
        Args:
            paths: List of Path messages to execute
        Returns:
            bool: True if execution completes successfully (server returns SUCCESS), False otherwise.
        """
        try:
            goal = FollowPath.Goal()
            goal.paths = paths
            self.node.get_logger().debug("[follow_path client] sending paths goal...")
            self.node.get_logger().debug("[follow_path client] Waiting for result...")
            result = self.client.send_goal(goal).result

            if result and result.success:
                self.node.get_logger().debug("[follow_path client] Task succeeded")
                return True
            else:
                self.node.get_logger().warning("[follow_path client] Task failed")
                return False

        except Exception as e:
            self.node.get_logger().error(
                "[follow_path client] Error executing paths: %s" % (str(e),)
            )
            return False
