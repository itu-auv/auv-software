#!/usr/bin/env python3
import rospy
import actionlib
from auv_msgs.msg import FollowPathAction, FollowPathGoal
from typing import List
from nav_msgs.msg import Path
class FollowPathActionClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient("taluy/follow_path", FollowPathAction)
        
        rospy.logdebug("[follow_path client] waiting for action server...")
        self.client.wait_for_server()
        rospy.logdebug("[follow_path client] action server is up!")

    def execute_paths(self, paths: List[Path]) -> bool:
        """
        Sends a lists of Path messages to the follow_path action server for execution.
        Args:
            paths: List of Path messages to execute
        Returns:
            bool: True if execution completes successfully (server returns SUCCESS), False otherwise.
        """
        try:
            goal = FollowPathGoal()
            goal.paths = [path for path in paths]
            rospy.logdebug("[follow_path client] sending paths goal...")
            self.client.send_goal(goal)

            rospy.logdebug("[follow_path client] Waiting for result...")
            self.client.wait_for_result()
            result = self.client.get_result()
            
            if result and result.success:
                rospy.logdebug("[follow_path client] Task succeeded: execution time: %.2f seconds", result.execution_time)
                return True
            else:
                rospy.logwarn("[follow_path client] Task failed")
                return False

        except Exception as e:
            rospy.logerr("[follow_path client] Error executing paths: %s", str(e))
            return False
