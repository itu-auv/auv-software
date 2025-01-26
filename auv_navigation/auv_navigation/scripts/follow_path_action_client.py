#!/usr/bin/env python3

import rospy
import actionlib
import tf2_ros
from auv_msgs.msg import FollowPathAction, FollowPathGoal
from auv_navigation import path_planners

class FollowPathActionClient:
    def __init__(self, server_wait_timeout=10.0):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.client = actionlib.SimpleActionClient("follow_path", FollowPathAction)
        
        rospy.loginfo("[Action Client] Waiting for 'follow_path' action server...")
        server_found = self.client.wait_for_server(timeout=rospy.Duration(server_wait_timeout))
        if not server_found:
            raise rospy.ROSException("Action server not available within timeout")
        rospy.loginfo("[Action Client] Action server is up!")

    def execute_path(self, path, timeout=30.0):
        """
        Send a given path to the follow_path action server.
        Args:
            path: Path message to execute
            timeout: How long to wait for result in seconds
        Returns:
            bool: True if execution completes successfully
        """
        try:
            goal = FollowPathGoal()
            goal.path = path
            rospy.loginfo("[Action Client] Sending path goal...")
            self.client.send_goal(goal)

            rospy.loginfo("[Action Client] Waiting for result...")
            finished = self.client.wait_for_result(timeout=rospy.Duration(timeout))

            if not finished:
                rospy.logwarn("[Action Client] Action did not finish before timeout")
                self.client.cancel_goal()
                return False

            result = self.client.get_result()
            if result:
                rospy.loginfo("[Action Client] Task succeeded!")
                return True
            else:
                rospy.logwarn("[Action Client] Task failed")
                return False

        except Exception as e:
            rospy.logerr("[Action Client] Error executing path: %s", str(e))
            return False

    def navigate_to_frame(self, source_frame, target_frame, timeout=30.0):
        """
        Navigate from source_frame to target_frame.
        Args:
            source_frame: Starting frame
            target_frame: Goal frame
            timeout: How long to wait for result in seconds
        Returns:
            bool: True if navigation succeeds
        """
        try:
            rospy.loginfo("[Action Client] Creating path from %s to %s...", 
                         source_frame, target_frame)
            
            path = path_planners.create_path_from_frame(
                tf_buffer=self.tf_buffer,
                source_frame=source_frame,
                target_frame=target_frame
            )
            
            if path is None:
                rospy.logerr("[Action Client] Failed to create path")
                return False
                
            return self.execute_path(path, timeout=timeout)
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("[Action Client] TF error: %s", str(e))
            return False
        except Exception as e:
            rospy.logerr("[Action Client] Error in navigate_to_frame: %s", str(e))
            return False

    def cancel_current_goal(self):
        """Cancel any ongoing navigation."""
        self.client.cancel_all_goals()

if __name__ == '__main__':
    rospy.init_node('follow_path_action_client')
    try:
        client = FollowPathActionClient()
        success = client.navigate_to_frame("base_link", "target_frame")
        if success:
            rospy.loginfo("Navigation succeeded")
        else:
            rospy.loginfo("Navigation failed")
    except rospy.ROSException as e:
        rospy.logerr("Failed to initialize client: %s", str(e))
