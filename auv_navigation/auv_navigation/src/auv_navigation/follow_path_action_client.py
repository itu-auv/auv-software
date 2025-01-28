#!/usr/bin/env python3

import rospy
import actionlib
import tf2_ros
from auv_msgs.msg import FollowPathAction, FollowPathGoal
from auv_navigation import path_utils

class FollowPathActionClient:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.client = actionlib.SimpleActionClient("taluy/follow_path", FollowPathAction)
        
        rospy.loginfo("[Action Client] Waiting for 'follow_path' action server...")
        self.client.wait_for_server()
        rospy.loginfo("[Action Client] Action server is up!")

    def execute_path(self, path):
        """
        Send a given path to the follow_path action server.
        Args:
            path: Path message to execute
        Returns:
            bool: True if execution completes successfully
        """
        try:
            goal = FollowPathGoal()
            goal.path = path
            rospy.loginfo("[Action Client] Sending path goal...")
            self.client.send_goal(goal)

            rospy.loginfo("[Action Client] Waiting for result...")
            finished = self.client.wait_for_result()

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

    def navigate_to_frame(self, source_frame, target_frame, n_turns=0, path_creation_timeout=10.0):

        try:
            rospy.loginfo("[Action Client] Creating path from %s to %s...", 
                         source_frame, target_frame)
            
            start_time = rospy.Time.now()
            path = None
            
            while (rospy.Time.now() - start_time).to_sec() < path_creation_timeout:
                try:
                    path = path_utils.straight_path_to_frame(
                        tf_buffer=self.tf_buffer,
                        source_frame=source_frame,
                        target_frame=target_frame,
                        n_turns=n_turns
                    )
                    if path is not None:
                        # Print yaw angles for all waypoints
                        path_utils.print_path_yaws(path) # !Delete, for debugging
                        break
                    rospy.logwarn("[Action Client] Failed to create path, retrying... Time elapsed: %.1f seconds", 
                                 (rospy.Time.now() - start_time).to_sec())
                    rospy.sleep(1.0)  # Wait 1 second before retrying
                except (tf2_ros.LookupException, 
                        tf2_ros.ConnectivityException, 
                        tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn("[Action Client] TF error while creating path: %s. Retrying...", str(e))
                    rospy.sleep(1.0)  # Wait 1 second before retrying
            
            if path is None:
                rospy.logerr("[Action Client] Failed to create path after %.1f seconds", path_creation_timeout)
                return False
            
            return self.execute_path(path)
            
        except Exception as e:
            rospy.logerr("[Action Client] Error in navigate_to_frame: %s", str(e))
            return False

    def cancel_current_goal(self):
        """Cancel any ongoing navigation."""
        self.client.cancel_all_goals()
