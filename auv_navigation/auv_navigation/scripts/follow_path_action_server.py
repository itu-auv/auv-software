#!/usr/bin/env python3
import rospy
import actionlib
from nav_msgs.msg import Path
from auv_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult
import tf2_ros

from auv_navigation import path_utils


class FollowPathActionServer:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.dynamic_target_lookahead_distance = rospy.get_param('~dynamic_target_lookahead_distance', 1.0)
        self.alignment_distance_tolerance = rospy.get_param('~alignment_distance_tolerance', 0.1)
        self.alignment_yaw_tolerance = rospy.get_param('~alignment_yaw_tolerance', 0.1)
        self.source_frame = rospy.get_param('~source_frame', "taluy/base_link")
        self.dynamic_target_frame = rospy.get_param('~dynamic_target_frame', "dynamic_target")
        self.control_rate = rospy.Rate(rospy.get_param('~control_rate', 20))

        self.path_pub = rospy.Publisher('target_path', Path, queue_size=1)
        self.server = actionlib.SimpleActionServer(
            'follow_path',
            FollowPathAction,
            self.execute,
            auto_start=False
        )
        self.server.start()
        

    def do_path_following(self, path):
        """
        Returns true if the path was successfully followed to completion,
        false if interrupted (preempted, shutdown, or error occurred).
        """
        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.server.set_preempted()
                    rospy.logdebug("Path following preempted")
                    return False

                self.path_pub.publish(path)
                current_pose = path_utils.get_current_pose(self.tf_buffer, self.source_frame)
                if current_pose is None:
                    rospy.logwarn("Failed to get current pose. Retrying...")
                    self.control_rate.sleep()
                    continue
                
                dynamic_target_pose = path_utils.calculate_dynamic_target(path, current_pose, self.dynamic_target_lookahead_distance)
                if dynamic_target_pose is None:
                    rospy.logwarn("Failed to calculate dynamic target. Retrying...")
                    self.control_rate.sleep()
                    continue
                
                # Publish dynamic target frame    
                path_utils.broadcast_dynamic_target_frame(self.tf_broadcaster, self.tf_buffer, self.source_frame, dynamic_target_pose)
                
                # (AlignFrameController requested in smach to follow dynamic target)
                
                # Check if we've completed the path
                if path_utils.is_path_completed(
                    self.alignment_distance_tolerance,
                    self.alignment_yaw_tolerance,
                    current_pose,
                    path
                ):
                    rospy.logdebug("Path following is complete")
                    return True
                
                feedback = FollowPathFeedback()          
                # Calculate progress based on closest point in path
                closest_idx = path_utils.find_closest_point_index(path, current_pose)
                progress = float(closest_idx) / (len(path.poses) - 1) if len(path.poses) > 1 else 1.0
                feedback.progress = progress
                self.server.publish_feedback(feedback)
                
                self.control_rate.sleep() # Sleep to maintain control rate. control rate
                
            return False # If exited the loop, success is False
            
        except Exception as e:
            rospy.logerr(f"Error during path following: {e}")
            return False

    def execute(self, goal):
        rospy.logdebug("FollowPathActionServer: Received a new path following goal.")
        
        if not goal.path or not goal.path.poses:
            rospy.logerr("Received invalid path")
            self.server.set_aborted(FollowPathResult(success=False, execution_time=0.0))
            return
            
        start_time = rospy.Time.now() 
        success = self.do_path_following(path=goal.path)
        execution_time = (rospy.Time.now() - start_time).to_sec()
        
        result = FollowPathResult(success=success, execution_time=execution_time)
        
        if success:
            rospy.logdebug(f"Path following succeeded. Execution time: {execution_time:.2f} seconds")
            self.server.set_succeeded(result)
        else:
            rospy.logdebug(f"Path following did not succeed. Execution time: {execution_time:.2f} seconds")
            if not self.server.is_preempt_requested():
                self.server.set_aborted(result)

if __name__ == "__main__":
    rospy.init_node('follow_path_action_server')
    FollowPathActionServer()
    rospy.spin()
