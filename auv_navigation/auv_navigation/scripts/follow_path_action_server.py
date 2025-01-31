#!/usr/bin/env python3
import rospy
import actionlib
from nav_msgs.msg import Path
from auv_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathActionGoal
import tf2_ros
from auv_navigation import path_utils
from typing import List

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
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 20))

        self.path_pub = rospy.Publisher('target_path', Path, queue_size=1)
        self.server = actionlib.SimpleActionServer(
            'follow_path',
            FollowPathAction,
            self.execute,
            auto_start=False
        )
        self.server.start()
        

    def do_path_following(self, path: Path, path_endpoints: List[int]) -> bool:
        """
        Performs dynamic target following while tracking path progress and segment completion.
            The function continuously:
            - computes the dynamic target ahead of the vehicle.
            - tracks whether each segment of the path (that is, the individual paths before they were combined)
            has been completed.
            - tracks the overall path completion.   

        Args:
            path (Path): The combined path to be followed.
            path_endpoints (List[int]): Indices marking the endpoints of individual path segments.

        Returns:
            bool: True if the entire path is completed successfully, False if interrupted or failed.
        """

        try:
            n_paths = len(path_endpoints)
            current_path_index = 0
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.server.set_preempted()
                    rospy.logdebug("Path following preempted")
                    return False

                self.path_pub.publish(path)
                current_pose = path_utils.get_current_pose(self.tf_buffer, self.source_frame)
                if current_pose is None:
                    rospy.logwarn("Failed to get current pose. Retrying...")
                    self.loop_rate.sleep()
                    continue
                
                dynamic_target_pose = path_utils.calculate_dynamic_target(path, current_pose, self.dynamic_target_lookahead_distance)
                if dynamic_target_pose is None:
                    rospy.logwarn("Failed to calculate dynamic target. Retrying...")
                    self.loop_rate.sleep()
                    continue
                
                # Publish dynamic target frame    
                path_utils.broadcast_dynamic_target_frame(self.tf_broadcaster, self.tf_buffer, self.source_frame, dynamic_target_pose)
                
                # (AlignFrameController would be requested in smach to follow dynamic target)
                
                # Feedback
                current_path_progress, overall_progress = path_utils.check_path_progress(
                    path,
                    current_pose,
                    current_path_index,
                    path_endpoints
                )
                feedback = FollowPathFeedback()
                feedback.current_path_progress = current_path_progress
                feedback.overall_progress = overall_progress
                feedback.current_path_index = current_path_index
                self.server.publish_feedback(feedback)
                
                # Check if current segment is completed
                path_end_index = path_endpoints[current_path_index]
                if path_utils.is_path_completed(
                    current_pose,
                    path,
                    path_end_index
                ):
                    if current_path_index < n_paths - 1:
                        current_path_index += 1
                    else: # was on the last path and it's completed
                        rospy.logdebug("All paths completed")
                        return True
                self.loop_rate.sleep()
                
            return False
            
        except Exception as e:
            rospy.logerr(f"Error during path following: {e}")
            return False

    def execute(self, goal: FollowPathActionGoal) -> None:
        rospy.logdebug("FollowPathActionServer: Received a new path following goal.")
        
        if not goal.paths:
            rospy.logerr("Received empty paths list")
            self.server.set_aborted(FollowPathResult(success=False, execution_time=0.0))
            return
        # Combine all paths and get endpoints
        combined_path, path_endpoints = path_utils.combine_paths(goal.paths)
        
        rospy.logdebug(f"combined path type: {type(combined_path)}")
        
        if not combined_path.poses:
            rospy.logwarn("Combined path has no poses")
            self.server.set_aborted(FollowPathResult(success=False, execution_time=0.0))
            return
            
        start_time = rospy.Time.now()
        success = self.do_path_following(combined_path, path_endpoints)
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
