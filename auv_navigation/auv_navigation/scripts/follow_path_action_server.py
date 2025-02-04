#!/usr/bin/env python3
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
from auv_navigation import follow_path_utils

class FollowPathActionServer:
    def __init__(self) -> None:
        # tf2 objects
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # load parameters
        self.dynamic_target_lookahead_distance: float = rospy.get_param(
            '~dynamic_target_lookahead_distance', 1.0
            )
        self.source_frame: str = rospy.get_param('~source_frame', "taluy/base_link")
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 20))
        
        self.path_pub = rospy.Publisher('target_path', Path, queue_size=1)
        
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
                
                # get current pose
                current_pose = follow_path_utils.get_current_pose(
                    self.tf_buffer, self.source_frame
                    )
                if current_pose is None:
                    rospy.logwarn("Failed to get current pose. Retrying...")
                    self.loop_rate.sleep()
                    continue
                
                dynamic_target_pose = follow_path_utils.calculate_dynamic_target(
                    path, current_pose, self.dynamic_target_lookahead_distance
                    )
                if dynamic_target_pose is None:
                    rospy.logwarn("Failed to calculate dynamic target. Retrying...")
                    self.loop_rate.sleep()
                    continue
                
                # Broadcast the dynamic target frame so that controllers can use it
                follow_path_utils.broadcast_dynamic_target_frame(
                    self.tf_broadcaster, self.tf_buffer, self.source_frame, dynamic_target_pose
                )

                # Check progress along the current segment and overall path
                current_segment_progress, overall_progress = follow_path_utils.check_segment_progress(
                    path, current_pose, current_segment_index, segment_endpoints
                )
                feedback = FollowPathFeedback()
                feedback.current_segment_progress = current_segment_progress
                feedback.overall_progress = overall_progress
                feedback.current_segment_index = current_segment_index
                self.server.publish_feedback(feedback)
                
                # Check if current segment is completed
                segment_end_index = segment_endpoints[current_segment_index]
                if follow_path_utils.is_segment_completed(
                    current_pose,
                    path,
                    segment_end_index
                ):
                    if current_segment_index < num_segments - 1:
                        current_segment_index += 1
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
        combined_path, segment_endpoints = follow_path_utils.combine_segments(goal.paths)
        
        rospy.logdebug(f"combined path type: {type(combined_path)}")
        
        if not combined_path.poses:
            rospy.logwarn("Combined path has no poses")
            self.server.set_aborted(FollowPathResult(success=False, execution_time=0.0))
            return
            
        start_time = rospy.Time.now()
        success = self.do_path_following(combined_path, segment_endpoints)
        execution_time = (rospy.Time.now() - start_time).to_sec()
        result = FollowPathResult(success=success, execution_time=execution_time)
        
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
