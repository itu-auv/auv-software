#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import tf
import tf2_geometry_msgs
from typing import Optional, List, Tuple
import angles

ZERO_DISTANCE_TOLERANCE: float = 1e-6  # Minimum distance to consider a path segment non-zero

def get_current_pose(tf_buffer: tf2_ros.Buffer, source_frame: str) -> Optional[PoseStamped]:
    try:
        transform = tf_buffer.lookup_transform(
            "odom",
            source_frame,
            rospy.Time(0),  
            rospy.Duration(1.0)
        )
        pose = PoseStamped()
        pose.header.frame_id = "odom"
        pose.header.stamp = transform.header.stamp 
        pose.pose.position = transform.transform.translation
        pose.pose.orientation = transform.transform.rotation
        return pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get current pose: {e}")
        return None

def calculate_dynamic_target(path: Path, robot_pose: PoseStamped, 
                        dynamic_target_lookahead_distance: float) -> Optional[PoseStamped]:
    
    if not path.poses or dynamic_target_lookahead_distance <= 0:
        return None

    closest_index = find_closest_point_index(path, robot_pose)

    # If closest to last point, return it
    if closest_index >= len(path.poses) - 1:
        return path.poses[-1]
    
    # Walk along path segments until we've consumed dynamic_target_lookahead_distance
    remaining_distance = dynamic_target_lookahead_distance
    current_index = closest_index
    
    while remaining_distance > 0 and current_index < len(path.poses) - 1:
        segment_start = path.poses[current_index].pose
        segment_end = path.poses[current_index + 1].pose
        
        segment_distance = np.linalg.norm(
            np.array([
                segment_end.position.x - segment_start.position.x,
                segment_end.position.y - segment_start.position.y,
                segment_end.position.z - segment_start.position.z
            ])
        )
        
        # Skip zero-length segments
        if segment_distance < ZERO_DISTANCE_TOLERANCE:
            current_index += 1
            continue

        # If we can place target on this segment
        if remaining_distance <= segment_distance:
            # interpolation of position
            dynamic_target_pose = PoseStamped()
            dynamic_target_pose.header = path.header
            ratio = remaining_distance / segment_distance
            dynamic_target_pose.pose.position.x = segment_start.position.x + \
                ratio * (segment_end.position.x - segment_start.position.x)
            dynamic_target_pose.pose.position.y = segment_start.position.y + \
                ratio * (segment_end.position.y - segment_start.position.y)
            dynamic_target_pose.pose.position.z = segment_start.position.z + \
                ratio * (segment_end.position.z - segment_start.position.z)
            
            # Use end segment orientation as dynamic target orientation (precise enough)
            dynamic_target_pose.pose.orientation.x = segment_end.orientation.x
            dynamic_target_pose.pose.orientation.y = segment_end.orientation.y
            dynamic_target_pose.pose.orientation.z = segment_end.orientation.z
            dynamic_target_pose.pose.orientation.w = segment_end.orientation.w
            
            return dynamic_target_pose
        
        remaining_distance -= segment_distance # Move to next segment
        current_index += 1
    
    # If we've consumed all segments, return last pose
    return path.poses[-1]


def broadcast_dynamic_target_frame(tf_broadcaster: tf2_ros.TransformBroadcaster, 
                         tf_buffer: tf2_ros.Buffer,
                         source_frame: str,
                         dynamic_target_pose: PoseStamped) -> None:
    try:
        odom_to_source = tf_buffer.lookup_transform(
            source_frame, 
            "odom",  
            rospy.Time(0),  
            rospy.Duration(1.0)
        )
        dynamic_target_in_source = tf2_geometry_msgs.do_transform_pose(dynamic_target_pose, odom_to_source)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = "dynamic_target"
        t.transform.translation.x = dynamic_target_in_source.pose.position.x
        t.transform.translation.y = dynamic_target_in_source.pose.position.y
        t.transform.translation.z = dynamic_target_in_source.pose.position.z
        t.transform.rotation = dynamic_target_in_source.pose.orientation

        tf_broadcaster.sendTransform(t)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to broadcast dynamic target frame: {e}")

def find_closest_point_index(path: Path, current_pose: PoseStamped) -> int:        
    min_dist = float('inf')
    closest_index = 0
    
    for i, pose in enumerate(path.poses):
        dist = np.linalg.norm(
            np.array([
                pose.pose.position.x - current_pose.pose.position.x,
                pose.pose.position.y - current_pose.pose.position.y,
                pose.pose.position.z - current_pose.pose.position.z
            ])
        )
        if dist < min_dist:
            min_dist = dist
            closest_index = i
            
    return closest_index

def is_path_completed(current_pose: PoseStamped,
                path: Path, path_end_index: int) -> bool:
    closest_index = find_closest_point_index(path, current_pose)
    #! delete rospy.loginfo(f"Closest index: {closest_index}, path end index: {path_end_index}")
    return closest_index >= path_end_index

def combine_paths(paths: List[Path]) -> Tuple[Path, List[int]]:
    """
    Combines multiple paths into a single path while keeping track of segment endpoints.
    
    Args:
        paths: List of Path messages to combine
        
    Returns:
        tuple containing:
            - Combined Path message
            - List of indices where each original path ends in the combined path
    """
    if not paths:
        return Path(), []
        
    combined_path = Path()
    combined_path.header = paths[0].header
    path_endpoints = []
    current_length = 0
    
    for path in paths:
        if not path.poses:
            continue
            
        # Add all poses from current path
        combined_path.poses.extend(path.poses)
        
        # Store the endpoint of this path segment
        current_length += len(path.poses)
        path_endpoints.append(current_length - 1)
    
    return combined_path, path_endpoints

def check_path_progress(path: Path, current_pose: PoseStamped,
                        current_path_index: int, path_endpoints: List[int]):
    """
    Args:
        path (Path): The combined path being followed.
        current_pose: Current pose of the robot
        current_path_index (int): The index of the current path segment.
        path_endpoints (List[int]): Indices marking the endpoints of individual path segments.

    Returns:
        Tuple[float, float]: (current path progress, overall progress)
    """
    try:
        closest_index = find_closest_point_index(path, current_pose)
        
        path_start_index = 0 if current_path_index == 0 else path_endpoints[current_path_index - 1] + 1
        path_end_index = path_endpoints[current_path_index]
        closest_index = min(closest_index, path_end_index)

        path_length = max(1, (path_end_index - path_start_index))  # Avoid division by zero
        current_path_progress = (closest_index - path_start_index) / path_length
        current_path_progress = max(0.0, min(1.0, current_path_progress))

        # Compute overall progress
        overall_progress = closest_index / max(1, len(path.poses)) # Avoid division by zero
        overall_progress = max(0.0, min(1.0, overall_progress))

        return current_path_progress, overall_progress
    except Exception as e:
        rospy.logerr(f"Error computing path progress: {e}")
        return 0.0, 0.0
