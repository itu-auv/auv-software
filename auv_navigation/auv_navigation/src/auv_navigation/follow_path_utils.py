#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import tf2_geometry_msgs
from typing import Optional, List, Tuple

ZERO_DISTANCE_TOLERANCE: float = 1e-6  # Minimum distance to consider a path segment non-zero
DYNAMIC_TARGET_FRAME: str = "dynamic_target"
ODOM_FRAME: str = "odom"
TIME_ZERO: rospy.Time = rospy.Time(0)
TF_LOOKUP_TIMEOUT: rospy.Duration = rospy.Duration(1.0)

def get_current_pose(tf_buffer: tf2_ros.Buffer, source_frame: str) -> Optional[PoseStamped]:
    try:
        transform = tf_buffer.lookup_transform(
            ODOM_FRAME,
            source_frame,
            TIME_ZERO,
            TF_LOOKUP_TIMEOUT
        )
        pose = PoseStamped()
        pose.header.frame_id = ODOM_FRAME
        pose.header.stamp = transform.header.stamp
        pose.pose.position = transform.transform.translation
        pose.pose.orientation = transform.transform.rotation
        return pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
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
        
        # Compute Euclidean distance between segment_start and segment_end.
        dx = segment_end.position.x - segment_start.position.x
        dy = segment_end.position.y - segment_start.position.y
        dz = segment_end.position.z - segment_start.position.z
        segment_distance = np.linalg.norm(np.array([dx, dy, dz]))
        
        # Skip zero-length segments
        if segment_distance < ZERO_DISTANCE_TOLERANCE:
            current_index += 1
            continue

        # If we can place target on this segment
        if remaining_distance <= segment_distance:
            ratio = remaining_distance / segment_distance
            dynamic_target_pose = PoseStamped()
            dynamic_target_pose.header = path.header
            dynamic_target_pose.pose.position.x = segment_start.position.x + ratio * dx
            dynamic_target_pose.pose.position.y = segment_start.position.y + ratio * dy
            dynamic_target_pose.pose.position.z = segment_start.position.z + ratio * dz
            # Use the orientation of the segment end.
            dynamic_target_pose.pose.orientation = segment_end.orientation
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
            ODOM_FRAME,
            TIME_ZERO,
            TF_LOOKUP_TIMEOUT
        )
        dynamic_target_in_source = tf2_geometry_msgs.do_transform_pose(
            dynamic_target_pose, odom_to_source
            )

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = DYNAMIC_TARGET_FRAME
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

def is_segment_completed(current_pose: PoseStamped,
                path: Path, segment_end_index: int) -> bool:
    closest_index = find_closest_point_index(path, current_pose)
    return closest_index >= segment_end_index

def combine_segments(paths: List[Path]) -> Tuple[Path, List[int]]:
    """
    Combines multiple path segments into a single path while keeping track of segment endpoints.
    
    Args:
        paths: List of Path messages to combine
        
    Returns:
        tuple containing:
            - Combined Path message
            - List of indices where each original path ends in the combined path
    """
    combined_path = Path()
    segment_endpoints: List[int] = []

    if not paths:
        return combined_path, segment_endpoints

    combined_path.header = paths[0].header
    current_length = 0

    for p in paths:
        if not p.poses:
            continue

        combined_path.poses.extend(p.poses)
        current_length += len(p.poses)
        segment_endpoints.append(current_length - 1)

    return combined_path, segment_endpoints

def check_segment_progress(path: Path, current_pose: PoseStamped,
                        current_segment_index: int, segment_endpoints: List[int]):
    """
    Args:
        path (Path): The combined path being followed.
        current_pose: Current pose of the robot
        current_segment_index (int): The index of the current path segment.
        segment_endpoints (List[int]): Indices marking the endpoints of individual path segments.

    Returns:
        Tuple[float, float]: (current path progress, overall progress)
    """
    try:
        closest_index = find_closest_point_index(path, current_pose)
        
        path_start_index = 0 if current_segment_index == 0 else segment_endpoints[current_segment_index - 1] + 1
        segment_end_index = segment_endpoints[current_segment_index]
        closest_index = min(closest_index, segment_end_index)

        path_length = max(1, (segment_end_index - path_start_index))  # Avoid division by zero
        current_path_progress = (closest_index - path_start_index) / path_length
        current_path_progress = max(0.0, min(1.0, current_path_progress))

        # Compute overall progress
        overall_progress = closest_index / max(1, len(path.poses)) # Avoid division by zero
        overall_progress = max(0.0, min(1.0, overall_progress))

        return current_path_progress, overall_progress
    except Exception as e:
        rospy.logerr(f"Error computing path progress: {e}")
        return 0.0, 0.0
