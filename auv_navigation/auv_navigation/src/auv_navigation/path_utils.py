#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import tf
import tf2_geometry_msgs
from typing import Optional
import angles

ZERO_DISTANCE_TOLERANCE: float = 1e-6  # Minimum distance to consider a path segment non-zero

def straight_path_to_frame(tf_buffer: tf2_ros.Buffer, 
                          source_frame: str, 
                          target_frame: str, 
                          angle_offset: float = 0.0, 
                          keep_orientation: bool = False, 
                          num_samples: int = 50,
                          n_turns: int = 0) -> Optional[Path]:
    try:
        source_transform = tf_buffer.lookup_transform(
            "odom", 
            source_frame, 
            rospy.Time(0),  
            rospy.Duration(1.0)
        )
        target_transform = tf_buffer.lookup_transform(
            "odom", 
            target_frame, 
            rospy.Time(0),  
            rospy.Duration(1.0)
        )
        source_position = source_transform.transform.translation
        target_position = target_transform.transform.translation
        
        source_quaternion = [source_transform.transform.rotation.x,
                source_transform.transform.rotation.y,
                source_transform.transform.rotation.z,
                source_transform.transform.rotation.w]
        target_quaternion = [target_transform.transform.rotation.x,
                target_transform.transform.rotation.y,
                target_transform.transform.rotation.z,
                target_transform.transform.rotation.w]
        
        t_vals = np.linspace(0, 1, num_samples) 
        
        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = rospy.Time.now()
        
        # Apply angle offset
        offset_quaternion = tf.transformations.quaternion_from_euler(0, 0, angle_offset)
        final_quaternion = tf.transformations.quaternion_multiply(offset_quaternion, target_quaternion)
        
        # Calculate angular difference including n_turns
        source_euler = tf.transformations.euler_from_quaternion(source_quaternion)
        target_euler = tf.transformations.euler_from_quaternion(final_quaternion)
        angular_diff = (target_euler[2] - source_euler[2]) + (2 * np.pi * n_turns)
        interpolated_euler = [0, 0, 0]
        
        for t in t_vals:
            pose = PoseStamped()
            pose.header = path.header
            
            pose.pose.position.x = source_position.x + t * (target_position.x - source_position.x)
            pose.pose.position.y = source_position.y + t * (target_position.y - source_position.y)
            pose.pose.position.z = source_position.z + t * (target_position.z - source_position.z)
            
            # Set orientation based on keep_orientation flag
            if keep_orientation:
                interp_quaternion = source_quaternion
            else:
                interpolated_euler[2] = source_euler[2] + t * angular_diff
                interp_quaternion = tf.transformations.quaternion_from_euler(
                    source_euler[0], source_euler[1], interpolated_euler[2]
                )
            
            pose.pose.orientation.x = interp_quaternion[0]
            pose.pose.orientation.y = interp_quaternion[1]
            pose.pose.orientation.z = interp_quaternion[2]
            pose.pose.orientation.w = interp_quaternion[3]
            
            path.poses.append(pose)
        
        return path
    
    except Exception as e:
        rospy.logerr(f"Error in create_straight_path: {e}")
        return None


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

def is_path_completed(position_threshold: float, angle_threshold: float, current_pose: PoseStamped,
                path: Path):
    
    # calculate error between current pose and last path pose
    last_pose = path.poses[-1].pose

    dx = last_pose.position.x - current_pose.pose.position.x
    dy = last_pose.position.y - current_pose.pose.position.y
    dz = last_pose.position.z - current_pose.pose.position.z
    #total_pos_error = np.linalg.norm([dx, dy, dz])
    total_pos_error = np.linalg.norm([dx, dy])
    
    _, _, current_yaw = tf.transformations.euler_from_quaternion([
        current_pose.pose.orientation.x,
        current_pose.pose.orientation.y,
        current_pose.pose.orientation.z,
        current_pose.pose.orientation.w
    ])
    
    _, _, last_yaw = tf.transformations.euler_from_quaternion([
        last_pose.orientation.x,
        last_pose.orientation.y,
        last_pose.orientation.z,
        last_pose.orientation.w
    ])
    
    yaw_error = angles.normalize_angle(last_yaw - current_yaw)
    rospy.logdebug(f"Path alignment - Position error: {total_pos_error:.4f}, "
                f"Angle error: {yaw_error:.4f}") 
    
    return (total_pos_error <= position_threshold and 
            abs(yaw_error) <= angle_threshold)

def print_path_yaws(path: Path) -> None: # !Delete  debugging              
    """
    Print yaw angles for all waypoints in the path.
    Args:
        path: Path message containing the waypoints
    """
    if not path or not path.poses:
        rospy.logwarn("Path is empty or None")
        return
    
    rospy.loginfo("Yaw angles for waypoints (in degrees):")
    for i, pose in enumerate(path.poses):
        quaternion = [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        ]
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw_deg = np.degrees(euler[2])
        rospy.loginfo(f"Waypoint {i}: {yaw_deg:.2f}°")
