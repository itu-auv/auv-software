#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_multiply, quaternion_inverse, quaternion_slerp
from scipy.interpolate import CubicSpline
import tf2_geometry_msgs



def create_path_from_frame(tf_buffer, source_frame, target_frame, angle_offset=0.0, keep_orientation=False, num_samples=50):
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
        src_pos = source_transform.transform.translation
        tgt_pos = target_transform.transform.translation
        
        src_q = [source_transform.transform.rotation.x,
                source_transform.transform.rotation.y,
                source_transform.transform.rotation.z,
                source_transform.transform.rotation.w]
        tgt_q = [target_transform.transform.rotation.x,
                target_transform.transform.rotation.y,
                target_transform.transform.rotation.z,
                target_transform.transform.rotation.w]
        
        t_vals = np.linspace(0, 1, num_samples) 
        
        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = rospy.Time.now()
        
        # Apply angle offset
        offset_q = quaternion_from_euler(0, 0, angle_offset)
        final_q = quaternion_multiply(offset_q, tgt_q)
        
        for t in t_vals:
            pose = PoseStamped()
            pose.header = path.header
            
            pose.pose.position.x = src_pos.x + t * (tgt_pos.x - src_pos.x)
            pose.pose.position.y = src_pos.y + t * (tgt_pos.y - src_pos.y)
            pose.pose.position.z = src_pos.z + t * (tgt_pos.z - src_pos.z)
            
            # Set orientation based on keep_orientation flag
            if keep_orientation:
                interp_q = src_q
            else:
                interp_q = quaternion_slerp(src_q, final_q, t)
            
            pose.pose.orientation.x = interp_q[0]
            pose.pose.orientation.y = interp_q[1]
            pose.pose.orientation.z = interp_q[2]
            pose.pose.orientation.w = interp_q[3]
            
            path.poses.append(pose)
        
        return path
    
    except Exception as e:
        rospy.logerr(f"Error in create_straight_path_with_slerp: {e}")
        return None

def get_current_pose(tf_buffer, source_frame):
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

def calculate_carrot_pose(path, robot_pose, carrot_distance):
    if not path.poses or carrot_distance <= 0:
        return None

    min_dist = float('inf')
    closest_index = 0
    
    for i, pose in enumerate(path.poses):
        dist = np.linalg.norm(
            np.array([
                pose.pose.position.x - robot_pose.pose.position.x,
                pose.pose.position.y - robot_pose.pose.position.y,
                pose.pose.position.z - robot_pose.pose.position.z
            ])
        )
        if dist < min_dist:
            min_dist = dist
            closest_index = i

    # If closest to last point, return it
    if closest_index >= len(path.poses) - 1:
        return path.poses[-1]
    
    # Walk along path segments until we've consumed carrot_distance
    remaining_distance = carrot_distance
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
        if segment_distance < 1e-6:
            current_index += 1
            continue

        # If we can place carrot on this segment
        if remaining_distance <= segment_distance:
            # interpolation of position
            carrot_pose = PoseStamped()
            carrot_pose.header = path.header
            ratio = remaining_distance / segment_distance
            carrot_pose.pose.position.x = segment_start.position.x + \
                ratio * (segment_end.position.x - segment_start.position.x)
            carrot_pose.pose.position.y = segment_start.position.y + \
                ratio * (segment_end.position.y - segment_start.position.y)
            carrot_pose.pose.position.z = segment_start.position.z + \
                ratio * (segment_end.position.z - segment_start.position.z)
            
            # Use end segment orientation as carrot orientation
            carrot_pose.pose.orientation.x = segment_end.orientation.x
            carrot_pose.pose.orientation.y = segment_end.orientation.y
            carrot_pose.pose.orientation.z = segment_end.orientation.z
            carrot_pose.pose.orientation.w = segment_end.orientation.w
            
            return carrot_pose
        
        remaining_distance -= segment_distance # Move to next segment
        current_index += 1
    
    # If we've consumed all segments, return last pose
    return path.poses[-1]


def broadcast_carrot_frame(tf_broadcaster, tf_buffer, source_frame, carrot_pose):
    try:
        odom_to_source = tf_buffer.lookup_transform(
            source_frame, 
            "odom",  
            rospy.Time(0),  
            rospy.Duration(1.0)
        )
        carrot_in_source = tf2_geometry_msgs.do_transform_pose(carrot_pose, odom_to_source)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = "carrot"
        t.transform.translation.x = carrot_in_source.pose.position.x
        t.transform.translation.y = carrot_in_source.pose.position.y
        t.transform.translation.z = carrot_in_source.pose.position.z
        t.transform.rotation = carrot_in_source.pose.orientation

        tf_broadcaster.sendTransform(t)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to broadcast carrot frame: {e}")

