#!/usr/bin/env python3

import rospy
import tf2_ros
import tf
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from typing import Optional, List
def straight_path_to_frame(
    tf_buffer: tf2_ros.Buffer, 
    source_frame: str, 
    target_frame: str, 
    angle_offset: float = 0.0, 
    keep_orientation: bool = False, 
    num_samples: int = 50,
    n_turns: int = 0
) -> Optional[Path]:
    """
    Creates a straight path from source frame to target frame.
    
    Args:
        tf_buffer:
        source_frame:
        target_frame:
        angle_offset: Additional angle offset to add to target orientation
        keep_orientation: If True, keep source orientation throughout path
        num_samples: Number of waypoints to generate
        n_turns: Number of full 360-degree turns to add
        
    Returns:
        Path message if successful, None otherwise
    """
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

def path_for_gate(
    tf_buffer: tf2_ros.Buffer,
    path_creation_timeout: float = 10.0
) -> Optional[List[Path]]:
    """
    Plans paths for the gate task, which includes the two paths:
    1. Current position to gate entrance
    2. Gate entrance to gate exit (with 1 360 degrees turn)
    """
    try:
        rospy.loginfo("[GatePathPlanner] Planning paths for gate task...")
        start_time = rospy.Time.now()
        
        # Plan path to gate entrance
        entrance_path = None
        exit_path = None
        
        while (rospy.Time.now() - start_time).to_sec() < path_creation_timeout:
            try:
                # create the first segment
                if entrance_path is None:
                    entrance_path = straight_path_to_frame(
                        tf_buffer=tf_buffer,
                        source_frame="taluy/base_link",
                        target_frame="gate_enterance"
                    )
                #create the second segment
                if exit_path is None:
                    exit_path = straight_path_to_frame(
                        tf_buffer=tf_buffer,
                        source_frame="gate_enterance",
                        target_frame="gate_exit",
                        n_turns=1
                    )
                #TODO (somebody): fix the typo: gate_enterance -> gate_entrance
                    
                if entrance_path is not None and exit_path is not None:
                    return [entrance_path, exit_path]
                    
                rospy.logwarn("[GatePathPlanner] Failed to plan paths, retrying... Time elapsed: %.1f seconds", 
                             (rospy.Time.now() - start_time).to_sec())
                rospy.sleep(1.0)
                
            except (tf2_ros.LookupException, 
                    tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("[GatePathPlanner] TF error while planning paths: %s. Retrying...", str(e))
                rospy.sleep(1.0)
        
        # If we get here, we timed out
        rospy.logwarn("[GatePathPlanner] Failed to plan paths after %.1f seconds", path_creation_timeout)
        return None
        
    except Exception as e:
        rospy.logwarn("[GatePathPlanner] Error in gate path planning: %s", str(e))
        return None