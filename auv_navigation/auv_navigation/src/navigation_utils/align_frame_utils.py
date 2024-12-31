#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

class AlignFrameController:
    def __init__(self, max_linear_velocity=0.8, max_angular_velocity=0.9, 
                    kp=0.55, angle_kp=0.45):
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.kp = kp
        self.angle_kp = angle_kp
        self.enable_pub = rospy.Publisher("taluy/enable", Bool, queue_size=1) 
    
    def constrain(self, value, max_value):
        return max(min(value, max_value), -max_value)
    
    def wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def get_error(
        self, tf_buffer, source_frame, target_frame, angle_offset=0.0, keep_orientation=False
        ):
        try:
            transform = tf_buffer.lookup_transform(
                source_frame,
                target_frame,
                rospy.Time(0),
                rospy.Duration(2.0)
            )
            
            trans_error = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            )
            rot = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )

            _, _, yaw_error = euler_from_quaternion(rot)
            
            # Apply angle offset and handle keep orientation
            yaw_error = self.wrap_angle(yaw_error + angle_offset)
            if keep_orientation:
                yaw_error = 0.0
            
            return trans_error, yaw_error
            
        except Exception as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None, None

    def compute_cmd_vel(self, trans_error, yaw_error):
        twist = Twist()
        
        kp_linear = self.kp
        twist.linear.x = self.constrain(trans_error[0] * kp_linear, self.max_linear_velocity)
        twist.linear.y = self.constrain(trans_error[1] * kp_linear, self.max_linear_velocity)
        twist.linear.z = self.constrain(trans_error[2] * kp_linear, self.max_linear_velocity)
        
        # Angular velocity
        kp_angular = self.angle_kp
        twist.angular.z = self.constrain(yaw_error * kp_angular, self.max_angular_velocity)
        
        return twist
    
    def is_aligned(self, position_threshold, angle_threshold, current_pose,
                    path, trans_error, rot_error):
        if path is None:
            total_trans_error = np.linalg.norm(trans_error)
            within_position = (total_trans_error <= position_threshold)
            within_angle = (abs(rot_error) <= angle_threshold)
            return within_position and within_angle
        
        # For path-based alignment, calculate error between current pose and last path pose
        last_pose = path.poses[-1].pose

        dx = last_pose.position.x - current_pose.pose.position.x
        dy = last_pose.position.y - current_pose.pose.position.y
        dz = last_pose.position.z - current_pose.pose.position.z
        total_pos_error = np.linalg.norm([dx, dy, dz])
        
        _, _, current_yaw = euler_from_quaternion([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])
        
        _, _, last_yaw = euler_from_quaternion([
            last_pose.orientation.x,
            last_pose.orientation.y,
            last_pose.orientation.z,
            last_pose.orientation.w
        ])
        
        yaw_error = self.wrap_angle(last_yaw - current_yaw)
        rospy.loginfo(f"Path alignment - Position error: {total_pos_error:.4f}, "
                    f"Angle error: {yaw_error:.4f}") #TODO change to debug
        
        return (total_pos_error <= position_threshold and 
                abs(yaw_error) <= angle_threshold)
    
    def enable_alignment(self):
        self.enable_pub.publish(Bool(True))
