#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

class AlignFrameController:
    def __init__(self, max_linear_velocity=0.8, max_angular_velocity=0.9, 
                    kp=0.55, angle_kp=0.45):
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.kp = kp
        self.angle_kp = angle_kp

        self.enable_pub = rospy.Publisher("taluy/enable", Bool, queue_size=1) 

    def killswitch_callback(self, msg):
        if not msg.data:
            self.active = False 
    
    def get_transform(
        self, tf_buffer, source_frame, target_frame
        ):
        try:
            transform = tf_buffer.lookup_transform(
                source_frame,
                target_frame,
                rospy.Time(0),
                rospy.Duration(2.0)
            )
            
            trans = (
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
            return trans, rot
        
        except Exception as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None, None

    def constrain(self, value, max_value):
        return max(min(value, max_value), -max_value)

    def compute_cmd_vel(self, trans, rot):
        
        #TODO add kd. 
        
        twist = Twist()

        kp_linear = self.kp
        kp_angular = self.angle_kp

        vx = trans[0] * kp_linear
        vy = trans[1] * kp_linear
        vz = trans[2] * kp_linear

        twist.linear.x = self.constrain(vx, self.max_linear_velocity)
        twist.linear.y = self.constrain(vy, self.max_linear_velocity)
        twist.linear.z = self.constrain(vz, self.max_linear_velocity)

        _, _, yaw = euler_from_quaternion(rot)
        twist.angular.z = self.constrain(yaw * kp_angular, self.max_angular_velocity)
        
        return twist

    def is_aligned(self, trans, rot, position_threshold, angle_threshold):
        _, _, yaw = euler_from_quaternion(rot)
        position_error = np.sqrt(trans[0]**2 + trans[1]**2 + trans[2]**2)
        return position_error < position_threshold and abs(yaw) < angle_threshold

    def enable_alignment(self):
        self.enable_pub.publish(Bool(True))



