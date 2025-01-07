#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import angles
import tf2_ros

class FrameAligner:
    def __init__(self, max_linear_velocity=0.8, max_angular_velocity=0.9, 
                    linear_kp=0.55, angular_kp=0.45):

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.source_frame = ""
        self.target_frame = ""
        self.angle_offset = 0.0
        self.keep_orientation = False

        # Control parameters
        self.linear_kp = rospy.get_param("~linear_kp", 0.5)
        self.angular_kp = rospy.get_param("~angular_kp", 0.8)
        self.max_linear_velocity = rospy.get_param("~max_linear_velocity", 0.8)
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity", 0.9)

    def enable_alignment(self):
        self.enable_pub.publish(Bool(data=True))
        
    def get_error(self,
                  tf_buffer: tf2_ros.Buffer,
                  source_frame: str,
                  target_frame: str,
                  angle_offset: float,
                  keep_orientation: bool,
                  time: rospy.Time = rospy.Time(0)
    ):
        try:
            # Get the current transform
            transform = tf_buffer.lookup_transform(
                source_frame,
                target_frame,
                time,
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
            _, _, yaw_error = tf.transformations.euler_from_quaternion(rot)
            
            # Apply angle offset and handle keep orientation
            yaw_error = angles.normalize_angle(yaw_error + angle_offset)
            if keep_orientation:
                yaw_error = 0.0
            
            return trans_error, yaw_error
            
        except Exception as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None, None

    def constrain(self, value, max_value):
        return max(min(value, max_value), -max_value)
    
    def compute_cmd_vel(self, trans_error, yaw_error):
        twist = Twist()
        linear_kp = self.linear_kp
        # Set linear velocities based on translation differences
        twist.linear.x = self.constrain(trans_error[0] * linear_kp, self.max_linear_velocity)
        twist.linear.y = self.constrain(trans_error[1] * linear_kp, self.max_linear_velocity)
        twist.linear.z = self.constrain(trans_error[2] * linear_kp, self.max_linear_velocity)
        angular_kp = self.angular_kp
        twist.angular.z = self.constrain(yaw_error * angular_kp, self.max_angular_velocity)

        return twist

    def step(self):
        """Execute a single step of frame alignment.
        """
        trans_error, yaw_error = self.get_error(
            self.tf_buffer,
            self.source_frame,
            self.target_frame,
            self.angle_offset,
            self.keep_orientation  
        )
        
        if trans_error is None or yaw_error is None:
            return False

        # Compute and publish velocity commands
        cmd_vel = self.compute_cmd_vel(trans_error, yaw_error)
        self.cmd_vel_pub.publish(cmd_vel)
        return True
