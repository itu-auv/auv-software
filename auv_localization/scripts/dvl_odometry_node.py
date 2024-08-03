#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import math
import message_filters
import time

class DvlToOdom:
    def __init__(self):
        rospy.init_node('dvl_to_odom_node', anonymous=True)

        # Parameters for first-order filter
        self.tau = rospy.get_param('~tau', 0.1)
        
        # Subscribers and Publishers
        self.dvl_velocity_subscriber = message_filters.Subscriber("sensors/dvl/velocity_raw", Twist)
        self.is_valid_subscriber = message_filters.Subscriber("sensors/dvl/is_valid", Bool)
        self.cmd_vel_subscriber = rospy.Subscriber("cmd_vel", Twist, self.cmd_vel_callback)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.dvl_velocity_subscriber, self.is_valid_subscriber], queue_size=10, slop=0.1, allow_headerless=True)
        self.sync.registerCallback(self.dvl_callback)

        self.odom_publisher = rospy.Publisher("localization/odom_dvl", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = 'odom'
        self.odom_msg.child_frame_id = 'taluy/base_link'

        # Initialize covariances with default values
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()

        # Load calibration data
        self.load_calibration_data()

        # Initialize variables for fallback mechanism
        self.cmd_vel_twist = Twist()
        self.filtered_cmd_vel = Twist()
        self.last_update_time = rospy.Time.now()

    def transform_vector(self, vector):
        theta = np.radians(-135)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(rotation_matrix, np.array(vector))

    def cmd_vel_callback(self, cmd_vel_msg):
        self.cmd_vel_twist = cmd_vel_msg

    def filter_cmd_vel(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        self.alpha = dt / (self.tau + dt)

        self.filtered_cmd_vel.linear.x = self.filtered_cmd_vel.linear.x * (1.0 - self.alpha) + self.alpha * self.cmd_vel_twist.linear.x
        self.filtered_cmd_vel.linear.y = self.filtered_cmd_vel.linear.y * (1.0 - self.alpha) + self.alpha * self.cmd_vel_twist.linear.y
        self.filtered_cmd_vel.linear.z = self.filtered_cmd_vel.linear.z * (1.0 - self.alpha) + self.alpha * self.cmd_vel_twist.linear.z
        self.filtered_cmd_vel.angular.x = self.filtered_cmd_vel.angular.x * (1.0 - self.alpha) + self.alpha * self.cmd_vel_twist.angular.x
        self.filtered_cmd_vel.angular.y = self.filtered_cmd_vel.angular.y * (1.0 - self.alpha) + self.alpha * self.cmd_vel_twist.angular.y
        self.filtered_cmd_vel.angular.z = self.filtered_cmd_vel.angular.z * (1.0 - self.alpha) + self.alpha * self.cmd_vel_twist.angular.z

        self.last_update_time = current_time

    def dvl_callback(self, velocity_msg, is_valid_msg):
        self.filter_cmd_vel()
        # Determine which data to use for odometry based on DVL validity
        if is_valid_msg.data:
            rotated_vector = self.transform_vector([velocity_msg.linear.x, velocity_msg.linear.y, velocity_msg.linear.z])
            velocity_msg.linear.x = rotated_vector[0]
            velocity_msg.linear.y = rotated_vector[1]
            velocity_msg.linear.z = rotated_vector[2]
            self.last_valid_velocity = velocity_msg
        else:
            velocity_msg.linear.x = self.filtered_cmd_vel.linear.x
            velocity_msg.linear.y = self.filtered_cmd_vel.linear.y
            velocity_msg.linear.z = self.filtered_cmd_vel.linear.z
        
        # Fill the odometry message
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.twist.twist = velocity_msg

        # Set position to zero as we are not computing it here
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    def load_calibration_data(self):
        config_path = rospy.get_param(
            '~config_path', 'config/calibration_data.yaml')
        try:
            with open(config_path, 'r') as f:
                calibration_data = yaml.safe_load(f)
                self.odom_msg.twist.covariance[0] = calibration_data.get(
                    'twist_covariance_linear_x', 0.01)
                self.odom_msg.twist.covariance[7] = calibration_data.get(
                    'twist_covariance_linear_y', 0.01)
                self.odom_msg.twist.covariance[14] = calibration_data.get(
                    'twist_covariance_linear_z', 0.01)
                rospy.loginfo("Calibration data loaded. Twist covariance linear x: {}, linear y: {}, linear z: {}".format(
                    self.odom_msg.twist.covariance[0], self.odom_msg.twist.covariance[7], self.odom_msg.twist.covariance[14]))
        except FileNotFoundError:
            rospy.loginfo(
                "No calibration data found. Using default covariances.")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        dvl_to_odom = DvlToOdom()
        dvl_to_odom.run()
    except rospy.ROSInterruptException:
        pass
