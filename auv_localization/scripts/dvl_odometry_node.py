#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import yaml


class DvlToOdom:
    def __init__(self):
        rospy.init_node('dvl_to_odom_node', anonymous=True)

        # Subscribers and Publishers
        self.velocity_subscriber = rospy.Subscriber(
            "sensors/dvl/velocity_raw", Twist, self.velocity_callback)
        self.odom_publisher = rospy.Publisher(
            "localization/odom_dvl", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = 'odom'
        self.odom_msg.child_frame_id = 'base_link'

        # Initialize covariances with default values
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()

        # Load calibration data
        self.load_calibration_data()

    def velocity_callback(self, velocity_msg):
        # Fill the odometry message with DVL velocity data
        self.odom_msg.header.stamp = rospy.Time.now()

        # Set the twist data to the odometry message
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
