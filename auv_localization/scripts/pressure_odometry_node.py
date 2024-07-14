#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import yaml


class PressureToOdom:
    def __init__(self):
        rospy.init_node('pressure_to_odom_node', anonymous=True)

        # Subscribers and Publishers
        self.depth_subscriber = rospy.Subscriber(
            "sensors/external_pressure_sensor/depth", Float32, self.depth_callback)
        self.odom_publisher = rospy.Publisher(
            "localization/odom_pressure", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = 'odom'
        self.odom_msg.child_frame_id = 'taluy/base_link'

        # Initialize covariances with zeros
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()

        # Load calibration data
        self.load_calibration_data()

    def depth_callback(self, depth_msg):
        # Fill the odometry message with depth data as the z component of the linear position
        self.odom_msg.header.stamp = rospy.Time.now()

        # Set the z component in pose position to the depth value
        self.odom_msg.pose.pose.position.z = depth_msg.data

        # Set other components to zero as we are not using them
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0

        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    def load_calibration_data(self):
        config_path = rospy.get_param(
            '~config_path', 'config/pressure_calibration.yaml')
        try:
            with open(config_path, 'r') as f:
                calibration_data = yaml.safe_load(f)
                self.odom_msg.pose.covariance[14] = calibration_data.get(
                    'pose_position_z_covariance', 0.0)
                rospy.loginfo("Calibration data loaded. Pose position Z covariance: {}".format(
                    self.odom_msg.pose.covariance[14]))
        except FileNotFoundError:
            rospy.loginfo("No calibration data found.")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        pressure_to_odom = PressureToOdom()
        pressure_to_odom.run()
    except rospy.ROSInterruptException:
        pass
