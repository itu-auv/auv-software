#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import math
import message_filters
import time
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class DvlToOdom:
    def __init__(self):
        rospy.init_node("dvl_to_odom_node", anonymous=True)

        self.enabled = rospy.get_param("~enabled", True)
        self.enable_service = rospy.Service(
            "dvl_to_odom_node/enable", SetBool, self.enable_cb
        )

        self.cmdvel_tau = rospy.get_param("~cmdvel_tau", 0.1)
        self.linear_x_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_x", 0.000015
        )
        self.linear_y_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_y", 0.000015
        )
        self.linear_z_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_z", 0.00005
        )

        # Subscribers and Publishers
        self.dvl_velocity_subscriber = message_filters.Subscriber(
            "dvl/velocity_raw", Twist, tcp_nodelay=True
        )
        self.is_valid_subscriber = message_filters.Subscriber(
            "dvl/is_valid", Bool, tcp_nodelay=True
        )
        self.cmd_vel_subscriber = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.dvl_velocity_subscriber, self.is_valid_subscriber],
            queue_size=10,
            slop=0.1,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.dvl_callback)

        self.odom_publisher = rospy.Publisher("odom_dvl", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        # Initialize covariances with default values
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance[0] = self.linear_x_covariance
        self.odom_msg.twist.covariance[7] = self.linear_y_covariance
        self.odom_msg.twist.covariance[14] = self.linear_z_covariance

        # Logging
        DVL_odometry_colored = TerminalColors.color_text(
            "DVL Odometry Calibration data loaded", TerminalColors.PASTEL_BLUE
        )
        rospy.loginfo(f"{DVL_odometry_colored} : cmdvel_tau: {self.cmdvel_tau}")
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear x covariance: {self.linear_x_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear y covariance: {self.linear_y_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear z covariance: {self.linear_z_covariance}"
        )

        # Fallback variables
        self.cmd_vel_twist = Twist()
        self.filtered_cmd_vel = Twist()
        self.last_update_time = rospy.Time.now()
        self.is_dvl_enabled = False

    def enable_cb(self, req):
        """Service callback to enable/disable DVL->Odom processing"""
        self.enabled = req.data
        state = "enabled" if self.enabled else "disabled"
        rospy.loginfo(f"DVL->Odom node {state} via service call.")
        return SetBoolResponse(success=True, message=f"DVL->Odom {state}")

    def transform_vector(self, vector):
        theta = np.radians(-45)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(rotation_matrix, np.array(vector))

    def cmd_vel_callback(self, cmd_vel_msg):
        self.cmd_vel_twist = cmd_vel_msg

    def filter_cmd_vel(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        self.alpha = dt / (self.cmdvel_tau + dt)

        self.filtered_cmd_vel.linear.x = (
            self.filtered_cmd_vel.linear.x * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.linear.x
        )
        self.filtered_cmd_vel.linear.y = (
            self.filtered_cmd_vel.linear.y * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.linear.y
        )
        self.filtered_cmd_vel.linear.z = (
            self.filtered_cmd_vel.linear.z * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.linear.z
        )
        self.filtered_cmd_vel.angular.x = (
            self.filtered_cmd_vel.angular.x * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.angular.x
        )
        self.filtered_cmd_vel.angular.y = (
            self.filtered_cmd_vel.angular.y * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.angular.y
        )
        self.filtered_cmd_vel.angular.z = (
            self.filtered_cmd_vel.angular.z * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.angular.z
        )

        self.last_update_time = current_time

    def dvl_callback(self, velocity_msg, is_valid_msg):
        self.is_dvl_enabled = True

        if not self.enabled:
            return

        self.filter_cmd_vel()
        # Determine which data to use for odometry based on DVL validity
        if is_valid_msg.data:
            rotated_vector = self.transform_vector(
                [velocity_msg.linear.x, velocity_msg.linear.y, velocity_msg.linear.z]
            )
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

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.is_dvl_enabled:
                self.odom_msg.header.stamp = rospy.Time.now()
                self.odom_msg.twist.twist = Twist()
                self.odom_publisher.publish(self.odom_msg)
            rate.sleep()


if __name__ == "__main__":
    try:
        dvl_to_odom = DvlToOdom()
        dvl_to_odom.run()
    except rospy.ROSInterruptException:
        pass
