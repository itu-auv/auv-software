#!/usr/bin/env python

import rospy

# Import TwistWithCovarianceStamped, keep Twist for cmd_vel and fallback types
from geometry_msgs.msg import Twist, TwistWithCovarianceStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import numpy as np
import yaml  # Assuming this might still be needed for future param loading
import math
import message_filters
import time

# Keep the import, but we won't add extra error handling for now
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class DvlToOdom:
    def __init__(self):
        rospy.init_node("dvl_to_odom_node", anonymous=True)

        # Parameters for first-order filter (Original)
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
        # *** CHANGE 1: Subscribe to the stamped topic ***
        self.dvl_velocity_subscriber = message_filters.Subscriber(
            "dvl/velocity_raw_stamped",
            TwistWithCovarianceStamped,
            tcp_nodelay=True,  # Changed topic and type
        )
        self.is_valid_subscriber = message_filters.Subscriber(
            "dvl/is_valid", Bool, tcp_nodelay=True
        )
        self.cmd_vel_subscriber = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
        )

        # *** CHANGE 2: Update Synchronizer for the new type ***
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.dvl_velocity_subscriber, self.is_valid_subscriber],
            queue_size=10,
            slop=0.1,
            allow_headerless=False,  # Prefer headers now that velocity has one
            # Test if True is needed if is_valid is truly headerless
        )
        self.sync.registerCallback(self.dvl_callback)

        self.odom_publisher = rospy.Publisher("odom_dvl", Odometry, queue_size=10)

        # Initialize the odometry message (Original)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        # Initialize covariances with default values (Original)
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()

        self.odom_msg.twist.covariance[0] = self.linear_x_covariance
        self.odom_msg.twist.covariance[7] = self.linear_y_covariance
        self.odom_msg.twist.covariance[14] = self.linear_z_covariance

        # Log loaded parameters (Original)
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

        # Initialize variables for fallback mechanism (Original)
        self.cmd_vel_twist = Twist()
        self.filtered_cmd_vel = Twist()
        self.last_update_time = rospy.Time.now()  # Keep original name

    # Original transform function
    def transform_vector(self, vector):
        theta = np.radians(-135)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(rotation_matrix, np.array(vector))

    # Original cmd_vel callback
    def cmd_vel_callback(self, cmd_vel_msg):
        self.cmd_vel_twist = cmd_vel_msg

    # Original filter function (no robustness check added)
    def filter_cmd_vel(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        # Original alpha calculation - potential issue if dt or tau is zero remains
        if dt <= 0 or self.cmdvel_tau <= 0:  # Basic check to prevent division by zero
            alpha = 1.0
        else:
            alpha = dt / (self.cmdvel_tau + dt)

        self.filtered_cmd_vel.linear.x = (
            self.filtered_cmd_vel.linear.x * (1.0 - alpha)
            + alpha * self.cmd_vel_twist.linear.x
        )
        self.filtered_cmd_vel.linear.y = (
            self.filtered_cmd_vel.linear.y * (1.0 - alpha)
            + alpha * self.cmd_vel_twist.linear.y
        )
        self.filtered_cmd_vel.linear.z = (
            self.filtered_cmd_vel.linear.z * (1.0 - alpha)
            + alpha * self.cmd_vel_twist.linear.z
        )
        self.filtered_cmd_vel.angular.x = (
            self.filtered_cmd_vel.angular.x * (1.0 - alpha)
            + alpha * self.cmd_vel_twist.angular.x
        )
        self.filtered_cmd_vel.angular.y = (
            self.filtered_cmd_vel.angular.y * (1.0 - alpha)
            + alpha * self.cmd_vel_twist.angular.y
        )
        self.filtered_cmd_vel.angular.z = (
            self.filtered_cmd_vel.angular.z * (1.0 - alpha)
            + alpha * self.cmd_vel_twist.angular.z
        )

        self.last_update_time = current_time

    # *** CHANGE 3: Update callback signature ***
    def dvl_callback(self, velocity_stamped_msg, is_valid_msg):  # Renamed first arg
        self.filter_cmd_vel()

        # Store the twist part of the input message for convenience
        # *** CHANGE 4: Access twist data correctly from stamped msg ***
        current_dvl_twist = velocity_stamped_msg.twist.twist

        # Create a Twist message to eventually put into odom_msg
        output_twist = Twist()

        if is_valid_msg.data:
            # *** CHANGE 5: Use data from current_dvl_twist ***
            rotated_vector = self.transform_vector(
                [
                    current_dvl_twist.linear.x,
                    current_dvl_twist.linear.y,
                    current_dvl_twist.linear.z,
                ]
            )
            output_twist.linear.x = rotated_vector[0]
            output_twist.linear.y = rotated_vector[1]
            output_twist.linear.z = rotated_vector[2]
            output_twist.angular = (
                current_dvl_twist.angular
            )  # Use angular from valid DVL
            # self.last_valid_velocity = velocity_msg # Original line commented out, adapt if needed later
        else:
            # Fallback uses filtered cmd_vel linear, but angular comes from the invalid DVL message
            output_twist.linear.x = self.filtered_cmd_vel.linear.x
            output_twist.linear.y = self.filtered_cmd_vel.linear.y
            output_twist.linear.z = self.filtered_cmd_vel.linear.z
            # *** CHANGE 6: Still use angular from (invalid) DVL message struct ***
            output_twist.angular = current_dvl_twist.angular

        # Fill the odometry message
        # *** CHANGE 7: Use the timestamp from the DVL header ***
        self.odom_msg.header.stamp = velocity_stamped_msg.header.stamp
        # Assign the determined twist (either DVL-based or fallback-based)
        self.odom_msg.twist.twist = output_twist

        # Set position to zero (Original)
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    # Original run method
    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        # Keep original import structure
        try:
            from auv_common_lib.logging.terminal_color_utils import TerminalColors
        except ImportError:
            rospy.logwarn(
                "Package 'auv_common_lib' not found. Proceeding without colored logging."
            )

            class TerminalColors:  # Minimal fallback
                PASTEL_BLUE = ""

                @staticmethod
                def color_text(text, color):
                    return text

        dvl_to_odom = DvlToOdom()
        dvl_to_odom.run()
    except rospy.ROSInterruptException:
        pass
