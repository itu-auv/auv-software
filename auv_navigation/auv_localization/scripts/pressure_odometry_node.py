#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
import yaml
import auv_common_lib.transform.transformer
from auv_common_lib.logging.terminal_color_utils import TerminalColors
import math
from geometry_msgs.msg import Quaternion
import tf.transformations


class PressureToOdom:
    def __init__(self):
        rospy.init_node("pressure_to_odom_node", anonymous=True)

        self.depth_calibration_offset = rospy.get_param(
            "sensors/external_pressure_sensor/depth_offset", 0.0
        )
        self.depth_calibration_covariance = rospy.get_param(
            "sensors/external_pressure_sensor/depth_covariance", 0.00005
        )
        self.odom_publisher = rospy.Publisher("odom_pressure", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        self.transformer = auv_common_lib.transform.transformer.Transformer()

        # Initialize covariances with zeros
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.pose.covariance[14] = self.depth_calibration_covariance

        # Log loaded parameters
        pressure_odometry_colored = TerminalColors.color_text(
            "Pressure Odometry Calibration data loaded", TerminalColors.PASTEL_GREEN
        )
        rospy.loginfo(
            f"{pressure_odometry_colored} : depth offset: {self.depth_calibration_offset}"
        )
        rospy.loginfo(
            f"{pressure_odometry_colored} : depth covariance: {self.depth_calibration_covariance}"
        )

        self.imu_data = None
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )

        self.depth_subscriber = rospy.Subscriber(
            "depth", Float32, self.depth_callback, tcp_nodelay=True
        )

        self.base_to_pressure_translation = None
        rate = rospy.Rate(1.0)

    def imu_callback(self, imu_msg):
        self.imu_data = imu_msg

    def get_base_to_pressure_height(self):
        # Try to fetch and cache the TF once
        if self.base_to_pressure_translation is None:
            try:
                trans, _ = self.transformer.get_transform(
                    "taluy/base_link", "taluy/base_link/external_pressure_sensor_link"
                )
                arr = np.array(trans)
                # flatten any nested structure to 1D [x, y, z]
                self.base_to_pressure_translation = arr.flatten()
            except Exception as e:
                rospy.logwarn_throttle(
                    10, f"Pressure TF not available: {e}; using zero offset."
                )
                return 0.0

        # If no IMU data arrived yet, use the static Z offset
        if self.imu_data is None:
            rospy.logwarn_throttle(
                10, "No IMU data received yet. Using default orientation."
            )
            return float(self.base_to_pressure_translation[2])

        # Compute rotated Z-offset based on current orientation
        orientation = self.imu_data.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(quat)[:3, :3]
        rotated_vector = rotation_matrix.dot(self.base_to_pressure_translation)
        return float(rotated_vector[2])

    def depth_callback(self, depth_msg):
        # Fill the odometry message with depth data as the z component of the linear position
        self.odom_msg.header.stamp = rospy.Time.now()

        calibrated_depth = depth_msg.data + self.depth_calibration_offset

        base_depth = calibrated_depth + self.get_base_to_pressure_height()

        # Set the z component in pose position to the depth value
        self.odom_msg.pose.pose.position.z = base_depth

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

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        pressure_to_odom = PressureToOdom()
        pressure_to_odom.run()
    except rospy.ROSInterruptException:
        pass
