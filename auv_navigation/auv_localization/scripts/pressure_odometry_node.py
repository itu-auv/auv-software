#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
import message_filters
import auv_common_lib.transform.transformer
from auv_common_lib.logging.terminal_color_utils import TerminalColors
import tf.transformations


class PressureToOdom:
    def __init__(self):
        rospy.init_node("pressure_to_odom_node", anonymous=True)

        # Sensor calibration and parameters
        self.depth_calibration_offset = rospy.get_param("~depth_offset", 0.0)
        self.depth_calibration_covariance = rospy.get_param(
            "~depth_covariance", 0.00005
        )
        self.pool_depth = rospy.get_param("/env/pool_depth", 2.2)
        self.min_valid_altitude = rospy.get_param("~min_valid_altitude", 0.3)
        self.max_valid_altitude = rospy.get_param("~max_valid_altitude", 2.2)
        self.min_valid_depth = rospy.get_param("~min_valid_depth", 0.0)
        self.max_valid_depth = rospy.get_param("~max_valid_depth", 2.2)
        self.dvl_frame_id = rospy.get_param("~dvl_frame_id", "taluy/base_link/dvl_link")

        self.odom_publisher = rospy.Publisher("odom_pressure", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        self.transformer = auv_common_lib.transform.transformer.Transformer()

        # Initialize covariances
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
        rospy.loginfo(f"{pressure_odometry_colored} : pool_depth: {self.pool_depth}")

        self.imu_data = None
        self.base_to_pressure_translation = None
        self.base_to_dvl_translation = None

        # Subscribers
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )
        depth_sub = message_filters.Subscriber("depth", Float32)
        dvl_sub = message_filters.Subscriber("dvl/altitude", Float32)

        # Synchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, dvl_sub], queue_size=10, slop=0.1
        )
        ts.registerCallback(self.fused_depth_callback)

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
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
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

    def get_base_to_dvl_height(self):
        # Try to fetch and cache the TF once
        if self.base_to_dvl_translation is None:
            try:
                trans, _ = self.transformer.get_transform(
                    "taluy/base_link", self.dvl_frame_id
                )
                arr = np.array(trans)
                self.base_to_dvl_translation = arr.flatten()
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    10, f"DVL TF not available: {e}; using zero offset."
                )
                return 0.0

        if self.imu_data is None:
            rospy.logwarn_throttle(
                10, "No IMU data for DVL height calculation. Using default orientation."
            )
            return float(self.base_to_dvl_translation[2])

        orientation = self.imu_data.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(quat)[:3, :3]
        rotated_vector = rotation_matrix.dot(self.base_to_dvl_translation)
        return float(rotated_vector[2])

    def fused_depth_callback(self, depth_msg, dvl_msg):
        # 1. Get calibrated pressure depth
        pressure_depth_calibrated = depth_msg.data + self.depth_calibration_offset
        pressure_depth = pressure_depth_calibrated + self.get_base_to_pressure_height()

        # 2. Convert DVL altitude to depth
        dvl_altitude = dvl_msg.data
        dvl_depth = self.pool_depth - dvl_altitude - self.get_base_to_dvl_height()

        # 3. Validate both sensor readings
        is_pressure_valid = (
            self.min_valid_depth <= pressure_depth <= self.max_valid_depth
        )
        is_dvl_valid = (
            self.min_valid_altitude <= dvl_altitude <= self.max_valid_altitude
        )

        final_depth = 0.0
        publish = True

        # 4. Fusion Logic
        if is_pressure_valid and is_dvl_valid:
            final_depth = (pressure_depth + dvl_depth) / 2.0
        elif is_pressure_valid:
            rospy.logwarn_throttle(
                5, "DVL data is out of range, using only pressure data."
            )
            final_depth = pressure_depth
        elif is_dvl_valid:
            rospy.logwarn_throttle(
                5, "Pressure data is out of range, using only DVL data."
            )
            final_depth = dvl_depth
        else:
            rospy.logerr_throttle(
                5, "Both pressure and DVL data are out of range. No odometry published."
            )
            publish = False

        if publish:
            self.odom_msg.header.stamp = rospy.Time.now()
            self.odom_msg.pose.pose.position.z = final_depth

            # Set other components to zero as they are not provided by these sensors
            self.odom_msg.pose.pose.position.x = 0.0
            self.odom_msg.pose.pose.position.y = 0.0
            self.odom_msg.twist.twist.linear.x = 0.0
            self.odom_msg.twist.twist.linear.y = 0.0
            self.odom_msg.twist.twist.linear.z = 0.0
            self.odom_msg.twist.twist.angular.x = 0.0
            self.odom_msg.twist.twist.angular.y = 0.0
            self.odom_msg.twist.twist.angular.z = 0.0

            self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        pressure_to_odom = PressureToOdom()
        pressure_to_odom.run()
    except rospy.ROSInterruptException:
        pass
