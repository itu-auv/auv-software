#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np
import yaml
from auv_localization.srv import CalibrateIMU, CalibrateIMUResponse
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class ImuToOdom:
    def __init__(self):
        rospy.init_node("imu_to_odom_node", anonymous=True)

        self.imu_calibration_data_path = rospy.get_param(
            "~imu_calibration_path", "config/imu_calibration_data.yaml"
        )
        # Subscribers and Publishers
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )
        self.odom_publisher = rospy.Publisher("odom_imu", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        # Initialize covariances with zeros
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()

        # Variables for drift correction
        self.drift = np.zeros(3)
        self.calibrating = False
        self.calibration_data = []

        # Load calibration data if available
        self.load_calibration_data()

        # Service for IMU calibration
        self.calibration_service = rospy.Service(
            "calibrate_imu", CalibrateIMU, self.calibrate_imu
        )

    def imu_callback(self, imu_msg):
        if self.calibrating:
            self.calibration_data.append(
                [
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,
                ]
            )

        # Fill the odometry message with orientation and angular velocity data from the IMU
        self.odom_msg.header.stamp = rospy.Time.now()

        # Orientation
        self.odom_msg.pose.pose.orientation = imu_msg.orientation
        # Only the first 9 elements for orientation
        self.odom_msg.pose.covariance[:9] = list(imu_msg.orientation_covariance)

        # Angular Velocity
        corrected_angular_velocity = Vector3(
            imu_msg.angular_velocity.x - self.drift[0],
            imu_msg.angular_velocity.y - self.drift[1],
            imu_msg.angular_velocity.z - self.drift[2],
        )
        self.odom_msg.twist.twist.angular = corrected_angular_velocity
        # Angular velocity covariance in the correct position
        self.odom_msg.twist.covariance[21:30] = list(
            imu_msg.angular_velocity_covariance
        )

        # Set the position to zero as we are not computing it here
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        # Set linear velocity to zero as we are not using it here
        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    def calibrate_imu(self, req):
        duration = req.duration
        rospy.loginfo(f"IMU Calibration started for {duration} seconds...")
        self.calibrating = True
        self.calibration_data = []

        # Wait for the calibration duration
        rospy.sleep(duration)

        self.calibrating = False
        if len(self.calibration_data) > 0:
            self.drift = np.mean(self.calibration_data, axis=0)
            self.save_calibration_data()
            rospy.loginfo(f"IMU Calibration completed. Drift: {self.drift}")
            return CalibrateIMUResponse(
                success=True, message="IMU Calibration successful"
            )
        else:
            return CalibrateIMUResponse(
                success=False, message="IMU Calibration failed, no data recorded"
            )

    def save_calibration_data(self):
        calibration_data = {"drift": self.drift.tolist()}
        with open(self.imu_calibration_data_path, "w") as f:
            yaml.dump(calibration_data, f)
        rospy.loginfo(
            f"{TerminalColors.OKGREEN} Calibration data saved.{TerminalColors.ENDC}"
        )

    def load_calibration_data(self):
        try:
            with open(self.imu_calibration_data_path, "r") as f:
                calibration_data = yaml.safe_load(f)
                self.drift = np.array(calibration_data["drift"])
            rospy.loginfo(
                f"{TerminalColors.OKYELLOW}IMU Calibration data loaded.{TerminalColors.ENDC} Drift: {self.drift}"
            )
        except FileNotFoundError:
            rospy.logerr("No calibration data found.")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        imu_to_odom = ImuToOdom()
        imu_to_odom.run()
    except rospy.ROSInterruptException:
        pass
