#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np
import yaml
from auv_localization.srv import CalibrateIMU, CalibrateIMUResponse
from auv_common_lib.logging.terminal_color_utils import TerminalColors


HIGH_COVARIANCE = 1e6


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
        self.odom_msg.child_frame_id = "taluy/base_link"  # TODO: NO absolute frames

        # Build default covariance matrices
        self.default_pose_cov = np.zeros((6, 6))
        self.default_pose_cov[:3, :3] = np.eye(3) * HIGH_COVARIANCE

        self.default_twist_cov = np.zeros((6, 6))
        self.default_twist_cov[:3, :3] = np.eye(3) * HIGH_COVARIANCE

        # Initialize covariances with the default matrices
        self.odom_msg.pose.covariance = self.default_pose_cov.flatten().tolist()
        self.odom_msg.twist.covariance = self.default_twist_cov.flatten().tolist()

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

    def update_pose_covariance(self, imu_orientation_covariance):
        """
        Update the 6x6 pose covariance matrix by inserting the 3x3 orientation
        covariance from the IMU message.
        """
        pose_covariance = self.default_pose_cov.copy()
        if len(imu_orientation_covariance) == 9:
            imu_orientation_covariance_matrix = np.array(
                imu_orientation_covariance
            ).reshape(3, 3)
            # Insert the IMU orientation covariance into the orientation block
            pose_covariance[3:6, 3:6] = imu_orientation_covariance_matrix
        else:
            rospy.logwarn_throttle(
                3,
                f"Received invalid IMU orientation covariance size: {len(imu_orientation_covariance)}. Expected 9 elements. Using default pose covariance.",
            )
        return pose_covariance.flatten().tolist()

    def update_twist_covariance(self, imu_angular_velocity_covariance):
        """
        Update the 6x6 twist covariance matrix by inserting the 3x3 angular velocity
        covariance from the IMU message.
        """
        twist_cov = self.default_twist_cov.copy()
        if len(imu_angular_velocity_covariance) == 9:
            imu_angular_velocity_covariance_matrix = np.array(
                imu_angular_velocity_covariance
            ).reshape(3, 3)
            # Insert the IMU angular velocity covariance into the angular velocity block
            twist_cov[3:6, 3:6] = imu_angular_velocity_covariance_matrix
        else:
            rospy.logwarn_throttle(
                3,
                f"Received invalid IMU angular velocity covariance size: {len(imu_angular_velocity_covariance)}. Expected 9 elements. Using default twist covariance.",
            )
        return twist_cov.flatten().tolist()

    def imu_callback(self, imu_msg):
        if self.calibrating:
            self.calibration_data.append(
                [
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,
                ]
            )

        self.odom_msg.header.stamp = rospy.Time.now()

        # Correct angular velocity using the drift
        corrected_angular_velocity = Vector3(
            imu_msg.angular_velocity.x - self.drift[0],
            imu_msg.angular_velocity.y - self.drift[1],
            imu_msg.angular_velocity.z - self.drift[2],
        )

        # Update twist and twist covariance
        self.odom_msg.twist.twist.angular = corrected_angular_velocity
        self.odom_msg.twist.covariance = self.update_twist_covariance(
            imu_msg.angular_velocity_covariance
        )

        # Update orientation and orientation covariance
        self.odom_msg.pose.pose.orientation = imu_msg.orientation
        self.odom_msg.pose.covariance = self.update_pose_covariance(
            imu_msg.orientation_covariance
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
