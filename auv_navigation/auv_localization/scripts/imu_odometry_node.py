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

    def insert_covariance_block(
        self,
        base_covariance_6x6,
        input_covariance_3x3_flat,
    ):
        """
        Inserts a 3x3 covariance block (provided as a flat list) into a 6x6 NumPy matrix
        and returns the resulting flattened 6x6 list.

        The 3x3 block is always inserted into the bottom-right quadrant (indices 3:6, 3:6),
        which corresponds to orientation in pose covariance or angular velocity in twist covariance.
        """
        updated_covariance_6x6 = base_covariance_6x6.copy()

        # Only insert the covariance block if it has 9 elements
        if len(input_covariance_3x3_flat) == 9:

            input_covariance_3x3_matrix = np.array(input_covariance_3x3_flat).reshape(
                3, 3
            )

            # Insert the 3x3 matrix into the bottom-right quadrant (angular/orientation part)
            updated_covariance_6x6[3:6, 3:6] = input_covariance_3x3_matrix
        else:
            rospy.logwarn_throttle(
                3,
                f"Received invalid IMU covariance size: {len(input_covariance_3x3_flat)}. "
                f"Using default covariance",
            )
        return updated_covariance_6x6.flatten().tolist()

    def update_pose_covariance(self, imu_orientation_covariance):
        return self.insert_covariance_block(
            self.default_pose_cov, imu_orientation_covariance
        )

    def update_twist_covariance(self, imu_angular_velocity_covariance):
        return self.insert_covariance_block(
            self.default_twist_cov, imu_angular_velocity_covariance
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

        self.odom_msg.header.stamp = imu_msg.header.stamp

        #! Remove these
        # Add proper timestamp logging for debugging
        current_time = rospy.Time.now()
        time_diff = (current_time - imu_msg.header.stamp).to_sec()

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
