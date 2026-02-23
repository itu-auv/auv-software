#!/usr/bin/env python3

import rospy
import tf
import tf.transformations
import numpy as np
import yaml
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
from auv_localization.srv import CalibrateIMU, CalibrateIMUResponse

HIGH_COVARIANCE = 1e6


class ImuToOdom:
    def __init__(self):
        rospy.init_node("imu_to_odom_node", anonymous=True)

        self.namespace = rospy.get_param("~namespace", "taluy")
        self.imu_calibration_data_path = rospy.get_param(
            "~imu_calibration_path", "config/imu_calibration_data.yaml"
        )

        self.base_frame = rospy.get_param("~base_frame", f"{self.namespace}/base_link")
        self.imu_frame = rospy.get_param(
            "~imu_frame", f"{self.namespace}/base_link/imu"
        )

        self.imu_to_base_q = self.get_frame_rotation(self.imu_frame)

        # Subscribers and Publishers
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )
        self.odom_publisher = rospy.Publisher("odom_imu", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.base_frame

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

    def get_frame_rotation(self, frame_id):
        tf_listener = tf.TransformListener()
        try:
            tf_listener.waitForTransform(
                self.base_frame, frame_id, rospy.Time(0), rospy.Duration(2.0)
            )
            (trans, rot) = tf_listener.lookupTransform(
                self.base_frame, frame_id, rospy.Time(0)
            )
            rospy.loginfo(f"Loaded {frame_id} rotation from TF: {rot}")
            return np.array(rot)
        except Exception as e:
            rospy.logwarn(
                f"Could not get TF for {frame_id} relative to {self.base_frame}: {e}. No frame rotation will be applied."
            )
            return None

    def quaternion_multiply(self, q1, q2):
        x0, y0, z0, w0 = q1
        x1, y1, z1, w1 = q2

        x_new = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y_new = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        z_new = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        w_new = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1

        return np.array([x_new, y_new, z_new, w_new])

    def insert_covariance_block(self, base_covariance_6x6, input_covariance_3x3_flat):
        updated_covariance_6x6 = base_covariance_6x6.copy()
        if len(input_covariance_3x3_flat) == 9:
            input_covariance_3x3_matrix = np.array(input_covariance_3x3_flat).reshape(
                3, 3
            )
            updated_covariance_6x6[3:6, 3:6] = input_covariance_3x3_matrix
        return updated_covariance_6x6.flatten().tolist()

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

        corrected_angular_velocity = np.array(
            [
                imu_msg.angular_velocity.x - self.drift[0],
                imu_msg.angular_velocity.y - self.drift[1],
                imu_msg.angular_velocity.z - self.drift[2],
            ]
        )

        orientation_q = np.array(
            [
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w,
            ]
        )

        if self.imu_to_base_q is not None:
            # q_base^odom = q_imu^odom * q_base^imu = q_imu^odom * inv(q_imu^base)
            imu_to_base_q_inv = tf.transformations.quaternion_inverse(
                self.imu_to_base_q
            )
            orientation_q = self.quaternion_multiply(orientation_q, imu_to_base_q_inv)

            # For angular velocity: omega_base = R_imu^base * omega_imu
            # Get rotation matrix from quaternion
            rotation_matrix = tf.transformations.quaternion_matrix(self.imu_to_base_q)[
                :3, :3
            ]
            corrected_angular_velocity = np.dot(
                rotation_matrix, corrected_angular_velocity
            )
        self.odom_msg.twist.twist.angular.x = corrected_angular_velocity[0]
        self.odom_msg.twist.twist.angular.y = corrected_angular_velocity[1]
        self.odom_msg.twist.twist.angular.z = corrected_angular_velocity[2]

        self.odom_msg.pose.pose.orientation = Quaternion(
            x=orientation_q[0],
            y=orientation_q[1],
            z=orientation_q[2],
            w=orientation_q[3],
        )

        self.odom_msg.pose.covariance = self.insert_covariance_block(
            self.default_pose_cov, imu_msg.orientation_covariance
        )
        self.odom_msg.twist.covariance = self.insert_covariance_block(
            self.default_twist_cov, imu_msg.angular_velocity_covariance
        )

        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0
        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        self.odom_publisher.publish(self.odom_msg)

    def calibrate_imu(self, req):
        duration = req.duration
        rospy.loginfo(f"IMU Calibration started for {duration} seconds...")
        self.calibrating = True
        self.calibration_data = []
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
            return CalibrateIMUResponse(success=False, message="IMU Calibration failed")

    def save_calibration_data(self):
        calibration_data = {"drift": self.drift.tolist()}
        try:
            with open(self.imu_calibration_data_path, "w") as f:
                yaml.dump(calibration_data, f)
            rospy.loginfo("Calibration data saved.")
        except Exception as e:
            rospy.logerr(f"Failed to save calibration data: {e}")

    def load_calibration_data(self):
        try:
            with open(self.imu_calibration_data_path, "r") as f:
                calibration_data = yaml.safe_load(f)
                self.drift = np.array(calibration_data["drift"])
            rospy.loginfo(f"IMU Calibration data loaded. Drift: {self.drift}")
        except FileNotFoundError:
            rospy.logwarn("No calibration data found.")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        imu_to_odom = ImuToOdom()
        imu_to_odom.run()
    except rospy.ROSInterruptException:
        pass
