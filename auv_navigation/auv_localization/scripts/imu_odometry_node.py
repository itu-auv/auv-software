#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
import tf_transformations
import numpy as np
import yaml
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
from auv_msgs.srv import CalibrateIMU

HIGH_COVARIANCE = 1e6


class ImuToOdom(Node):
    def __init__(self):
        super().__init__("imu_to_odom_node")
        self.callback_group = ReentrantCallbackGroup()

        self.namespace = self.declare_parameter("namespace", "taluy").value
        self.imu_calibration_data_path = self.declare_parameter(
            "imu_calibration_path", "config/imu_calibration_data.yaml"
        ).value

        self.base_frame = self.declare_parameter("base_frame", f"{self.namespace}/base_link").value
        self.imu_frame = self.declare_parameter(
            "imu_frame", f"{self.namespace}/base_link/imu"
        ).value

        self.tf_node = Node("imu_tf_listener", use_global_arguments=False)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)

        self.imu_to_base_q = self.get_frame_rotation(self.imu_frame)

        # Subscribers and Publishers
        self.imu_subscriber = self.create_subscription(
            Imu, "imu/data", self.imu_callback, 10, callback_group=self.callback_group
        )
        self.odom_publisher = self.create_publisher(Odometry, "odom_imu", 10)

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
        self.calibration_service = self.create_service(
            CalibrateIMU, "calibrate_imu", self.calibrate_imu, callback_group=self.callback_group
        )

    def get_frame_rotation(self, frame_id):
        self.get_logger().info(f"Waiting for TF: {self.base_frame} <- {frame_id}")
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, frame_id, Time(), timeout=Duration(seconds=10.0)
            )
            q = transform.transform.rotation
            rot = (q.x, q.y, q.z, q.w)
            self.get_logger().info(f"Loaded {frame_id} rotation from TF: {rot}")
            return np.array(rot)
        except Exception as e:
            self.get_logger().warning(
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
            imu_to_base_q_inv = tf_transformations.quaternion_inverse(
                self.imu_to_base_q
            )
            orientation_q = self.quaternion_multiply(orientation_q, imu_to_base_q_inv)

            # For angular velocity: omega_base = R_imu^base * omega_imu
            # Get rotation matrix from quaternion
            rotation_matrix = tf_transformations.quaternion_matrix(self.imu_to_base_q)[
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

    def calibrate_imu(self, req, response):
        duration = req.duration
        self.get_logger().info(f"IMU Calibration started for {duration} seconds...")
        self.calibrating = True
        self.calibration_data = []
        self.get_clock().sleep_for(Duration(seconds=duration))
        self.calibrating = False

        if len(self.calibration_data) > 0:
            self.drift = np.mean(self.calibration_data, axis=0)
            self.save_calibration_data()
            self.get_logger().info(f"IMU Calibration completed. Drift: {self.drift}")
            response.success = True
            response.message = "IMU Calibration successful"
            return response
        else:
            response.success = False
            response.message = "IMU Calibration failed"
            return response

    def save_calibration_data(self):
        calibration_data = {"drift": self.drift.tolist()}
        try:
            with open(self.imu_calibration_data_path, "w") as f:
                yaml.dump(calibration_data, f)
            self.get_logger().info("Calibration data saved.")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration data: {e}")

    def load_calibration_data(self):
        try:
            with open(self.imu_calibration_data_path, "r") as f:
                calibration_data = yaml.safe_load(f)
                self.drift = np.array(calibration_data["drift"])
            self.get_logger().info(f"IMU Calibration data loaded. Drift: {self.drift}")
        except FileNotFoundError:
            self.get_logger().warning("No calibration data found.")

    def run(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        try:
            executor.spin()
        finally:
            executor.shutdown()


def main():
    rclpy.init()
    imu_to_odom = ImuToOdom()
    try:
        imu_to_odom.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        imu_to_odom.tf_listener.unregister()
        imu_to_odom.tf_node.destroy_node()
        imu_to_odom.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
