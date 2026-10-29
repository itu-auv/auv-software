#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import numpy as np
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import tf_transformations
import tf2_ros


class PressureToOdom(Node):
    def __init__(self):
        super().__init__("pressure_to_odom_node")

        self.odom_orientation = None
        self.depth_data = None
        self.base_to_sensor_translation = None

        # params
        self.namespace = self.declare_parameter("namespace", "taluy").value
        self.base_frame = self.namespace + "/base_link"
        self.pressure_sensor_frame = (
            self.namespace + "/base_link/external_pressure_sensor_link"
        )

        self.depth_calibration_offset = self.declare_parameter(
            "sensors.external_pressure_sensor.depth_offset", -0.10
        ).value
        self.depth_calibration_covariance = self.declare_parameter(
            "sensors.external_pressure_sensor.depth_covariance", 0.05
        ).value
        self.pool_depth = self.declare_parameter("env.pool_depth", 2.2).value
        self.min_valid_depth = self.declare_parameter("min_valid_depth", 0.1).value
        self.max_valid_depth = self.declare_parameter("max_valid_depth", -self.pool_depth).value

        # tf transformer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_node = Node("pressure_tf_listener", use_global_arguments=False)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)

        # publishers and subscribers
        self.odom_publisher = self.create_publisher(Odometry, "odom_pressure", 10)
        self.odom_orientation_subscriber = self.create_subscription(
            Odometry, "odometry", self.odom_callback, 10
        )
        self.depth_subscriber = self.create_subscription(
            Float32, "depth", self.depth_callback, 10
        )

        # initialize odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.namespace + "/base_link"
        self.odom_msg.pose.covariance[14] = self.depth_calibration_covariance

        self.get_logger().info("[PressureToOdom] Node initialized with parameters:")
        self.get_logger().info(f"  depth_calibration_offset: {self.depth_calibration_offset}")
        self.get_logger().info(
            f"  depth_calibration_covariance: {self.depth_calibration_covariance}"
        )
        self.get_logger().info(f"  pool_depth: {self.pool_depth}")
        self.get_logger().info(f"  min_valid_depth: {self.min_valid_depth}")
        self.get_logger().info(f"  max_valid_depth: {self.max_valid_depth}")

    def odom_callback(self, odom_msg):
        self.odom_orientation = odom_msg.pose.pose.orientation

    def get_base_to_pressure_height(self):
        if self.base_to_sensor_translation is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.pressure_sensor_frame,
                    self.base_frame,
                    Time(),
                    Duration(seconds=4.0),
                )
                self.base_to_sensor_translation = np.array(
                    [
                        trans.transform.translation.x,
                        trans.transform.translation.y,
                        trans.transform.translation.z,
                    ]
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self.get_logger().warning(
                    f"{self.pressure_sensor_frame} TF not available: {e}; using zero offset."
                )
                return 0.0

        translation = self.base_to_sensor_translation

        if self.odom_orientation is None:
            return float(translation[2])

        # compute rotated z-offset based on current orientation
        orientation = self.odom_orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf_transformations.quaternion_matrix(quat)[:3, :3]
        rotated_vector = rotation_matrix.dot(translation)
        return float(rotated_vector[2])

    def depth_callback(self, msg):
        self.depth_data = msg.data
        self.process_pressure_depth()

    def process_pressure_depth(self):
        if self.depth_data is None:
            self.get_logger().warning("Waiting for depth data...", throttle_duration_sec=5.0)
            return

        pressure_depth_calibrated = self.depth_data + self.depth_calibration_offset
        pressure_depth = pressure_depth_calibrated + self.get_base_to_pressure_height()

        clamped_pressure_depth = float(
            np.clip(pressure_depth, self.max_valid_depth, self.min_valid_depth)
        )
        is_pressure_valid = abs(pressure_depth - clamped_pressure_depth) <= 1e-6

        if not is_pressure_valid:
            self.get_logger().warning(
                f"Pressure depth out of valid range [{self.max_valid_depth:.3f}, {self.min_valid_depth:.3f}]: {pressure_depth:.3f}; publishing clamped value {clamped_pressure_depth:.3f}", throttle_duration_sec=3.0
            )

        self.publish_odometry(clamped_pressure_depth)

    def publish_odometry(self, depth):
        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_msg.pose.pose.position.z = depth

        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    pressure_to_odom = PressureToOdom()
    try:
        pressure_to_odom.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        pressure_to_odom.tf_listener.unregister()
        pressure_to_odom.tf_node.destroy_node()
        pressure_to_odom.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
