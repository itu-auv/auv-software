#!/usr/bin/env python

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from geometry_msgs.msg import Twist, WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import math
import message_filters
import time
from types import SimpleNamespace
import tf_transformations


class DvlToOdom(Node):
    def __init__(self):
        super().__init__("dvl_to_odom_node")

        self.enabled = self.declare_parameter("enabled", True).value
        self.enable_service = self.create_service(
            SetBool, "dvl_to_odom_node/enable", self.enable_cb
        )

        self.namespace = self.declare_parameter("namespace", "taluy").value
        self.base_frame = self.declare_parameter(
            "base_frame", f"{self.namespace}/base_link"
        ).value
        self.dvl_frame = self.declare_parameter(
            "dvl_frame", f"{self.namespace}/base_link/dvl_link"
        ).value

        self.tf_node = Node("dvl_tf_listener", use_global_arguments=False)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self.tf_node, spin_thread=True
        )
        self.dvl_yaw = self.get_dvl_yaw()

        self.cmdvel_tau = self.declare_parameter("cmdvel_tau", 0.1).value
        self.linear_x_covariance = self.declare_parameter(
            "sensors.dvl.covariance.linear_x", 0.000015
        ).value
        self.linear_y_covariance = self.declare_parameter(
            "sensors.dvl.covariance.linear_y", 0.000015
        ).value
        self.linear_z_covariance = self.declare_parameter(
            "sensors.dvl.covariance.linear_z", 0.00005
        ).value

        self.model_covariance_multiplier = self.declare_parameter(
            "model_covariance_multiplier", 10.0
        ).value

        self.load_dynamic_model()

        # Subscribers and Publishers
        self.dvl_velocity_subscriber = message_filters.Subscriber(
            self, Twist, "dvl/velocity_raw"
        )
        self.is_valid_subscriber = message_filters.Subscriber(
            self, Bool, "dvl/is_valid"
        )
        self.desired_velocity_subscriber = self.create_subscription(
            Twist, "desired_velocity", self.desired_velocity_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self.odom_callback, 10
        )

        self.wrench_sub = self.create_subscription(
            WrenchStamped, "wrench", self.wrench_callback, 10
        )

        self.velocity_arrival_filter = message_filters.SimpleFilter()
        self.valid_arrival_filter = message_filters.SimpleFilter()
        self.dvl_velocity_subscriber.registerCallback(
            lambda msg: self.velocity_arrival_filter.signalMessage(
                SimpleNamespace(
                    header=SimpleNamespace(stamp=self.get_clock().now().to_msg()),
                    message=msg,
                )
            )
        )
        self.is_valid_subscriber.registerCallback(
            lambda msg: self.valid_arrival_filter.signalMessage(
                SimpleNamespace(
                    header=SimpleNamespace(stamp=self.get_clock().now().to_msg()),
                    message=msg,
                )
            )
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.velocity_arrival_filter, self.valid_arrival_filter],
            queue_size=10,
            slop=0.1,
            allow_headerless=True,
        )
        self.sync.registerCallback(
            lambda velocity, valid: self.dvl_callback(velocity.message, valid.message)
        )

        self.odom_publisher = self.create_publisher(Odometry, "odom_dvl", 10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.namespace + "/base_link"

        # Initialize covariances with default values
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.update_twist_covariance(False)

        # Logging
        self.get_logger().info(
            "\033[38;5;110mDVL Odometry Calibration data loaded\033[0m:"
        )
        self.get_logger().info(f"\tcmdvel_tau: {self.cmdvel_tau}")
        self.get_logger().info(f"\tlinear x covariance: {self.linear_x_covariance}")
        self.get_logger().info(f"\tlinear y covariance: {self.linear_y_covariance}")
        self.get_logger().info(f"\tlinear z covariance: {self.linear_z_covariance}")
        self.get_logger().info(
            f"\tmodel covariance multiplier: {self.model_covariance_multiplier}"
        )

        # Fallback variables
        self.desired_velocity_twist = Twist()
        self.filtered_desired_velocity = Twist()
        self.last_update_time = self.get_clock().now()
        self.is_dvl_enabled = False

        # Model-based velocity estimator variables
        self.current_wrench = np.zeros(6)
        self.last_wrench_time = self.get_clock().now()
        self.estimated_velocity = np.zeros(6)
        self.odom_velocity = np.zeros(6)
        self.last_model_update = self.get_clock().now()
        self.odom_received = False

    def load_dynamic_model(self):
        try:
            # Mass-inertia matrix
            M_list = self.declare_parameter(
                "model.mass_inertia_matrix", rclpy.Parameter.Type.DOUBLE_ARRAY
            ).value
            self.M = np.array(M_list).reshape(6, 6)
            self.M_inv = np.linalg.inv(self.M)

            # Damping matrices
            D_linear_list = self.declare_parameter(
                "model.linear_damping_matrix", rclpy.Parameter.Type.DOUBLE_ARRAY
            ).value
            self.D_linear = np.array(D_linear_list).reshape(6, 6)

            D_quad_list = self.declare_parameter(
                "model.quadratic_damping_matrix", rclpy.Parameter.Type.DOUBLE_ARRAY
            ).value
            self.D_quadratic = np.array(D_quad_list).reshape(6, 6)
            if not all(
                np.isfinite(matrix).all()
                for matrix in (self.M, self.D_linear, self.D_quadratic)
            ):
                raise ValueError("dynamic model matrices must be finite")

            self.dynamic_model_available = True

            self.get_logger().info(
                "\033[38;5;157mDynamic model loaded successfully\033[0m"
            )
            self.get_logger().info(f"Mass matrix diagonal: {np.diag(self.M)}")
            self.get_logger().info(f"Linear damping diagonal: {np.diag(self.D_linear)}")
            self.get_logger().info(
                f"Quadratic damping diagonal: {np.diag(self.D_quadratic)}"
            )

        except Exception as e:
            self.get_logger().error(
                f"\033[38;5;203mFailed to load dynamic model: {e}\033[0m"
            )
            self.get_logger().warning("Using identity/zero matrices as fallback")
            # Fallback to safe defaults
            self.M = np.eye(6)
            self.M_inv = np.eye(6)
            self.D_linear = np.zeros((6, 6))
            self.D_quadratic = np.zeros((6, 6))
            self.dynamic_model_available = False

    def update_twist_covariance(self, use_model_based):
        multiplier = self.model_covariance_multiplier if use_model_based else 1.0

        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance[0] = self.linear_x_covariance * multiplier
        self.odom_msg.twist.covariance[7] = self.linear_y_covariance * multiplier
        self.odom_msg.twist.covariance[14] = self.linear_z_covariance * multiplier

    def enable_cb(self, req, response):
        """Service callback to enable/disable DVL->Odom processing"""
        self.enabled = req.data
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(f"DVL->Odom node {state} via service call.")
        response.success = True
        response.message = f"DVL->Odom {state}"
        return response

    def get_dvl_yaw(self):
        self.get_logger().info(f"Waiting for TF: {self.base_frame} <- {self.dvl_frame}")
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.dvl_frame, Time(), timeout=Duration(seconds=10.0)
            )
            q = transform.transform.rotation
            rot = (q.x, q.y, q.z, q.w)
            euler = tf_transformations.euler_from_quaternion(rot)
            yaw = euler[2]
            self.get_logger().info(
                f"Loaded {self.dvl_frame} yaw from TF: {yaw} radians"
            )
            return yaw
        except Exception as e:
            self.get_logger().warning(
                f"Could not get TF for {self.dvl_frame} relative to {self.base_frame}: {e}. Falling back to default -45 degrees."
            )
            return np.radians(-45)

    def transform_vector(self, vector):
        theta = self.dvl_yaw
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(rotation_matrix, np.array(vector))

    def desired_velocity_callback(self, desired_velocity_msg):
        self.desired_velocity_twist = desired_velocity_msg

    def odom_callback(self, odom_msg):
        self.odom_velocity[0] = odom_msg.twist.twist.linear.x
        self.odom_velocity[1] = odom_msg.twist.twist.linear.y
        self.odom_velocity[2] = odom_msg.twist.twist.linear.z
        self.odom_velocity[3] = odom_msg.twist.twist.angular.x
        self.odom_velocity[4] = odom_msg.twist.twist.angular.y
        self.odom_velocity[5] = odom_msg.twist.twist.angular.z
        self.odom_received = True

    def wrench_callback(self, wrench_msg):
        self.last_wrench_time = self.get_clock().now()
        self.current_wrench[0] = wrench_msg.wrench.force.x
        self.current_wrench[1] = wrench_msg.wrench.force.y
        self.current_wrench[2] = wrench_msg.wrench.force.z
        self.current_wrench[3] = wrench_msg.wrench.torque.x
        self.current_wrench[4] = wrench_msg.wrench.torque.y
        self.current_wrench[5] = wrench_msg.wrench.torque.z

    def filter_desired_velocity(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.alpha = dt / (self.cmdvel_tau + dt)

        self.filtered_desired_velocity.linear.x = (
            self.filtered_desired_velocity.linear.x * (1.0 - self.alpha)
            + self.alpha * self.desired_velocity_twist.linear.x
        )
        self.filtered_desired_velocity.linear.y = (
            self.filtered_desired_velocity.linear.y * (1.0 - self.alpha)
            + self.alpha * self.desired_velocity_twist.linear.y
        )
        self.filtered_desired_velocity.linear.z = (
            self.filtered_desired_velocity.linear.z * (1.0 - self.alpha)
            + self.alpha * self.desired_velocity_twist.linear.z
        )
        self.filtered_desired_velocity.angular.x = (
            self.filtered_desired_velocity.angular.x * (1.0 - self.alpha)
            + self.alpha * self.desired_velocity_twist.angular.x
        )
        self.filtered_desired_velocity.angular.y = (
            self.filtered_desired_velocity.angular.y * (1.0 - self.alpha)
            + self.alpha * self.desired_velocity_twist.angular.y
        )
        self.filtered_desired_velocity.angular.z = (
            self.filtered_desired_velocity.angular.z * (1.0 - self.alpha)
            + self.alpha * self.desired_velocity_twist.angular.z
        )

        self.last_update_time = current_time

    def skew(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def calculate_coriolis_matrix(self, v):
        C = np.zeros((6, 6))

        M11 = self.M[0:3, 0:3]
        M12 = self.M[0:3, 3:6]
        M21 = self.M[3:6, 0:3]
        M22 = self.M[3:6, 3:6]

        v1 = v[0:3]
        v2 = v[3:6]

        Mv1 = np.dot(M11, v1) + np.dot(M12, v2)
        Mv2 = np.dot(M21, v1) + np.dot(M22, v2)

        s_mv1 = self.skew(Mv1)
        s_mv2 = self.skew(Mv2)

        C[0:3, 3:6] = -s_mv1
        C[3:6, 0:3] = -s_mv1
        C[3:6, 3:6] = -s_mv2

        return C

    def compute_model_based_velocity(self, dt):
        if self.odom_received:
            v = self.odom_velocity.copy()
        else:
            v = self.estimated_velocity.copy()

        # Check for stale wrench (timeout > 0.5s)
        if (self.get_clock().now() - self.last_wrench_time).nanoseconds / 1e9 > 0.5:
            self.current_wrench = np.zeros(6)

        # Linear damping
        damping_linear = np.dot(self.D_linear, v)

        # Quadratic damping
        v_abs = np.abs(v)
        damping_quad = np.dot(self.D_quadratic, v_abs * v)

        # Coriolis effects
        C_matrix = self.calculate_coriolis_matrix(v)
        coriolis_force = np.dot(C_matrix, v)

        # Net force
        net_force = self.current_wrench - damping_linear - damping_quad - coriolis_force

        # Acceleration
        acceleration = np.dot(self.M_inv, net_force)

        # Velocity update
        self.estimated_velocity = v + acceleration * dt

        return self.estimated_velocity

    def dvl_callback(self, velocity_msg, is_valid_msg):
        self.is_dvl_enabled = True

        if not self.enabled:
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_model_update).nanoseconds / 1e9
        self.last_model_update = current_time

        if is_valid_msg.data:
            # rospy.loginfo_throttle(1.0, "Using valid DVL measurements for velocity")
            rotated_vector = self.transform_vector(
                [velocity_msg.linear.x, velocity_msg.linear.y, velocity_msg.linear.z]
            )
            velocity_msg.linear.x = rotated_vector[0]
            velocity_msg.linear.y = rotated_vector[1]
            velocity_msg.linear.z = rotated_vector[2]

            # Seed the model with the valid DVL velocity so it starts right where DVL left off
            self.estimated_velocity[0] = velocity_msg.linear.x
            self.estimated_velocity[1] = velocity_msg.linear.y
            self.estimated_velocity[2] = velocity_msg.linear.z
            self.estimated_velocity[3] = velocity_msg.angular.x
            self.estimated_velocity[4] = velocity_msg.angular.y
            self.estimated_velocity[5] = velocity_msg.angular.z

            self.update_twist_covariance(use_model_based=False)

        else:
            # DVL invalid - use model-based estimation or desired_velocity fallback
            self.filter_desired_velocity()

            if self.dynamic_model_available and dt > 0.001 and dt < 1.0:
                self.get_logger().info(
                    "DVL invalid: Using dynamic model for velocity estimation",
                    throttle_duration_sec=1.0,
                )
                model_vel = self.compute_model_based_velocity(dt)

                velocity_msg.linear.x = model_vel[0]
                velocity_msg.linear.y = model_vel[1]
                velocity_msg.linear.z = model_vel[2]
                velocity_msg.angular.x = model_vel[3]
                velocity_msg.angular.y = model_vel[4]
                velocity_msg.angular.z = model_vel[5]

                self.update_twist_covariance(use_model_based=True)

            else:
                self.get_logger().warning(
                    "DVL invalid and model unavailable/timeout: Using desired_velocity fallback",
                    throttle_duration_sec=1.0,
                )
                velocity_msg.linear.x = self.filtered_desired_velocity.linear.x
                velocity_msg.linear.y = self.filtered_desired_velocity.linear.y
                velocity_msg.linear.z = self.filtered_desired_velocity.linear.z
                velocity_msg.angular.x = self.filtered_desired_velocity.angular.x
                velocity_msg.angular.y = self.filtered_desired_velocity.angular.y
                velocity_msg.angular.z = self.filtered_desired_velocity.angular.z

                self.update_twist_covariance(use_model_based=True)

        self.odom_msg.header.stamp = current_time.to_msg()
        self.odom_msg.twist.twist = velocity_msg

        # Set position to zero as we are not computing it here
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        def publish_initial():
            if not self.is_dvl_enabled:
                self.odom_msg.header.stamp = self.get_clock().now().to_msg()
                self.odom_msg.twist.twist = Twist()
                self.odom_publisher.publish(self.odom_msg)

        self.timer = self.create_timer(0.05, publish_initial)
        rclpy.spin(self)


def main():
    rclpy.init()
    dvl_to_odom = DvlToOdom()
    try:
        dvl_to_odom.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        dvl_to_odom.tf_listener.unregister()
        dvl_to_odom.tf_node.destroy_node()
        dvl_to_odom.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
