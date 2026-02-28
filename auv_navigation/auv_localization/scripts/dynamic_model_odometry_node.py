#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
import numpy as np
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class DynamicModelOdometry:
    def __init__(self):
        rospy.init_node("dynamic_model_odometry_node", anonymous=True)

        self.rate_hz = rospy.get_param("~rate", 20)

        # Covariance parameters
        self.linear_x_covariance = rospy.get_param(
            "dynamic_model/covariance/linear_x", 0.01
        )
        self.linear_y_covariance = rospy.get_param(
            "dynamic_model/covariance/linear_y", 0.01
        )
        self.linear_z_covariance = rospy.get_param(
            "dynamic_model/covariance/linear_z", 0.01
        )
        self.angular_x_covariance = rospy.get_param(
            "dynamic_model/covariance/angular_x", 0.01
        )
        self.angular_y_covariance = rospy.get_param(
            "dynamic_model/covariance/angular_y", 0.01
        )
        self.angular_z_covariance = rospy.get_param(
            "dynamic_model/covariance/angular_z", 0.01
        )

        # Load dynamic model parameters
        self.load_dynamic_model()

        # Subscribers
        self.odom_sub = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, tcp_nodelay=True
        )
        self.wrench_sub = rospy.Subscriber(
            "wrench", WrenchStamped, self.wrench_callback, tcp_nodelay=True
        )

        # Publisher
        self.odom_publisher = rospy.Publisher(
            "odom_dynamic_model", Odometry, queue_size=10
        )

        # Initialize odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.update_twist_covariance()

        # State variables
        self.current_wrench = np.zeros(6)
        self.estimated_velocity = np.zeros(6)
        self.odom_velocity = np.zeros(6)
        self.last_model_update = rospy.Time.now()
        self.odom_received = False

        # Logging
        node_colored = TerminalColors.color_text(
            "Dynamic Model Odometry", TerminalColors.PASTEL_GREEN
        )
        rospy.loginfo(f"{node_colored} : Node initialized at {self.rate_hz} Hz")
        rospy.loginfo(
            f"{node_colored} : Dynamic model available: {self.dynamic_model_available}"
        )

    def load_dynamic_model(self):
        try:
            M_list = rospy.get_param("model/mass_inertia_matrix")
            self.M = np.array(M_list)
            self.M_inv = np.linalg.inv(self.M)

            D_linear_list = rospy.get_param("model/linear_damping_matrix")
            self.D_linear = np.array(D_linear_list)

            D_quad_list = rospy.get_param("model/quadratic_damping_matrix")
            self.D_quadratic = np.array(D_quad_list)

            self.dynamic_model_available = True

            model_loaded_colored = TerminalColors.color_text(
                "Dynamic model loaded successfully", TerminalColors.PASTEL_GREEN
            )
            rospy.loginfo(model_loaded_colored)
            rospy.loginfo(f"Mass matrix diagonal: {np.diag(self.M)}")
            rospy.loginfo(f"Linear damping diagonal: {np.diag(self.D_linear)}")
            rospy.loginfo(f"Quadratic damping diagonal: {np.diag(self.D_quadratic)}")

        except Exception as e:
            model_error_colored = TerminalColors.color_text(
                f"Failed to load dynamic model: {e}", TerminalColors.PASTEL_RED
            )
            rospy.logerr(model_error_colored)
            rospy.logwarn("Using identity/zero matrices as fallback")
            self.M = np.eye(6)
            self.M_inv = np.eye(6)
            self.D_linear = np.zeros((6, 6))
            self.D_quadratic = np.zeros((6, 6))
            self.dynamic_model_available = False

    def update_twist_covariance(self):
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance[0] = self.linear_x_covariance
        self.odom_msg.twist.covariance[7] = self.linear_y_covariance
        self.odom_msg.twist.covariance[14] = self.linear_z_covariance
        self.odom_msg.twist.covariance[21] = self.angular_x_covariance
        self.odom_msg.twist.covariance[28] = self.angular_y_covariance
        self.odom_msg.twist.covariance[35] = self.angular_z_covariance

    def odom_callback(self, odom_msg):
        self.odom_velocity[0] = odom_msg.twist.twist.linear.x
        self.odom_velocity[1] = odom_msg.twist.twist.linear.y
        self.odom_velocity[2] = odom_msg.twist.twist.linear.z
        self.odom_velocity[3] = odom_msg.twist.twist.angular.x
        self.odom_velocity[4] = odom_msg.twist.twist.angular.y
        self.odom_velocity[5] = odom_msg.twist.twist.angular.z
        self.odom_received = True

    def wrench_callback(self, wrench_msg):
        self.current_wrench[0] = wrench_msg.wrench.force.x
        self.current_wrench[1] = wrench_msg.wrench.force.y
        self.current_wrench[2] = wrench_msg.wrench.force.z
        self.current_wrench[3] = wrench_msg.wrench.torque.x
        self.current_wrench[4] = wrench_msg.wrench.torque.y
        self.current_wrench[5] = wrench_msg.wrench.torque.z

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

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            dt = (current_time - self.last_model_update).to_sec()
            self.last_model_update = current_time

            if dt > 0.001 and dt < 1.0:
                model_vel = self.compute_model_based_velocity(dt)

                self.odom_msg.header.stamp = current_time
                self.odom_msg.twist.twist.linear.x = model_vel[0]
                self.odom_msg.twist.twist.linear.y = model_vel[1]
                self.odom_msg.twist.twist.linear.z = model_vel[2]
                self.odom_msg.twist.twist.angular.x = model_vel[3]
                self.odom_msg.twist.twist.angular.y = model_vel[4]
                self.odom_msg.twist.twist.angular.z = model_vel[5]

                self.odom_msg.pose.pose.position.x = 0.0
                self.odom_msg.pose.pose.position.y = 0.0
                self.odom_msg.pose.pose.position.z = 0.0

                self.odom_publisher.publish(self.odom_msg)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = DynamicModelOdometry()
        node.run()
    except rospy.ROSInterruptException:
        pass
