#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import math
import message_filters
import time
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class DvlToOdom:
    def __init__(self):
        rospy.init_node("dvl_to_odom_node", anonymous=True)

        self.enabled = rospy.get_param("~enabled", True)
        self.enable_service = rospy.Service(
            "dvl_to_odom_node/enable", SetBool, self.enable_cb
        )

        self.cmdvel_tau = rospy.get_param("~cmdvel_tau", 0.1)
        self.linear_x_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_x", 0.000015
        )
        self.linear_y_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_y", 0.000015
        )
        self.linear_z_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_z", 0.00005
        )

        self.model_covariance_multiplier = rospy.get_param(
            "~model_covariance_multiplier", 10.0
        )

        self.load_dynamic_model()

        # Subscribers and Publishers
        self.dvl_velocity_subscriber = message_filters.Subscriber(
            "dvl/velocity_raw", Twist, tcp_nodelay=True
        )
        self.is_valid_subscriber = message_filters.Subscriber(
            "dvl/is_valid", Bool, tcp_nodelay=True
        )
        self.cmd_vel_subscriber = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
        )

        self.odom_sub = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, tcp_nodelay=True
        )

        self.wrench_sub = rospy.Subscriber(
            "wrench", WrenchStamped, self.wrench_callback, tcp_nodelay=True
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.dvl_velocity_subscriber, self.is_valid_subscriber],
            queue_size=10,
            slop=0.1,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.dvl_callback)

        self.odom_publisher = rospy.Publisher("odom_dvl", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        # Initialize covariances with default values
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.update_twist_covariance(False)

        # Logging
        DVL_odometry_colored = TerminalColors.color_text(
            "DVL Odometry Calibration data loaded", TerminalColors.PASTEL_BLUE
        )
        rospy.loginfo(f"{DVL_odometry_colored} : cmdvel_tau: {self.cmdvel_tau}")
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear x covariance: {self.linear_x_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear y covariance: {self.linear_y_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear z covariance: {self.linear_z_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : model covariance multiplier: {self.model_covariance_multiplier}"
        )

        # Fallback variables
        self.cmd_vel_twist = Twist()
        self.filtered_cmd_vel = Twist()
        self.last_update_time = rospy.Time.now()
        self.is_dvl_enabled = False

        # Model-based velocity estimator variables
        self.current_wrench = np.zeros(6)
        self.estimated_velocity = np.zeros(6)
        self.odom_velocity = np.zeros(6)
        self.last_model_update = rospy.Time.now()
        self.odom_received = False

    def load_dynamic_model(self):
        try:
            # Mass-inertia matrix
            M_list = rospy.get_param("model/mass_inertia_matrix")
            self.M = np.array(M_list)
            self.M_inv = np.linalg.inv(self.M)

            # Damping matrices
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

    def enable_cb(self, req):
        """Service callback to enable/disable DVL->Odom processing"""
        self.enabled = req.data
        state = "enabled" if self.enabled else "disabled"
        rospy.loginfo(f"DVL->Odom node {state} via service call.")
        return SetBoolResponse(success=True, message=f"DVL->Odom {state}")

    def transform_vector(self, vector):
        theta = np.radians(-135)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(rotation_matrix, np.array(vector))

    def cmd_vel_callback(self, cmd_vel_msg):
        self.cmd_vel_twist = cmd_vel_msg

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

    def filter_cmd_vel(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        self.alpha = dt / (self.cmdvel_tau + dt)

        self.filtered_cmd_vel.linear.x = (
            self.filtered_cmd_vel.linear.x * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.linear.x
        )
        self.filtered_cmd_vel.linear.y = (
            self.filtered_cmd_vel.linear.y * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.linear.y
        )
        self.filtered_cmd_vel.linear.z = (
            self.filtered_cmd_vel.linear.z * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.linear.z
        )
        self.filtered_cmd_vel.angular.x = (
            self.filtered_cmd_vel.angular.x * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.angular.x
        )
        self.filtered_cmd_vel.angular.y = (
            self.filtered_cmd_vel.angular.y * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.angular.y
        )
        self.filtered_cmd_vel.angular.z = (
            self.filtered_cmd_vel.angular.z * (1.0 - self.alpha)
            + self.alpha * self.cmd_vel_twist.angular.z
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

        current_time = rospy.Time.now()
        dt = (current_time - self.last_model_update).to_sec()
        self.last_model_update = current_time

        if is_valid_msg.data:
            rotated_vector = self.transform_vector(
                [velocity_msg.linear.x, velocity_msg.linear.y, velocity_msg.linear.z]
            )
            velocity_msg.linear.x = rotated_vector[0]
            velocity_msg.linear.y = rotated_vector[1]
            velocity_msg.linear.z = rotated_vector[2]

            self.update_twist_covariance(use_model_based=False)

        else:
            # DVL invalid - use model-based estimation or cmd_vel fallback
            self.filter_cmd_vel()

            if self.dynamic_model_available and dt > 0.001 and dt < 1.0:
                model_vel = self.compute_model_based_velocity(dt)

                velocity_msg.linear.x = model_vel[0]
                velocity_msg.linear.y = model_vel[1]
                velocity_msg.linear.z = model_vel[2]
                velocity_msg.angular.x = model_vel[3]
                velocity_msg.angular.y = model_vel[4]
                velocity_msg.angular.z = model_vel[5]

                self.update_twist_covariance(use_model_based=True)

            else:
                velocity_msg.linear.x = self.filtered_cmd_vel.linear.x
                velocity_msg.linear.y = self.filtered_cmd_vel.linear.y
                velocity_msg.linear.z = self.filtered_cmd_vel.linear.z
                velocity_msg.angular.x = self.filtered_cmd_vel.angular.x
                velocity_msg.angular.y = self.filtered_cmd_vel.angular.y
                velocity_msg.angular.z = self.filtered_cmd_vel.angular.z

                self.update_twist_covariance(use_model_based=True)

        self.odom_msg.header.stamp = current_time
        self.odom_msg.twist.twist = velocity_msg

        # Set position to zero as we are not computing it here
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.is_dvl_enabled:
                self.odom_msg.header.stamp = rospy.Time.now()
                self.odom_msg.twist.twist = Twist()
                self.odom_publisher.publish(self.odom_msg)
            rate.sleep()


if __name__ == "__main__":
    try:
        dvl_to_odom = DvlToOdom()
        dvl_to_odom.run()
    except rospy.ROSInterruptException:
        pass
