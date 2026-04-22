#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, WrenchStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import numpy as np
from std_srvs.srv import SetBool, SetBoolResponse
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class ModelOdometryNode:
    def __init__(self):
        rospy.init_node("model_odometry_node", anonymous=True)

        self.enabled = rospy.get_param("~enabled", True)
        self.enable_service = rospy.Service(
            "model_odometry_node/enable", SetBool, self.enable_cb
        )

        self.namespace = rospy.get_param("~namespace", "taluy")

        self.publish_rate = rospy.get_param("~rate", 20.0)
        self.dvl_timeout = rospy.get_param("~dvl_timeout", 0.5)
        self.cmdvel_tau = rospy.get_param("~cmdvel_tau", 0.1)

        self.load_dynamic_model()

        # Subscribers
        self.is_valid_subscriber = rospy.Subscriber(
            "dvl/is_valid", Bool, self.dvl_valid_cb, tcp_nodelay=True
        )
        self.desired_velocity_subscriber = rospy.Subscriber(
            "desired_velocity", Twist, self.desired_velocity_callback, tcp_nodelay=True
        )
        self.odom_sub = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, tcp_nodelay=True
        )
        self.wrench_sub = rospy.Subscriber(
            "wrench", WrenchStamped, self.wrench_callback, tcp_nodelay=True
        )

        self.odom_publisher = rospy.Publisher("odom_model", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.namespace + "/base_link"

        self.odom_msg.pose.covariance = np.zeros(36).tolist()

        # Covariance initialization
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

        self.update_twist_covariance()

        # Fallback variables
        self.desired_vel_twist = Twist()
        self.filtered_desired_vel = Twist()
        self.last_desired_vel_update_time = rospy.Time.now()

        # Model-based velocity estimator variables
        self.current_wrench = np.zeros(6)
        self.last_wrench_time = rospy.Time.now()
        self.estimated_velocity = np.zeros(6)
        self.odom_velocity = np.zeros(6)
        self.last_model_update = rospy.Time.now()
        self.odom_received = False

        self.last_dvl_valid_time = rospy.Time.now()
        self.is_dvl_valid = False

        self.model_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self.model_timer_callback
        )

    def update_twist_covariance(self):
        multiplier = self.model_covariance_multiplier
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance[0] = self.linear_x_covariance * multiplier
        self.odom_msg.twist.covariance[7] = self.linear_y_covariance * multiplier
        self.odom_msg.twist.covariance[14] = self.linear_z_covariance * multiplier

    def enable_cb(self, req):
        self.enabled = req.data
        state = "enabled" if self.enabled else "disabled"
        rospy.loginfo(f"Model Odom node {state} via service call.")
        return SetBoolResponse(success=True, message=f"Model Odom {state}")

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
            rospy.loginfo(
                TerminalColors.color_text(
                    "Dynamic model loaded successfully", TerminalColors.PASTEL_GREEN
                )
            )
        except Exception as e:
            rospy.logerr(
                TerminalColors.color_text(
                    f"Failed to load dynamic model: {e}", TerminalColors.PASTEL_RED
                )
            )
            self.M = np.eye(6)
            self.M_inv = np.eye(6)
            self.D_linear = np.zeros((6, 6))
            self.D_quadratic = np.zeros((6, 6))
            self.dynamic_model_available = False

    def dvl_valid_cb(self, msg):
        self.is_dvl_valid = msg.data
        if self.is_dvl_valid:
            self.last_dvl_valid_time = rospy.Time.now()
            if self.odom_received:
                self.estimated_velocity = self.odom_velocity.copy()

    def desired_velocity_callback(self, msg):
        self.desired_vel_twist = msg

    def odom_callback(self, msg):
        self.odom_velocity[0] = msg.twist.twist.linear.x
        self.odom_velocity[1] = msg.twist.twist.linear.y
        self.odom_velocity[2] = msg.twist.twist.linear.z
        self.odom_velocity[3] = msg.twist.twist.angular.x
        self.odom_velocity[4] = msg.twist.twist.angular.y
        self.odom_velocity[5] = msg.twist.twist.angular.z
        self.odom_received = True

    def wrench_callback(self, msg):
        self.last_wrench_time = rospy.Time.now()
        self.current_wrench[0] = msg.wrench.force.x
        self.current_wrench[1] = msg.wrench.force.y
        self.current_wrench[2] = msg.wrench.force.z
        self.current_wrench[3] = msg.wrench.torque.x
        self.current_wrench[4] = msg.wrench.torque.y
        self.current_wrench[5] = msg.wrench.torque.z

    def filter_desired_velocity(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_desired_vel_update_time).to_sec()
        self.alpha = dt / (self.cmdvel_tau + dt)

        self.filtered_desired_vel.linear.x = (
            self.filtered_desired_vel.linear.x * (1.0 - self.alpha)
            + self.alpha * self.desired_vel_twist.linear.x
        )
        self.filtered_desired_vel.linear.y = (
            self.filtered_desired_vel.linear.y * (1.0 - self.alpha)
            + self.alpha * self.desired_vel_twist.linear.y
        )
        self.filtered_desired_vel.linear.z = (
            self.filtered_desired_vel.linear.z * (1.0 - self.alpha)
            + self.alpha * self.desired_vel_twist.linear.z
        )
        self.filtered_desired_vel.angular.x = (
            self.filtered_desired_vel.angular.x * (1.0 - self.alpha)
            + self.alpha * self.desired_vel_twist.angular.x
        )
        self.filtered_desired_vel.angular.y = (
            self.filtered_desired_vel.angular.y * (1.0 - self.alpha)
            + self.alpha * self.desired_vel_twist.angular.y
        )
        self.filtered_desired_vel.angular.z = (
            self.filtered_desired_vel.angular.z * (1.0 - self.alpha)
            + self.alpha * self.desired_vel_twist.angular.z
        )

        self.last_desired_vel_update_time = current_time

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
        v = self.estimated_velocity.copy()

        if (rospy.Time.now() - self.last_wrench_time).to_sec() > 0.5:
            self.current_wrench = np.zeros(6)

        damping_linear = np.dot(self.D_linear, v)
        v_abs = np.abs(v)
        damping_quad = np.dot(self.D_quadratic, v_abs * v)
        C_matrix = self.calculate_coriolis_matrix(v)
        coriolis_force = np.dot(C_matrix, v)

        net_force = self.current_wrench - damping_linear - damping_quad - coriolis_force
        acceleration = np.dot(self.M_inv, net_force)

        self.estimated_velocity = v + acceleration * dt
        return self.estimated_velocity

    def model_timer_callback(self, event):
        if not self.enabled:
            return

        current_time = rospy.Time.now()
        dt = (current_time - self.last_model_update).to_sec()
        self.last_model_update = current_time

        if dt <= 0.001 or dt >= 1.0:
            return

        if not self.is_dvl_valid:
            self.filter_desired_velocity()
            velocity_msg = Twist()

            if self.dynamic_model_available:
                rospy.loginfo_throttle(
                    5.0, "DVL is invalid: Publishing dynamic model velocity"
                )
                model_vel = self.compute_model_based_velocity(dt)
                velocity_msg.linear.x = model_vel[0]
                velocity_msg.linear.y = model_vel[1]
                velocity_msg.linear.z = model_vel[2]
                velocity_msg.angular.x = model_vel[3]
                velocity_msg.angular.y = model_vel[4]
                velocity_msg.angular.z = model_vel[5]
            else:
                rospy.logwarn_throttle(
                    5.0,
                    "DVL is invalid and model unavailable: Publishing desired velocity fallback",
                )
                velocity_msg = self.filtered_desired_vel

            self.odom_msg.header.stamp = current_time
            self.odom_msg.twist.twist = velocity_msg
            self.odom_msg.pose.pose.position.x = 0.0
            self.odom_msg.pose.pose.position.y = 0.0
            self.odom_msg.pose.pose.position.z = 0.0

            self.odom_publisher.publish(self.odom_msg)


if __name__ == "__main__":
    try:
        node = ModelOdometryNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
