#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
import numpy as np
import message_filters
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class DynamicModelOdometry:
    def __init__(self):
        rospy.init_node("dynamic_model_odometry_node", anonymous=True)

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

        self.load_dynamic_model()

        self.wrench_sub = message_filters.Subscriber(
            "wrench", WrenchStamped, tcp_nodelay=True
        )
        self.odom_sub = message_filters.Subscriber(
            "odometry", Odometry, tcp_nodelay=True
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.wrench_sub, self.odom_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.synced_callback)

        self.odom_publisher = rospy.Publisher(
            "odom_dynamic_model", Odometry, queue_size=10
        )
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"

        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.update_twist_covariance()

        self.estimated_velocity = np.zeros(6)
        self.last_model_update = rospy.Time.now()

        node_colored = TerminalColors.color_text(
            "Dynamic Model Odometry", TerminalColors.PASTEL_GREEN
        )
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

    def compute_model_based_velocity(self, wrench, velocity, dt):
        v = velocity.copy()

        damping_linear = np.dot(self.D_linear, v)

        v_abs = np.abs(v)
        damping_quad = np.dot(self.D_quadratic, v_abs * v)

        C_matrix = self.calculate_coriolis_matrix(v)
        coriolis_force = np.dot(C_matrix, v)

        net_force = wrench - damping_linear - damping_quad - coriolis_force

        acceleration = np.dot(self.M_inv, net_force)

        self.estimated_velocity = v + acceleration * dt

        return self.estimated_velocity

    def synced_callback(self, wrench_msg, odom_msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_model_update).to_sec()
        self.last_model_update = current_time

        if dt <= 0.001 or dt >= 1.0:
            return

        wrench = np.array(
            [
                wrench_msg.wrench.force.x,
                wrench_msg.wrench.force.y,
                wrench_msg.wrench.force.z,
                wrench_msg.wrench.torque.x,
                wrench_msg.wrench.torque.y,
                wrench_msg.wrench.torque.z,
            ]
        )

        velocity = np.array(
            [
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y,
                odom_msg.twist.twist.linear.z,
                odom_msg.twist.twist.angular.x,
                odom_msg.twist.twist.angular.y,
                odom_msg.twist.twist.angular.z,
            ]
        )

        model_vel = self.compute_model_based_velocity(wrench, velocity, dt)

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

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = DynamicModelOdometry()
        node.run()
    except rospy.ROSInterruptException:
        pass
