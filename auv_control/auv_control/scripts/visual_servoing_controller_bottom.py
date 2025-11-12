#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bottom Camera Visual Servoing Controller (ROS1)

Bu kontrolcü, bottle_error_detector tarafından yayınlanan hataları kullanarak
araç üzerinde wrench (kuvvet/tork) üretir.

Mode 1: Basit PD kontrolcü
  - data[0] (e_vert_px)  → Y wrench
  - data[1] (e_horiz_px) → X wrench
  - data[2] (angle_error) → Angular Z wrench

PD kontrolcü kullanır ve dynamic reconfigure ile ayarlanabilir.
"""

import rospy
import math

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Trigger, TriggerResponse
from dynamic_reconfigure.server import Server
from auv_control.cfg import BottomVisualServoingConfig


class BottomVisualServoingController:
    """
    Bottom camera visual servoing controller using PD control.

    Mode 1: Basit hizalama
      - data[0] → Y wrench
      - data[1] → X wrench
      - data[2] → Angular Z wrench
    """

    def __init__(self):
        rospy.init_node("bottom_visual_servoing_controller", anonymous=True)
        rospy.loginfo("Bottom Visual Servoing Controller node started")

        self._load_parameters()
        self._setup_state()
        self._setup_ros_communication()
        self._setup_dynamic_reconfigure()

    def _load_parameters(self):
        """Load parameters from the ROS parameter server."""
        # PD gains (will be overridden by dynamic reconfigure)
        self.kp_x = rospy.get_param("~kp_x", 0.2)
        self.kd_x = rospy.get_param("~kd_x", 0.2)
        self.kp_y = rospy.get_param("~kp_y", 0.2)
        self.kd_y = rospy.get_param("~kd_y", 0.2)
        self.kp_angle = rospy.get_param("~kp_yaw", 0.8)
        self.kd_angle = rospy.get_param("~kd_yaw", 0.4)

        # Max wrench limits
        self.max_force_x = rospy.get_param("~max_force_x", 50.0)
        self.max_force_y = rospy.get_param("~max_force_y", 50.0)
        self.max_torque_z = rospy.get_param("~max_torque_z", 20.0)

        # Control rate
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)

    def _setup_state(self):
        """Initialize the controller's state."""
        self.active = False

        # Error values from bottle_error_detector
        self.e_vert_px = float("nan")  # Y error in pixels
        self.e_horiz_px = float("nan")  # X error in pixels
        self.angle_error = float("nan")  # Angular error in radians

        # Previous errors for derivative calculation
        self.prev_e_x = None
        self.prev_e_y = None
        self.prev_e_angle = None
        self.last_error_time = None

        # Killswitch state
        self.killswitch_active = True

    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and services."""
        # Publishers
        self.wrench_pub = rospy.Publisher("wrench", WrenchStamped, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "bottle_vsc_errors", Float32MultiArray, self.error_callback, queue_size=1
        )
        rospy.Subscriber(
            "propulsion_board/status", Bool, self.killswitch_callback, queue_size=1
        )

        # Services
        rospy.Service("bottom_vsc/start", Trigger, self.handle_start)
        rospy.Service("bottom_vsc/cancel", Trigger, self.handle_cancel)

    def _setup_dynamic_reconfigure(self):
        """Setup dynamic reconfigure server."""
        self.srv = Server(BottomVisualServoingConfig, self.reconfigure_callback)

    def reconfigure_callback(self, config, level):
        """Handles dynamic reconfigure updates for controller gains."""
        self.kp_x = config.kp_x
        self.kd_x = config.kd_x
        self.kp_y = config.kp_y
        self.kd_y = config.kd_y
        self.kp_angle = config.kp_yaw
        self.kd_angle = config.kd_yaw

        self.max_force_x = config.max_force_x
        self.max_force_y = config.max_force_y
        self.max_torque_z = config.max_torque_z

        rospy.loginfo(
            f"Updated gains: Kp_x={self.kp_x}, Kd_x={self.kd_x}, "
            f"Kp_y={self.kp_y}, Kd_y={self.kd_y}, "
            f"Kp_angle={self.kp_angle}, Kd_angle={self.kd_angle}"
        )
        return config

    def error_callback(self, msg: Float32MultiArray):
        """
        Handles incoming error messages from bottle_error_detector.
        data[0] = e_vert_px (Y error)
        data[1] = e_horiz_px (X error)
        data[2] = angle_to_vertical (angular error)
        """
        if len(msg.data) >= 3:
            self.e_vert_px = msg.data[0]
            self.e_horiz_px = msg.data[1]
            self.angle_error = msg.data[2]

    def killswitch_callback(self, msg: Bool):
        """Handles killswitch status."""
        self.killswitch_active = not msg.data
        if self.killswitch_active:
            self.active = False
            rospy.logwarn("Killswitch activated! Controller stopped.")

    def handle_start(self, req):
        """
        Start Mode 1: Simple XY + Angular alignment
        """
        if self.killswitch_active:
            return TriggerResponse(
                success=False, message="Cannot start: killswitch is active"
            )

        self.active = True
        self.prev_e_x = None
        self.prev_e_y = None
        self.prev_e_angle = None
        self.last_error_time = None

        self.enable_pub.publish(Bool(True))
        rospy.loginfo("Started Mode 1: XY + Angular alignment")
        return TriggerResponse(success=True, message="Mode 1 started")

    def handle_cancel(self, req):
        """Cancel the visual servoing control."""
        was_active = self.active
        self.active = False

        # Publish zero wrench
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.header.frame_id = "taluy/base_link"
        self.wrench_pub.publish(wrench_msg)

        self.enable_pub.publish(Bool(False))

        if was_active:
            rospy.loginfo("Bottom visual servoing cancelled")
            return TriggerResponse(success=True, message="Control cancelled")
        else:
            return TriggerResponse(success=True, message="Controller was not active")

    def control_step(self):
        """Executes one iteration of the control loop."""
        if not self.active:
            return

        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.header.frame_id = "taluy/base_link"

        current_time = rospy.Time.now()
        dt = 0.0
        if self.last_error_time is not None:
            dt = (current_time - self.last_error_time).to_sec()

        # Calculate forces and torques
        force_x = self._calculate_force_x(dt)
        force_y = self._calculate_force_y(dt)
        torque_z = self._calculate_torque_z(dt)

        # Apply limits
        force_x = self._clamp(force_x, -self.max_force_x, self.max_force_x)
        force_y = self._clamp(force_y, -self.max_force_y, self.max_force_y)
        torque_z = self._clamp(torque_z, -self.max_torque_z, self.max_torque_z)

        # Populate wrench message
        wrench_msg.wrench.force.x = force_x
        wrench_msg.wrench.force.y = force_y
        wrench_msg.wrench.torque.z = torque_z

        self.wrench_pub.publish(wrench_msg)
        self.last_error_time = current_time

    def _calculate_force_x(self, dt: float) -> float:
        """
        Calculate force in X direction (body frame).
        Uses data[1] (e_horiz_px)
        Error convention: obje solda → e_horiz_px pozitif → araç -X (sağa) gitmeli
        """
        if math.isnan(self.e_horiz_px):
            return 0.0

        # Direkt piksel değeriyle çalış
        error = self.e_horiz_px

        # PD control: force = -Kp * error - Kd * derror/dt
        # Negative because positive error means we want negative force (right)
        p_term = -self.kp_x * error

        d_term = 0.0
        if self.prev_e_x is not None and dt > 0:
            error_derivative = (error - self.prev_e_x) / dt
            d_term = -self.kd_x * error_derivative

        self.prev_e_x = error
        return p_term + d_term

    def _calculate_force_y(self, dt: float) -> float:
        """
        Calculate force in Y direction (body frame).
        Uses data[0] (e_vert_px)
        Error convention: obje yukarıda → e_vert_px pozitif → araç -Y (aşağı) gitmeli
        """
        if math.isnan(self.e_vert_px):
            return 0.0

        # Direkt piksel değeriyle çalış
        error = self.e_vert_px

        # PD control: force = -Kp * error - Kd * derror/dt
        # Negative because positive error means we want negative force (down)
        p_term = -self.kp_y * error

        d_term = 0.0
        if self.prev_e_y is not None and dt > 0:
            error_derivative = (error - self.prev_e_y) / dt
            d_term = -self.kd_y * error_derivative

        self.prev_e_y = error
        return p_term + d_term

    def _calculate_torque_z(self, dt: float) -> float:
        """
        Calculate torque for angular alignment with bottle orientation.
        Uses data[2] (angle_error) from bottle_error_detector.
        """
        if math.isnan(self.angle_error):
            return 0.0

        # Angle error is already in radians
        # Positive angle → bottle tilted clockwise → need counter-clockwise torque
        error = self.angle_error

        # PD control
        p_term = -self.kp_angle * error

        d_term = 0.0
        if self.prev_e_angle is not None and dt > 0:
            error_derivative = (error - self.prev_e_angle) / dt
            d_term = -self.kd_angle * error_derivative

        self.prev_e_angle = error
        return p_term + d_term

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo(f"Bottom VSC running at {self.rate_hz} Hz")

        while not rospy.is_shutdown():
            try:
                self.control_step()
            except Exception as e:
                rospy.logerr(f"Error in control step: {e}")

            rate.sleep()


def main():
    try:
        controller = BottomVisualServoingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
