#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bottom Camera Visual Servoing Controller (ROS1)
STEP 1: YAW-only control

Subscribes:
  - bottle_vsc_errors (Float32MultiArray):
      data[0] = e_vert_px (Y error in pixels) [NOT USED YET]
      data[1] = e_horiz_px (X error in pixels) [NOT USED YET]
      data[2] = angle_to_vertical (radians, [-pi/2, +pi/2]) [USED]
  - propulsion_board/status (Bool): killswitch status

Publishes:
  - wrench (WrenchStamped): Only torque.z for yaw alignment

Services:
  - bottom_vsc/start: Start the controller
  - bottom_vsc/cancel: Stop the controller

Uses P or PD control (configurable via dynamic reconfigure).
"""

import rospy
import math

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Trigger, TriggerResponse
from dynamic_reconfigure.server import Server
from auv_control.cfg import BottomVisualServoingConfig


# Timeout for error messages (seconds)
ERROR_TIMEOUT = 0.5


class BottomVisualServoingController:
    """
    STEP 1: Yaw-only visual servoing controller.
    Reads data[2] (angle_error) and produces torque.z
    """

    def __init__(self):
        rospy.init_node("bottom_visual_servoing_controller", anonymous=True)
        rospy.loginfo("=== Bottom VSC - STEP 1: YAW ONLY ===")

        self._load_parameters()
        self._setup_state()
        self._setup_ros_communication()
        self._setup_dynamic_reconfigure()

    def _load_parameters(self):
        """Load parameters from ROS parameter server."""
        self.kp_yaw = rospy.get_param("~kp_yaw", 0.01)
        self.kd_yaw = rospy.get_param("~kd_yaw", 0.005)
        self.max_torque_z = rospy.get_param("~max_torque_z", 5.0)
        self.use_derivative = rospy.get_param("~use_derivative", False)
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)

    def _setup_state(self):
        """Initialize controller state."""
        self.active = False
        self.angle_error = float("nan")
        self.prev_angle_error = None
        self.last_error_time = None
        self.last_error_received_time = None  # Track when we last got error data
        self.killswitch_active = True

    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and services."""
        # Publisher (removed 'enable' publisher - not needed)
        self.wrench_pub = rospy.Publisher("wrench", WrenchStamped, queue_size=10)

        # Subscribers
        rospy.Subscriber(
            "bottle_vsc_errors", Float32MultiArray, self._error_callback, queue_size=1
        )
        rospy.Subscriber(
            "propulsion_board/status", Bool, self._killswitch_callback, queue_size=1
        )

        # Services
        rospy.Service("bottom_vsc/start", Trigger, self._handle_start)
        rospy.Service("bottom_vsc/cancel", Trigger, self._handle_cancel)

    def _setup_dynamic_reconfigure(self):
        """Setup dynamic reconfigure server."""
        self.dyn_srv = Server(BottomVisualServoingConfig, self._reconfigure_callback)

    def _reconfigure_callback(self, config, level):
        """Handle dynamic reconfigure updates."""
        self.kp_yaw = config.kp_yaw
        self.kd_yaw = config.kd_yaw
        self.max_torque_z = config.max_torque_z
        self.use_derivative = config.use_derivative

        rospy.loginfo(
            f"Config updated: Kp={self.kp_yaw:.4f}, Kd={self.kd_yaw:.4f}, "
            f"Max_T={self.max_torque_z:.2f}, Use_D={self.use_derivative}"
        )
        return config

    def _error_callback(self, msg: Float32MultiArray):
        """
        Receive error from bottle_error_detector.
        data[2] = angle_to_vertical (radians, [-pi/2, +pi/2])
        """
        if len(msg.data) >= 3:
            self.angle_error = msg.data[2]
            self.last_error_received_time = rospy.Time.now()

    def _killswitch_callback(self, msg: Bool):
        """Handle killswitch status."""
        self.killswitch_active = not msg.data
        if self.killswitch_active and self.active:
            self.active = False
            rospy.logwarn("Killswitch activated! Controller stopped.")

    def _handle_start(self, req):
        """Start the yaw controller."""
        if self.killswitch_active:
            return TriggerResponse(
                success=False, message="Cannot start: killswitch is active"
            )

        self.active = True
        self.prev_angle_error = None
        self.last_error_time = None

        rospy.loginfo("STEP 1: Yaw controller started")
        return TriggerResponse(success=True, message="Yaw control started")

    def _handle_cancel(self, req):
        """Cancel the controller."""
        was_active = self.active
        self.active = False

        # Publish zero wrench
        self._publish_zero_wrench()

        if was_active:
            rospy.loginfo("Controller cancelled")
            return TriggerResponse(success=True, message="Controller cancelled")
        else:
            return TriggerResponse(success=True, message="Controller was not active")

    def _publish_zero_wrench(self):
        """Publish a zero wrench message."""
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.header.frame_id = "taluy/base_link"
        self.wrench_pub.publish(wrench_msg)

    def control_step(self):
        """Execute one control loop iteration."""
        if not self.active:
            return

        # Check if we're receiving error messages
        if self.last_error_received_time is None:
            rospy.logwarn_throttle(5.0, "Waiting for bottle_vsc_errors messages...")
            return

        # Check if error data is fresh (timeout check)
        time_since_last_error = (
            rospy.Time.now() - self.last_error_received_time
        ).to_sec()
        if time_since_last_error > ERROR_TIMEOUT:
            rospy.logwarn_throttle(
                2.0,
                f"No fresh error data ({time_since_last_error:.2f}s old). Not publishing wrench.",
            )
            return

        # Check if we have valid error
        if math.isnan(self.angle_error):
            rospy.logwarn_throttle(5.0, "Angle error is NaN. Not publishing wrench.")
            return

        # Calculate dt
        current_time = rospy.Time.now()
        dt = 0.0
        if self.last_error_time is not None:
            dt = (current_time - self.last_error_time).to_sec()

        # Calculate torque
        torque_z = self._calculate_torque_z(dt)

        # Publish wrench
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = current_time
        wrench_msg.header.frame_id = "taluy/base_link"
        wrench_msg.wrench.torque.z = torque_z
        self.wrench_pub.publish(wrench_msg)

        # Debug log (every ~2 seconds)
        if rospy.get_time() % 2.0 < 0.1:
            rospy.loginfo(
                f"YAW: error={math.degrees(self.angle_error):+7.2f}° | "
                f"torque={torque_z:+7.3f} Nm | limit=±{self.max_torque_z:.1f} | "
                f"data_age={time_since_last_error:.3f}s"
            )

        self.last_error_time = current_time

    def _calculate_torque_z(self, dt: float) -> float:
        """
        Calculate yaw torque using P or PD control.
        Positive angle error → need positive torque (same sign)
        """
        if math.isnan(self.angle_error):
            return 0.0

        # Proportional term
        p_term = self.kp_yaw * self.angle_error

        # Derivative term (optional)
        d_term = 0.0
        if self.use_derivative and self.prev_angle_error is not None and dt > 1e-6:
            error_derivative = (self.angle_error - self.prev_angle_error) / dt
            # Clamp derivative to prevent spikes
            error_derivative = max(-100.0, min(100.0, error_derivative))
            d_term = self.kd_yaw * error_derivative

        self.prev_angle_error = self.angle_error

        # Total torque
        torque = p_term + d_term

        # Clamp to max limit
        return max(-self.max_torque_z, min(self.max_torque_z, torque))

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo(f"Controller running at {self.rate_hz} Hz")

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
