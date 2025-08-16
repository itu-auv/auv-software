#!/usr/bin/env python3

import rospy
import math
from enum import Enum

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.msg import PropsYaw
from auv_msgs.srv import VisualServoing, VisualServoingResponse
from dynamic_reconfigure.server import Server
from auv_control.cfg import VisualServoingConfig


class ControllerState(Enum):
    """Defines the states of the Visual Servoing Controller."""

    IDLE = "idle"
    CENTERING = "centering"
    NAVIGATING = "navigating"
    SEARCHING = "searching"


class VisualServoingControllerNoIMU:
    """
    A controller for visual servoing towards a target prop without using an IMU.
    It uses a PD controller to align the robot with the target based on the angle to the prop.
    Navigate towards the target with a constant forward velocity.
    """

    def __init__(self):
        rospy.init_node("visual_servoing_controller_no_imu", anonymous=True)
        rospy.loginfo("Visual Servoing Controller (No IMU) node started")

        self._load_parameters()
        self._setup_state()
        self._setup_ros_communication()
        self._setup_dynamic_reconfigure()

    def _load_parameters(self):
        """Load parameters from the ROS parameter server."""
        self.kp_gain = rospy.get_param("~kp_gain")
        self.kd_gain = rospy.get_param("~kd_gain")
        self.v_x_desired = rospy.get_param("~v_x_desired")
        self.navigation_timeout_after_prop_disappear_s = rospy.get_param(
            "~navigation_timeout_after_prop_disappear_s"
        )
        self.overall_timeout_s = rospy.get_param("~overall_timeout_s")
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)
        self.prop_lost_timeout_s = rospy.get_param("~prop_lost_timeout_s")
        self.search_angular_velocity = rospy.get_param("~search_angular_velocity")
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity")

    def _setup_state(self):
        """Initialize the controller's state."""
        self.state = ControllerState.IDLE
        self.target_prop = ""
        self.service_start_time = None
        self.last_prop_stamp_time = None
        # Controller state
        self.error = 0.0
        self.error_derivative = 0.0
        self.last_error_stamp = None
        self.last_known_error = 0.0

    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and services."""
        # Publishers
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.control_enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.error_pub = rospy.Publisher("visual_servoing/error", Float64, queue_size=1)

        # Subscribers
        rospy.Subscriber("props_yaw", PropsYaw, self.prop_yaw_callback, queue_size=1)

        # Services
        rospy.Service(
            "visual_servoing/start", VisualServoing, self.handle_start_request
        )
        rospy.Service("visual_servoing/cancel", Trigger, self.handle_cancel_request)
        rospy.Service("visual_servoing/navigate", Trigger, self.handle_navigate_request)
        rospy.Service(
            "visual_servoing/cancel_navigation",
            Trigger,
            self.handle_cancel_navigation_request,
        )

    def _setup_dynamic_reconfigure(self):
        """Setup dynamic reconfigure server."""
        self.srv = Server(VisualServoingConfig, self.reconfigure_callback)

    def prop_yaw_callback(self, msg: PropsYaw):
        """Handles incoming prop yaw messages to update the controller error."""
        if self.state == ControllerState.IDLE or msg.object != self.target_prop:
            return

        current_stamp = msg.header.stamp
        self.last_prop_stamp_time = current_stamp
        current_error = msg.angle

        if self.last_error_stamp is not None:
            dt = (current_stamp - self.last_error_stamp).to_sec()
            if dt > 0.001:  # Avoid division by zero and stale data
                self.error_derivative = (current_error - self.error) / dt

        self.error = current_error
        self.last_known_error = current_error
        self.last_error_stamp = current_stamp
        self.error_pub.publish(Float64(self.error))
        rospy.loginfo(
            f"Prop yaw callback. New error: {self.error:.4f} in state: {self.state}"
        )

        if self.state == ControllerState.SEARCHING:
            self.state = ControllerState.CENTERING
            rospy.loginfo("Prop found, returning to centering.")

    def control_step(self):
        """Executes one iteration of the control loop."""
        twist = Twist()
        twist.angular.z = self._calculate_angular_z()
        twist.linear.x = self._calculate_linear_x()
        self.cmd_vel_pub.publish(twist)

    def _calculate_angular_z(self) -> float:
        """Calculates the angular velocity command using a PD controller."""
        if self.state == ControllerState.SEARCHING:
            if self.last_known_error != 0.0:
                # Turn towards the side where the object was last seen.
                # A positive error means the object is to the right (positive angle),
                # so we need a negative (clockwise) angular velocity to turn right.
                return -math.copysign(
                    self.search_angular_velocity, self.last_known_error
                )
            else:
                # If we have no last error (e.g., prop never seen after start), turn one way.
                return -self.search_angular_velocity

        if self.last_prop_stamp_time is None:
            rospy.logwarn_throttle(
                1.0, "No prop has been seen yet, cannot calculate angular velocity."
            )
            return 0.0

        p_signal = self.kp_gain * self.error
        d_signal = self.kd_gain * self.error_derivative

        # We want to turn to reduce the error. If error is positive (prop on the right),
        # we need a negative angular velocity (turn right).
        # The damping term (d_signal) should oppose the motion.
        angular_z = p_signal + d_signal
        return max(
            min(angular_z, self.max_angular_velocity), -self.max_angular_velocity
        )

    def _calculate_linear_x(self) -> float:
        """Calculates the linear velocity command."""
        if self.state != ControllerState.NAVIGATING:
            return 0.0

        if self.last_prop_stamp_time is None:
            rospy.logwarn_throttle(
                1.0, "In navigation mode but no prop has been seen yet."
            )
            return 0.0

        # time_since_last_prop = (rospy.Time.now() - self.last_prop_stamp_time).to_sec()
        # if time_since_last_prop > self.navigation_timeout_after_prop_disappear_s:
        #     rospy.loginfo(
        #         "Navigation timeout reached. Stopping forward motion and returning to centering."
        #     )
        #     self.state = ControllerState.CENTERING
        #     return 0.0
        return self.v_x_desired

    def reconfigure_callback(self, config, level):
        """Handles dynamic reconfigure updates for controller gains."""
        self.kp_gain = config.kp_gain
        self.kd_gain = config.kd_gain
        self.v_x_desired = config.v_x_desired
        self.navigation_timeout_after_prop_disappear_s = (
            config.navigation_timeout_after_prop_disappear_s
        )
        self.overall_timeout_s = config.overall_timeout_s
        self.prop_lost_timeout_s = config.prop_lost_timeout_s
        self.search_angular_velocity = config.search_angular_velocity
        self.max_angular_velocity = config.max_angular_velocity
        rospy.loginfo(
            f"Updated params: Kp={self.kp_gain}, kd={self.kd_gain}, VxDesired={self.v_x_desired}, NavTimeout={self.navigation_timeout_after_prop_disappear_s}, OverallTimeout={self.overall_timeout_s}, PropLostTimeout={self.prop_lost_timeout_s}, SearchAngularVelocity={self.search_angular_velocity}, MaxAngularVelocity={self.max_angular_velocity}"
        )
        return config

    def handle_start_request(self, req: VisualServoing) -> VisualServoingResponse:
        """Starts the visual servoing process."""
        if self.state != ControllerState.IDLE and req.target_prop == self.target_prop:
            return VisualServoingResponse(
                success=False,
                message="VS Controller is already active for this target.",
            )

        self.target_prop = req.target_prop
        self.state = ControllerState.CENTERING
        self.service_start_time = rospy.Time.now()
        self.last_prop_stamp_time = None
        # Reset controller state
        self.error = 0.0
        self.error_derivative = 0.0
        self.last_error_stamp = None
        self.last_known_error = 0.0
        rospy.loginfo(f"Visual servoing started for target: {self.target_prop}")
        return VisualServoingResponse(
            success=True, message="Visual servoing activated."
        )

    def handle_cancel_request(self, req: Trigger) -> TriggerResponse:
        """Stops the visual servoing process."""
        if self.state == ControllerState.IDLE:
            return TriggerResponse(success=False, message="Controller is not active.")

        self._stop_controller("cancelled by request")
        return TriggerResponse(success=True, message="Visual servoing deactivated.")

    def handle_navigate_request(self, req: Trigger) -> TriggerResponse:
        """Switches to navigation mode."""
        if self.state == ControllerState.IDLE:
            return TriggerResponse(
                success=False, message="Controller is not in centering mode."
            )
        if self.state == ControllerState.NAVIGATING:
            return TriggerResponse(
                success=False, message="Controller is already in navigation mode."
            )

        self.state = ControllerState.NAVIGATING
        rospy.loginfo(
            f"Visual servoing navigation started. Current error: {self.error}"
        )
        return TriggerResponse(success=True, message="Navigation mode activated.")

    def handle_cancel_navigation_request(self, req: Trigger) -> TriggerResponse:
        """Cancels navigation mode and returns to centering."""
        if self.state != ControllerState.NAVIGATING:
            return TriggerResponse(
                success=False, message="Controller is not in navigation mode."
            )

        self.state = ControllerState.CENTERING
        rospy.loginfo("Visual servoing navigation cancelled.")
        return TriggerResponse(success=True, message="Navigation mode deactivated.")

    def _stop_controller(self, reason: str):
        """Helper function to stop all motion and reset state."""
        rospy.loginfo(f"Visual servoing stopped: {reason}")
        self.state = ControllerState.IDLE
        self.cmd_vel_pub.publish(Twist())  # Stop the vehicle
        rospy.sleep(0.1)  # Give time for the stop command to be sent
        self.control_enable_pub.publish(Bool(data=False))

    def spin(self):
        """The main loop of the controller."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.state == ControllerState.IDLE:
                rate.sleep()
                continue

            if (
                rospy.Time.now() - self.service_start_time
            ).to_sec() > self.overall_timeout_s:
                self._stop_controller("overall timeout reached")
                continue

            if self.state == ControllerState.CENTERING:
                if (
                    self.last_prop_stamp_time is not None
                    and (rospy.Time.now() - self.last_prop_stamp_time).to_sec()
                    > self.prop_lost_timeout_s
                ):
                    rospy.logwarn("Prop lost during centering, starting search.")
                    self.state = ControllerState.SEARCHING
                    self.error = 0.0
                    self.error_derivative = 0.0

            self.control_enable_pub.publish(Bool(True))
            self.control_step()
            rate.sleep()


if __name__ == "__main__":
    try:
        controller = VisualServoingControllerNoIMU()
        controller.spin()
    except rospy.ROSInterruptException:
        pass
