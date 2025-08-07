#!/usr/bin/env python3

import rospy
import tf.transformations
import math
from collections import deque
from enum import Enum

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.msg import PropsYaw
from auv_msgs.srv import VisualServoing, VisualServoingResponse
from dynamic_reconfigure.server import Server
from auv_control.cfg import VisualServoingConfig
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Wrench


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ControllerState(Enum):
    """Defines the states of the Visual Servoing Controller."""

    IDLE = "idle"
    CENTERING = "centering"
    NAVIGATING = "navigating"


class VisualServoingController:
    """
    A controller for visual servoing towards a target prop.
    It uses a PD controller to align the robot's yaw with the target.
    Navigate towards the target with a constant forward velocity.
    """

    def __init__(self):
        rospy.init_node("visual_servoing_controller", anonymous=True)
        rospy.loginfo("Visual Servoing Controller node started")

        self._load_parameters()
        self._setup_state()
        self._setup_ros_communication()
        self._setup_dynamic_reconfigure()

    def _load_parameters(self):
        """Load parameters from the ROS parameter server."""
        self.kp_gain = rospy.get_param("~kp_gain", 0.8)
        self.kd_gain = rospy.get_param("~kd_gain", 0.4)
        self.v_x_desired = rospy.get_param("~v_x_desired", 0.3)
        self.navigation_timeout_after_prop_disappear_s = 12.0
        self.overall_timeout_s = rospy.get_param("~overall_timeout_s", 1500.0)
        self.rate_hz = rospy.get_param("~rate_hz", 10.0)
        imu_history_secs = rospy.get_param("~imu_history_secs", 2.0)
        self.imu_history_size = int(self.rate_hz * imu_history_secs)

    def _setup_state(self):
        """Initialize the controller's state."""
        self.state = ControllerState.IDLE
        self.target_prop = ""
        self.service_start_time = None
        self.last_prop_stamp_time = None
        # Sensor-derived state
        self.current_yaw = 0.0
        self.angular_velocity_z = 0.0
        self.target_yaw_in_world = 0.0
        self.imu_history = deque(maxlen=self.imu_history_size)

    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and services."""
        # Publishers
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.control_enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.error_pub = rospy.Publisher("visual_servoing/error", Float64, queue_size=1)
        self.current_yaw_pub = rospy.Publisher(
            "visual_servoing/current_yaw", Float64, queue_size=1
        )
        self.target_yaw_pub = rospy.Publisher(
            "visual_servoing/target_yaw", Float64, queue_size=1
        )
        # WrenchStamped publisher
        self.wrench_pub = rospy.Publisher("cmd_wrench", Wrench, queue_size=10)

        # Subscribers
        rospy.Subscriber("props_yaw", PropsYaw, self.prop_yaw_callback, queue_size=1)
        rospy.Subscriber("imu/data", Imu, self.imu_callback, queue_size=1)

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

    def imu_callback(self, msg: Imu):
        """Handles incoming IMU messages to update yaw and angular velocity."""
        orientation_q = msg.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        self.current_yaw = yaw
        self.angular_velocity_z = msg.angular_velocity.z
        self.imu_history.append((msg.header.stamp, yaw))

    def prop_yaw_callback(self, msg: PropsYaw):
        """Handles incoming prop yaw messages to update the target yaw."""
        if self.state == ControllerState.IDLE or msg.object != self.target_prop:
            return

        if not self.imu_history:
            rospy.logwarn_throttle(
                1.0, "IMU history is empty, skipping prop yaw callback."
            )
            return

        prop_stamp = msg.header.stamp
        self.last_prop_stamp_time = prop_stamp
        angle_to_prop_from_robot = msg.angle

        # delay = (rospy.Time.now() - prop_stamp).to_sec()
        # rospy.loginfo_throttle(
        #    1.0, f"Visual Servoing perception delay: {delay:.3f} seconds"
        # )

        yaw_at_prop_time = self._get_yaw_at_time(prop_stamp)
        self.target_yaw_in_world = normalize_angle(
            yaw_at_prop_time + angle_to_prop_from_robot
        )

    def _get_yaw_at_time(self, stamp: rospy.Time) -> float:
        """Finds the yaw from IMU history closest to the given timestamp."""
        closest_imu_reading = min(self.imu_history, key=lambda x: abs(x[0] - stamp))
        return closest_imu_reading[1]

    def control_step(self):
        """Executes one iteration of the control loop."""
        twist = Twist()
        twist.angular.z = self._calculate_angular_z()
        twist.linear.x = self._calculate_linear_x()
        self.cmd_vel_pub.publish(twist)

        wrench = Wrench()
        scale_linear = 20.0
        scale_angular = 10.0
        wrench.force.x = twist.linear.x * scale_linear
        wrench.force.y = twist.linear.y * scale_linear
        wrench.force.z = twist.linear.z * scale_linear
        wrench.torque.x = twist.angular.x * scale_angular
        wrench.torque.y = twist.angular.y * scale_angular
        wrench.torque.z = twist.angular.z * scale_angular
        self.wrench_pub.publish(wrench)

    def _calculate_angular_z(self) -> float:
        """Calculates the angular velocity command using a PD controller."""
        error = normalize_angle(self.target_yaw_in_world - self.current_yaw)
        self.error_pub.publish(Float64(error))
        self.current_yaw_pub.publish(Float64(self.current_yaw))
        self.target_yaw_pub.publish(Float64(self.target_yaw_in_world))

        p_signal = self.kp_gain * error
        d_signal = self.kd_gain * self.angular_velocity_z
        return p_signal - d_signal

    def _calculate_linear_x(self) -> float:
        """Calculates the linear velocity command."""
        if self.state != ControllerState.NAVIGATING:
            return 0.0

        if self.last_prop_stamp_time is None:
            rospy.logwarn_throttle(
                1.0, "In navigation mode but no prop has been seen yet."
            )
            return 0.0

        time_since_last_prop = (rospy.Time.now() - self.last_prop_stamp_time).to_sec()
        if time_since_last_prop > self.navigation_timeout_after_prop_disappear_s:
            rospy.loginfo(
                "Navigation timeout reached. Stopping forward motion and returning to centering."
            )
            self.state = ControllerState.CENTERING
            return 0.0
        return self.v_x_desired

    def reconfigure_callback(self, config, level):
        """Handles dynamic reconfigure updates for controller gains."""
        self.kp_gain = config.kp_gain
        self.kd_gain = config.kd_gain
        self.navigation_timeout_after_prop_disappear_s = (
            config.navigation_timeout_after_prop_disappear_s
        )
        rospy.loginfo(
            f"Updated gains: Kp={self.kp_gain}, kd={self.kd_gain}, NavTimeout={self.navigation_timeout_after_prop_disappear_s}"
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
        self.target_yaw_in_world = self.current_yaw
        self.state = ControllerState.CENTERING
        self.service_start_time = rospy.Time.now()
        self.last_prop_stamp_time = None
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
        rospy.loginfo("Visual servoing navigation started.")
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

            self.control_enable_pub.publish(Bool(True))
            self.control_step()
            rate.sleep()


if __name__ == "__main__":
    try:
        controller = VisualServoingController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass
