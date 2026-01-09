#!/usr/bin/env python3
"""
Visual Servoing Controller

Modes:
- Standard: Track single prop using PropsYaw messages
- Slalom: Track between pipe pairs using ObjectDetectionArray

IMU Options:
- use_imu=True: World-frame heading control (compensates for perception delay)
- use_imu=False: Image-space control (direct error from detection angle/centroid)

Services:
- visual_servoing/start: Start tracking a target prop
- visual_servoing/navigate: Switch to forward navigation mode
- visual_servoing/cancel: Stop all tracking
- visual_servoing/cancel_navigation: Return to centering mode

Topics Subscribed:
- props_yaw (PropsYaw): Standard prop detection
- slalom_pipes (ObjectDetectionArray): Slalom pipe detections (when slalom_mode=True)
- imu/data (Imu): IMU for heading (when use_imu=True)

Topics Published:
- cmd_vel (Twist): Velocity commands
- enable (Bool): Control enable signal
- visual_servoing/error (Float64): Current heading error
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import rospy
import tf.transformations
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger, TriggerResponse

from auv_control.cfg import VisualServoingConfig
from auv_msgs.msg import ObjectDetection, ObjectDetectionArray, PropsYaw
from auv_msgs.srv import VisualServoing, VisualServoingResponse


# =============================================================================
# Utility Functions
# =============================================================================


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ControllerConfig:
    """Configuration parameters for the controller."""

    kp_gain: float = 0.8
    kd_gain: float = 0.4
    v_x_desired: float = 0.3
    rate_hz: float = 10.0
    overall_timeout_s: float = 1500.0
    navigation_timeout_s: float = 12.0
    imu_history_secs: float = 2.0
    max_angular_velocity: float = 1.0
    # Mode switches
    use_imu: bool = True
    slalom_mode: bool = False


@dataclass
class ErrorState:
    """Tracks error for PD control."""

    error: float = 0.0
    error_derivative: float = 0.0
    last_stamp: Optional[rospy.Time] = None

    def reset(self):
        """Reset error state."""
        self.error = 0.0
        self.error_derivative = 0.0
        self.last_stamp = None


@dataclass
class SlalomState:
    """State for slalom pipe navigation."""

    detections: List[ObjectDetection] = field(default_factory=list)
    last_stamp: Optional[rospy.Time] = None

    def reset(self):
        """Reset slalom state."""
        self.detections = []
        self.last_stamp = None


# =============================================================================
# Controller State Enum
# =============================================================================


class ControllerState(Enum):
    """Defines the states of the Visual Servoing Controller."""

    IDLE = "idle"
    CENTERING = "centering"
    NAVIGATING = "navigating"


# =============================================================================
# Main Controller Class
# =============================================================================


class VisualServoingController:
    """
    A controller for visual servoing towards a target prop.

    Supports two modes of operation:
    - IMU mode: Uses world-frame heading with IMU compensation
    - Image-space mode: Uses detection angle directly as error

    Also supports slalom mode for navigating between pipe pairs.
    """

    def __init__(self):
        rospy.init_node("visual_servoing_controller", anonymous=True)
        rospy.loginfo("Visual Servoing Controller node started")

        self._load_config()
        self._init_state()
        self._setup_ros_communication()
        self._setup_dynamic_reconfigure()

    # -------------------------------------------------------------------------
    # Initialization Methods
    # -------------------------------------------------------------------------

    def _load_config(self):
        """Load parameters from the ROS parameter server."""
        self.config = ControllerConfig(
            kp_gain=rospy.get_param("~kp_gain", 0.8),
            kd_gain=rospy.get_param("~kd_gain", 0.4),
            v_x_desired=rospy.get_param("~v_x_desired", 0.3),
            rate_hz=rospy.get_param("~rate_hz", 10.0),
            overall_timeout_s=rospy.get_param("~overall_timeout_s", 1500.0),
            navigation_timeout_s=rospy.get_param(
                "~navigation_timeout_after_prop_disappear_s", 12.0
            ),
            imu_history_secs=rospy.get_param("~imu_history_secs", 2.0),
            max_angular_velocity=rospy.get_param("~max_angular_velocity", 1.0),
            use_imu=rospy.get_param("~use_imu", True),
            slalom_mode=rospy.get_param("~slalom_mode", False),
        )

        self.imu_history_size = int(self.config.rate_hz * self.config.imu_history_secs)

        rospy.loginfo(
            f"[VS Config] use_imu={self.config.use_imu} slalom_mode={self.config.slalom_mode}"
        )

    def _init_state(self):
        """Initialize the controller's state."""
        self.state = ControllerState.IDLE
        self.target_prop = ""
        self.service_start_time: Optional[rospy.Time] = None
        self.last_detection_time: Optional[rospy.Time] = None

        # IMU state (only used when use_imu=True)
        self.current_yaw = 0.0
        self.angular_velocity_z = 0.0
        self.target_yaw_in_world = 0.0
        self.imu_history: Deque[Tuple[rospy.Time, float]] = deque(
            maxlen=self.imu_history_size
        )

        # Error state (used when use_imu=False or slalom_mode=True)
        self.error_state = ErrorState()

        # Slalom state
        self.slalom_state = SlalomState()

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

        # Prop detection subscriber (standard mode)
        rospy.Subscriber(
            "props_yaw", PropsYaw, self._handle_prop_detection, queue_size=1
        )

        # IMU subscriber (conditional)
        if self.config.use_imu:
            rospy.Subscriber("imu/data", Imu, self._imu_callback, queue_size=1)
            rospy.loginfo("[VS] IMU subscriber enabled")
        else:
            rospy.loginfo("[VS] Running in image-space mode (no IMU)")

        # Slalom subscriber (conditional)
        if self.config.slalom_mode:
            rospy.Subscriber(
                "slalom_pipes",
                ObjectDetectionArray,
                self._slalom_detection_callback,
                queue_size=1,
            )
            rospy.loginfo("[VS] Slalom mode enabled - subscribing to slalom_pipes")

        # Services
        rospy.Service("visual_servoing/start", VisualServoing, self._handle_start)
        rospy.Service("visual_servoing/cancel", Trigger, self._handle_cancel)
        rospy.Service("visual_servoing/navigate", Trigger, self._handle_navigate)
        rospy.Service(
            "visual_servoing/cancel_navigation", Trigger, self._handle_cancel_navigation
        )

    def _setup_dynamic_reconfigure(self):
        """Setup dynamic reconfigure server."""
        self.srv = Server(VisualServoingConfig, self._reconfigure_callback)

    # -------------------------------------------------------------------------
    # IMU Callback
    # -------------------------------------------------------------------------

    def _imu_callback(self, msg: Imu):
        """Handle incoming IMU messages to update yaw and angular velocity."""
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

    def _get_yaw_at_time(self, stamp: rospy.Time) -> float:
        """Find the yaw from IMU history closest to the given timestamp."""
        if not self.imu_history:
            return self.current_yaw
        closest_reading = min(self.imu_history, key=lambda x: abs(x[0] - stamp))
        return closest_reading[1]

    # -------------------------------------------------------------------------
    # Prop Detection Callback (Standard Mode)
    # -------------------------------------------------------------------------

    def _handle_prop_detection(self, msg: PropsYaw):
        """Handle incoming prop yaw messages."""
        if self.state == ControllerState.IDLE or msg.object != self.target_prop:
            return

        # Skip if in slalom mode (use slalom_pipes instead)
        if self.config.slalom_mode:
            return

        self.last_detection_time = msg.header.stamp

        if self.config.use_imu:
            self._update_target_with_imu(msg)
        else:
            self._update_error_direct(msg.angle)

    def _update_target_with_imu(self, msg: PropsYaw):
        """Update world-frame target yaw using IMU history."""
        if not self.imu_history:
            rospy.logwarn_throttle(1.0, "IMU history empty, skipping prop callback")
            return

        yaw_at_prop_time = self._get_yaw_at_time(msg.header.stamp)
        self.target_yaw_in_world = normalize_angle(yaw_at_prop_time + msg.angle)

    def _update_error_direct(self, angle: float):
        """Image-space control: use angle directly as error."""
        now = rospy.Time.now()
        if self.error_state.last_stamp is not None:
            dt = (now - self.error_state.last_stamp).to_sec()
            if dt > 0.001:
                self.error_state.error_derivative = (
                    angle - self.error_state.error
                ) / dt

        self.error_state.error = angle
        self.error_state.last_stamp = now

    # -------------------------------------------------------------------------
    # Slalom Detection Callback
    # -------------------------------------------------------------------------

    def _slalom_detection_callback(self, msg: ObjectDetectionArray):
        """Handle slalom pipe detections from depth segmentation."""
        if self.state == ControllerState.IDLE:
            return

        self.slalom_state.detections = list(msg.detections)
        self.slalom_state.last_stamp = msg.header.stamp
        self.last_detection_time = msg.header.stamp

        heading = self._compute_slalom_heading()
        if heading is not None:
            self._update_error_direct(heading)

    def _compute_slalom_heading(self) -> Optional[float]:
        """
        Select pipes for slalom navigation and compute heading.

        Strategy:
        1. If we have both red and white pipes, use closest of each color
        2. Otherwise, fall back to closest 2 pipes by depth

        Returns:
            Heading in [-1, 1] where 0 = centered, or None if insufficient data.
        """
        detections = self.slalom_state.detections
        if len(detections) < 2:
            rospy.logwarn_throttle(2.0, f"[SLALOM] Need 2 pipes, got {len(detections)}")
            return None

        # Try color-based selection first
        reds = [d for d in detections if d.color == "red"]
        whites = [d for d in detections if d.color == "white"]

        if reds and whites:
            # Use closest red and closest white
            pipe_red = min(reds, key=lambda d: d.depth)
            pipe_white = min(whites, key=lambda d: d.depth)
            heading = 0.5 * (pipe_red.centroid.x + pipe_white.centroid.x)

            rospy.loginfo_throttle(
                1.0,
                f"[SLALOM] RED x={pipe_red.centroid.x:.2f} d={pipe_red.depth:.2f} | "
                f"WHITE x={pipe_white.centroid.x:.2f} d={pipe_white.depth:.2f} | "
                f"heading={heading:.2f}",
            )
        else:
            # Fallback: closest 2 by depth
            sorted_pipes = sorted(detections, key=lambda d: d.depth)
            pipe_a = sorted_pipes[0]
            pipe_b = sorted_pipes[1]
            heading = 0.5 * (pipe_a.centroid.x + pipe_b.centroid.x)

            rospy.loginfo_throttle(
                1.0,
                f"[SLALOM] (depth fallback) pipe1: x={pipe_a.centroid.x:.2f} "
                f"pipe2: x={pipe_b.centroid.x:.2f} heading={heading:.2f}",
            )

        return heading

    # -------------------------------------------------------------------------
    # Control Computation
    # -------------------------------------------------------------------------

    def _execute_control_step(self):
        """Execute one iteration of the control loop."""
        twist = Twist()
        twist.angular.z = self._compute_angular_command()
        twist.linear.x = self._compute_linear_command()
        self.cmd_vel_pub.publish(twist)

    def _compute_angular_command(self) -> float:
        """Calculate angular velocity command using PD controller."""
        if self.config.use_imu and not self.config.slalom_mode:
            # World-frame control with IMU
            error = normalize_angle(self.target_yaw_in_world - self.current_yaw)
            d_term = self.config.kd_gain * self.angular_velocity_z

            # Publish debug info
            self.current_yaw_pub.publish(Float64(self.current_yaw))
            self.target_yaw_pub.publish(Float64(self.target_yaw_in_world))
        else:
            # Image-space control (no IMU or slalom mode)
            error = self.error_state.error
            d_term = self.config.kd_gain * self.error_state.error_derivative

        self.error_pub.publish(Float64(error))

        p_term = self.config.kp_gain * error
        cmd = p_term - d_term

        return max(
            min(cmd, self.config.max_angular_velocity),
            -self.config.max_angular_velocity,
        )

    def _compute_linear_command(self) -> float:
        """Calculate linear velocity command."""
        if self.state != ControllerState.NAVIGATING:
            return 0.0

        if self.last_detection_time is None:
            rospy.logwarn_throttle(1.0, "In navigation mode but no detection yet")
            return 0.0

        time_since_detection = (rospy.Time.now() - self.last_detection_time).to_sec()
        if time_since_detection > self.config.navigation_timeout_s:
            rospy.loginfo(
                "Navigation timeout reached. Stopping forward motion, returning to centering."
            )
            self.state = ControllerState.CENTERING
            return 0.0

        return self.config.v_x_desired

    # -------------------------------------------------------------------------
    # Dynamic Reconfigure
    # -------------------------------------------------------------------------

    def _reconfigure_callback(self, config, level):
        """Handle dynamic reconfigure updates."""
        self.config.kp_gain = config.kp_gain
        self.config.kd_gain = config.kd_gain
        self.config.navigation_timeout_s = (
            config.navigation_timeout_after_prop_disappear_s
        )

        if hasattr(config, "max_angular_velocity"):
            self.config.max_angular_velocity = config.max_angular_velocity
        if hasattr(config, "v_x_desired"):
            self.config.v_x_desired = config.v_x_desired
        if hasattr(config, "use_imu"):
            self.config.use_imu = config.use_imu
        if hasattr(config, "slalom_mode"):
            self.config.slalom_mode = config.slalom_mode

        rospy.loginfo(
            f"[VS Config] kp={self.config.kp_gain:.2f} kd={self.config.kd_gain:.2f} "
            f"max_w={self.config.max_angular_velocity:.2f}"
        )
        return config

    # -------------------------------------------------------------------------
    # Service Handlers
    # -------------------------------------------------------------------------

    def _handle_start(self, req: VisualServoing) -> VisualServoingResponse:
        """Start the visual servoing process."""
        if self.state != ControllerState.IDLE and req.target_prop == self.target_prop:
            return VisualServoingResponse(
                success=False,
                message="VS Controller is already active for this target.",
            )

        self.target_prop = req.target_prop
        self.target_yaw_in_world = self.current_yaw
        self.state = ControllerState.CENTERING
        self.service_start_time = rospy.Time.now()
        self.last_detection_time = None

        # Reset error states
        self.error_state.reset()
        self.slalom_state.reset()

        rospy.loginfo(f"Visual servoing started for target: {self.target_prop}")
        return VisualServoingResponse(
            success=True, message="Visual servoing activated."
        )

    def _handle_cancel(self, req: Trigger) -> TriggerResponse:
        """Stop the visual servoing process."""
        if self.state == ControllerState.IDLE:
            return TriggerResponse(success=False, message="Controller is not active.")

        self._stop_controller("cancelled by request")
        return TriggerResponse(success=True, message="Visual servoing deactivated.")

    def _handle_navigate(self, req: Trigger) -> TriggerResponse:
        """Switch to navigation mode."""
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

    def _handle_cancel_navigation(self, req: Trigger) -> TriggerResponse:
        """Cancel navigation mode and return to centering."""
        if self.state != ControllerState.NAVIGATING:
            return TriggerResponse(
                success=False, message="Controller is not in navigation mode."
            )

        self.state = ControllerState.CENTERING
        rospy.loginfo("Visual servoing navigation cancelled.")
        return TriggerResponse(success=True, message="Navigation mode deactivated.")

    def _stop_controller(self, reason: str):
        """Stop all motion and reset state."""
        rospy.loginfo(f"Visual servoing stopped: {reason}")
        self.state = ControllerState.IDLE
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(0.1)
        self.control_enable_pub.publish(Bool(data=False))

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def spin(self):
        """The main loop of the controller."""
        rate = rospy.Rate(self.config.rate_hz)
        while not rospy.is_shutdown():
            if self.state == ControllerState.IDLE:
                rate.sleep()
                continue

            elapsed = (rospy.Time.now() - self.service_start_time).to_sec()
            if elapsed > self.config.overall_timeout_s:
                self._stop_controller("overall timeout reached")
                continue

            self.control_enable_pub.publish(Bool(True))
            self._execute_control_step()
            rate.sleep()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        controller = VisualServoingController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass
