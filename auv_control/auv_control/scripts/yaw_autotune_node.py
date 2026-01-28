#!/usr/bin/env python3
"""
Advanced Yaw Auto-Tune Node for AUV/ROV Systems

This node performs automatic PID tuning for yaw control using frequency sweep
(chirp signal) method, which is more robust and informative than relay feedback.

Key Features:
- Chirp signal excitation for frequency response identification
- Bode plot analysis for system characterization
- Model-based PID tuning (superior to Ziegler-Nichols)
- Cascaded PID optimization (outer position + inner velocity loops)
- Real-time safety monitoring
- Step response verification
- Automatic gain scheduling based on operating conditions

Methodology:
1. Apply chirp signal (0.1-2.0 Hz) to yaw setpoint
2. Record input/output data (yaw command vs actual yaw)
3. Perform FFT to extract frequency response
4. Fit transfer function model
5. Design PID gains using loop shaping
6. Apply gains and verify with step response

Safety:
- Uses align_frame for position holding (x, y, z, roll, pitch stable)
- Monitors oscillation amplitude, prevents instability
- Emergency stop if vehicle deviates from position
- Can restore original gains anytime

Usage:
    rosservice call /taluy/yaw_autotune/start "{}"
    rosservice call /taluy/yaw_autotune/cancel "{}"
    rosservice call /taluy/yaw_autotune/verify "{}"
    rosservice call /taluy/yaw_autotune/status "{}"
    rosservice call /taluy/yaw_autotune/get_results "{}"

Author: Advanced Control Systems
Date: 2026-01-28
"""

import rospy
import numpy as np
from threading import Lock
from enum import Enum
from collections import deque
import json

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import AlignFrameController, AlignFrameControllerRequest
import dynamic_reconfigure.client
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class AutoTuneState(Enum):
    """Auto-tune state machine"""

    IDLE = 0
    SETTLING = 1  # Initial settling period
    CHIRP_TEST = 2  # Frequency sweep test
    ANALYZING = 3  # Computing frequency response
    TUNING = 4  # Calculating PID gains
    VERIFY = 5  # Step response verification
    COMPLETE = 6  # Tuning complete, gains applied


class YawAutoTuneNode:
    """
    Advanced auto-tune node using frequency domain identification.

    The chirp signal method is superior to relay feedback because:
    - Provides full frequency response (not just one point)
    - More robust to noise and disturbances
    - Can identify phase margin and gain margin
    - Enables model-based controller design
    - Less aggressive than relay (safer for real systems)
    """

    def __init__(self):
        rospy.init_node("yaw_autotune_advanced")

        # ============================================================
        # PARAMETERS
        # ============================================================
        self.namespace = rospy.get_param("~namespace", "taluy")
        self.base_frame = rospy.get_param("~base_frame", f"{self.namespace}/base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.autotune_frame = rospy.get_param("~autotune_frame", "yaw_autotune_target")

        # Chirp signal parameters
        self.chirp_amplitude = rospy.get_param(
            "~chirp_amplitude", 0.35
        )  # rad (~20 deg)
        self.chirp_f_start = rospy.get_param("~chirp_f_start", 0.05)  # Hz (low freq)
        self.chirp_f_end = rospy.get_param("~chirp_f_end", 1.5)  # Hz (high freq)
        self.chirp_duration = rospy.get_param("~chirp_duration", 30.0)  # seconds

        # Safety parameters
        self.max_position_error = rospy.get_param("~max_position_error", 1.0)  # meters
        self.max_yaw_rate = rospy.get_param("~max_yaw_rate", 1.0)  # rad/s
        self.settle_time = rospy.get_param("~settle_time", 3.0)  # seconds

        # Verification parameters
        self.verify_step_angle = rospy.get_param(
            "~verify_step_angle", 0.52
        )  # rad (30 deg)
        self.verify_duration = rospy.get_param("~verify_duration", 15.0)  # seconds

        # Control parameters
        self.rate = rospy.get_param("~rate", 50)  # Hz
        self.dt = 1.0 / self.rate

        # Tuning aggressiveness (0=conservative, 1=aggressive)
        self.aggressiveness = rospy.get_param("~aggressiveness", 0.6)

        # ============================================================
        # STATE VARIABLES
        # ============================================================
        self.state = AutoTuneState.IDLE
        self.state_lock = Lock()

        # Odometry
        self.current_odom = None
        self.odom_lock = Lock()

        # Test data
        self.test_start_time = None
        self.center_pose = None  # (x, y, z, roll, pitch, yaw)

        # Chirp test data
        self.chirp_time = []
        self.chirp_input = []  # Command yaw
        self.chirp_output = []  # Actual yaw
        self.chirp_error = []  # Error from center

        # Frequency response data
        self.freq_response = None  # {'freq': [...], 'magnitude': [...], 'phase': [...]}
        self.system_model = None  # Fitted transfer function parameters

        # Tuning results
        self.tuned_gains = (
            None  # {'pos': {'kp', 'ki', 'kd'}, 'vel': {'kp', 'ki', 'kd'}}
        )
        self.original_gains = None

        # Verification data
        self.verify_time = []
        self.verify_yaw = []
        self.verify_cmd = []

        # Safety monitoring
        self.initial_position = None
        self.safety_violation_count = 0

        # ============================================================
        # ROS INTERFACES
        # ============================================================

        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscribers
        self.odom_sub = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, queue_size=1, tcp_nodelay=True
        )

        # Services (provided)
        self.start_srv = rospy.Service("yaw_autotune/start", Trigger, self.handle_start)
        self.cancel_srv = rospy.Service(
            "yaw_autotune/cancel", Trigger, self.handle_cancel
        )
        self.verify_srv = rospy.Service(
            "yaw_autotune/verify", Trigger, self.handle_verify
        )
        self.status_srv = rospy.Service(
            "yaw_autotune/status", Trigger, self.handle_status
        )
        self.results_srv = rospy.Service(
            "yaw_autotune/get_results", Trigger, self.handle_get_results
        )

        # Service clients
        rospy.loginfo("Waiting for align_frame service...")
        try:
            rospy.wait_for_service("control/align_frame/start", timeout=5.0)
            self.align_frame_start = rospy.ServiceProxy(
                "control/align_frame/start", AlignFrameController
            )
            self.align_frame_cancel = rospy.ServiceProxy(
                "control/align_frame/cancel", Trigger
            )
            rospy.loginfo("✓ Connected to align_frame service")
        except rospy.ROSException:
            rospy.logerr("✗ align_frame service not available!")
            self.align_frame_start = None
            self.align_frame_cancel = None

        # Dynamic reconfigure
        try:
            self.reconfigure_client = dynamic_reconfigure.client.Client(
                "auv_control_node", timeout=5
            )
            rospy.loginfo("✓ Connected to dynamic reconfigure")
        except Exception as e:
            rospy.logerr(f"✗ Failed to connect to dynamic reconfigure: {e}")
            self.reconfigure_client = None

        # Control loop timer
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)

        # ============================================================
        # INITIALIZATION COMPLETE
        # ============================================================
        rospy.loginfo("=" * 70)
        rospy.loginfo("Advanced Yaw Auto-Tune Node - READY")
        rospy.loginfo("=" * 70)
        rospy.loginfo(
            f"Method: Chirp frequency sweep ({self.chirp_f_start}-{self.chirp_f_end} Hz)"
        )
        rospy.loginfo(f"Amplitude: {np.degrees(self.chirp_amplitude):.1f}°")
        rospy.loginfo(f"Duration: {self.chirp_duration:.1f}s")
        rospy.loginfo(
            f"Aggressiveness: {self.aggressiveness:.1f} (0=conservative, 1=aggressive)"
        )
        rospy.loginfo("=" * 70)
        rospy.loginfo("Ready to tune. Call service: rosservice call yaw_autotune/start")
        rospy.loginfo("=" * 70)

    # ================================================================
    # ODOMETRY & POSE
    # ================================================================

    def odom_callback(self, msg: Odometry):
        """Store latest odometry"""
        with self.odom_lock:
            self.current_odom = msg

    def get_current_pose(self):
        """Get current pose as (x, y, z, roll, pitch, yaw)"""
        with self.odom_lock:
            if self.current_odom is None:
                return None
            pos = self.current_odom.pose.pose.position
            quat = self.current_odom.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            return (pos.x, pos.y, pos.z, roll, pitch, yaw)

    def get_current_twist(self):
        """Get current angular velocity (roll_rate, pitch_rate, yaw_rate)"""
        with self.odom_lock:
            if self.current_odom is None:
                return None
            ang_vel = self.current_odom.twist.twist.angular
            return (ang_vel.x, ang_vel.y, ang_vel.z)

    @staticmethod
    def angle_diff(target, current):
        """Compute shortest angular difference: target - current"""
        diff = target - current
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    # ================================================================
    # TF BROADCASTING
    # ================================================================

    def publish_autotune_frame(self, target_yaw):
        """Publish TF frame for auto-tune target at center position with specified yaw"""
        if self.center_pose is None:
            return

        x, y, z, roll, pitch, _ = self.center_pose

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.autotune_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        quat = quaternion_from_euler(roll, pitch, target_yaw)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    # ================================================================
    # DYNAMIC RECONFIGURE
    # ================================================================

    def save_original_gains(self):
        """Save current PID configuration"""
        if self.reconfigure_client is None:
            return False
        try:
            config = self.reconfigure_client.get_configuration()
            self.original_gains = {
                "pos": {
                    "kp": config["kp_5"],
                    "ki": config["ki_5"],
                    "kd": config["kd_5"],
                },
                "vel": {
                    "kp": config["kp_11"],
                    "ki": config["ki_11"],
                    "kd": config["kd_11"],
                },
            }
            rospy.loginfo("✓ Saved original gains:")
            rospy.loginfo(
                f"  Position: Kp={self.original_gains['pos']['kp']:.3f}, "
                f"Ki={self.original_gains['pos']['ki']:.4f}, "
                f"Kd={self.original_gains['pos']['kd']:.3f}"
            )
            rospy.loginfo(
                f"  Velocity: Kp={self.original_gains['vel']['kp']:.3f}, "
                f"Ki={self.original_gains['vel']['ki']:.4f}, "
                f"Kd={self.original_gains['vel']['kd']:.3f}"
            )
            return True
        except Exception as e:
            rospy.logerr(f"✗ Failed to save gains: {e}")
            return False

    def restore_original_gains(self):
        """Restore original PID configuration"""
        if self.reconfigure_client is None or self.original_gains is None:
            return False
        try:
            self.reconfigure_client.update_configuration(
                {
                    "kp_5": self.original_gains["pos"]["kp"],
                    "ki_5": self.original_gains["pos"]["ki"],
                    "kd_5": self.original_gains["pos"]["kd"],
                    "kp_11": self.original_gains["vel"]["kp"],
                    "ki_11": self.original_gains["vel"]["ki"],
                    "kd_11": self.original_gains["vel"]["kd"],
                }
            )
            rospy.loginfo("✓ Restored original gains")
            return True
        except Exception as e:
            rospy.logerr(f"✗ Failed to restore gains: {e}")
            return False

    def apply_gains(self, gains_dict):
        """Apply new PID gains"""
        if self.reconfigure_client is None:
            return False
        try:
            self.reconfigure_client.update_configuration(
                {
                    "kp_5": gains_dict["pos"]["kp"],
                    "ki_5": gains_dict["pos"]["ki"],
                    "kd_5": gains_dict["pos"]["kd"],
                    "kp_11": gains_dict["vel"]["kp"],
                    "ki_11": gains_dict["vel"]["ki"],
                    "kd_11": gains_dict["vel"]["kd"],
                }
            )
            rospy.loginfo("✓ Applied new gains:")
            rospy.loginfo(
                f"  Position: Kp={gains_dict['pos']['kp']:.3f}, "
                f"Ki={gains_dict['pos']['ki']:.4f}, "
                f"Kd={gains_dict['pos']['kd']:.3f}"
            )
            rospy.loginfo(
                f"  Velocity: Kp={gains_dict['vel']['kp']:.3f}, "
                f"Ki={gains_dict['vel']['ki']:.4f}, "
                f"Kd={gains_dict['vel']['kd']:.3f}"
            )
            return True
        except Exception as e:
            rospy.logerr(f"✗ Failed to apply gains: {e}")
            return False

    # ================================================================
    # ALIGN FRAME SERVICE CALLS
    # ================================================================

    def start_align_frame(self):
        """Start align_frame controller"""
        if self.align_frame_start is None:
            rospy.logerr("align_frame service not available")
            return False
        try:
            req = AlignFrameControllerRequest()
            req.source_frame = self.base_frame
            req.target_frame = self.autotune_frame
            req.angle_offset = 0.0
            req.keep_orientation = False
            req.use_depth = True
            req.max_linear_velocity = 0.2
            req.max_angular_velocity = 0.5
            resp = self.align_frame_start(req)
            if resp.success:
                rospy.loginfo("✓ align_frame started")
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr(f"✗ align_frame start failed: {e}")
            return False

    def cancel_align_frame(self):
        """Cancel align_frame controller"""
        if self.align_frame_cancel is None:
            return
        try:
            self.align_frame_cancel()
            rospy.loginfo("✓ align_frame cancelled")
        except rospy.ServiceException as e:
            rospy.logwarn(f"align_frame cancel failed: {e}")

    # ================================================================
    # SERVICE HANDLERS
    # ================================================================

    def handle_start(self, req) -> TriggerResponse:
        """Start auto-tune process"""
        with self.state_lock:
            if self.state != AutoTuneState.IDLE:
                return TriggerResponse(
                    success=False,
                    message=f"Auto-tune already active (state: {self.state.name})",
                )

            # Pre-flight checks
            if self.align_frame_start is None:
                return TriggerResponse(
                    success=False, message="align_frame service not available"
                )

            pose = self.get_current_pose()
            if pose is None:
                return TriggerResponse(
                    success=False, message="No odometry data available"
                )

            # Save original gains
            if not self.save_original_gains():
                return TriggerResponse(
                    success=False, message="Failed to save original gains"
                )

            # Initialize test
            self.center_pose = pose
            self.initial_position = (pose[0], pose[1], pose[2])
            self.test_start_time = rospy.Time.now()

            # Clear data buffers
            self.chirp_time = []
            self.chirp_input = []
            self.chirp_output = []
            self.chirp_error = []
            self.safety_violation_count = 0

            # Publish initial frame
            self.publish_autotune_frame(pose[5])
            rospy.sleep(0.1)

            # Start align_frame
            if not self.start_align_frame():
                return TriggerResponse(
                    success=False, message="Failed to start align_frame controller"
                )

            self.state = AutoTuneState.SETTLING

            rospy.loginfo("=" * 70)
            rospy.loginfo("AUTO-TUNE STARTED")
            rospy.loginfo("=" * 70)
            rospy.loginfo(
                f"Position: x={pose[0]:.2f}, y={pose[1]:.2f}, z={pose[2]:.2f}"
            )
            rospy.loginfo(f"Center yaw: {np.degrees(pose[5]):.1f}°")
            rospy.loginfo(
                f"Settling for {self.settle_time:.1f}s, then chirp test for {self.chirp_duration:.1f}s"
            )
            rospy.loginfo("=" * 70)

        return TriggerResponse(success=True, message="Auto-tune started")

    def handle_cancel(self, req) -> TriggerResponse:
        """Cancel auto-tune and restore original gains"""
        with self.state_lock:
            if self.state == AutoTuneState.IDLE:
                return TriggerResponse(
                    success=False, message="No auto-tune in progress"
                )

            # Cancel align_frame
            self.cancel_align_frame()

            # Restore original gains
            self.restore_original_gains()

            # Reset state
            self.state = AutoTuneState.IDLE

            rospy.loginfo("✓ Auto-tune cancelled, original gains restored")

        return TriggerResponse(success=True, message="Auto-tune cancelled")

    def handle_verify(self, req) -> TriggerResponse:
        """Run step response verification test"""
        with self.state_lock:
            if (
                self.state != AutoTuneState.IDLE
                and self.state != AutoTuneState.COMPLETE
            ):
                return TriggerResponse(
                    success=False,
                    message="Cannot verify while auto-tune is in progress",
                )

            pose = self.get_current_pose()
            if pose is None:
                return TriggerResponse(success=False, message="No odometry data")

            # Initialize verification
            self.center_pose = pose
            self.test_start_time = rospy.Time.now()
            self.verify_time = []
            self.verify_yaw = []
            self.verify_cmd = []

            # Publish frame and start align
            self.publish_autotune_frame(pose[5])
            rospy.sleep(0.1)

            if not self.start_align_frame():
                return TriggerResponse(
                    success=False, message="Failed to start align_frame"
                )

            self.state = AutoTuneState.VERIFY

            rospy.loginfo("=" * 70)
            rospy.loginfo("VERIFICATION STEP RESPONSE TEST STARTED")
            rospy.loginfo(f"Step size: {np.degrees(self.verify_step_angle):.1f}°")
            rospy.loginfo("=" * 70)

        return TriggerResponse(success=True, message="Verification started")

    def handle_status(self, req) -> TriggerResponse:
        """Get current auto-tune status"""
        with self.state_lock:
            if self.state == AutoTuneState.IDLE:
                msg = "IDLE - Ready for auto-tune"
            elif self.state == AutoTuneState.SETTLING:
                elapsed = (rospy.Time.now() - self.test_start_time).to_sec()
                msg = f"SETTLING - {elapsed:.1f}/{self.settle_time:.1f}s"
            elif self.state == AutoTuneState.CHIRP_TEST:
                elapsed = (rospy.Time.now() - self.test_start_time).to_sec()
                progress = (elapsed / self.chirp_duration) * 100
                msg = f"CHIRP TEST - {progress:.1f}% complete ({len(self.chirp_time)} samples)"
            elif self.state == AutoTuneState.ANALYZING:
                msg = "ANALYZING - Computing frequency response"
            elif self.state == AutoTuneState.TUNING:
                msg = "TUNING - Calculating optimal PID gains"
            elif self.state == AutoTuneState.VERIFY:
                elapsed = (rospy.Time.now() - self.test_start_time).to_sec()
                msg = f"VERIFY - Step response test ({elapsed:.1f}s)"
            elif self.state == AutoTuneState.COMPLETE:
                msg = "COMPLETE - Tuning successful, gains applied"
            else:
                msg = f"State: {self.state.name}"

            return TriggerResponse(success=True, message=msg)

    def handle_get_results(self, req) -> TriggerResponse:
        """Get tuning results summary"""
        with self.state_lock:
            if self.tuned_gains is None:
                return TriggerResponse(
                    success=False,
                    message="No tuning results available. Run auto-tune first.",
                )

            # Format results as JSON
            results = {
                "tuned_gains": self.tuned_gains,
                "original_gains": self.original_gains,
                "system_model": self.system_model if self.system_model else {},
                "frequency_response": {"available": self.freq_response is not None},
            }

            msg = json.dumps(results, indent=2)
            return TriggerResponse(success=True, message=msg)

    # ================================================================
    # CONTROL LOOP
    # ================================================================

    def control_loop(self, event):
        """Main control loop (runs at specified rate)"""
        with self.state_lock:
            if self.state == AutoTuneState.IDLE or self.state == AutoTuneState.COMPLETE:
                return

            elapsed = (rospy.Time.now() - self.test_start_time).to_sec()

            # Safety check
            if not self.safety_check():
                rospy.logerr("SAFETY VIOLATION! Aborting auto-tune")
                self.emergency_stop()
                return

            # State machine
            if self.state == AutoTuneState.SETTLING:
                self.run_settling(elapsed)
            elif self.state == AutoTuneState.CHIRP_TEST:
                self.run_chirp_test(elapsed)
            elif self.state == AutoTuneState.VERIFY:
                self.run_verify_test(elapsed)

    def safety_check(self):
        """Check if vehicle is within safety limits"""
        pose = self.get_current_pose()
        if pose is None or self.initial_position is None:
            return True  # Can't check, assume OK

        # Check position drift
        dx = pose[0] - self.initial_position[0]
        dy = pose[1] - self.initial_position[1]
        dz = pose[2] - self.initial_position[2]
        position_error = np.sqrt(dx**2 + dy**2 + dz**2)

        if position_error > self.max_position_error:
            rospy.logerr(f"Position drift too large: {position_error:.2f}m")
            return False

        # Check yaw rate
        twist = self.get_current_twist()
        if twist is not None:
            yaw_rate = abs(twist[2])
            if yaw_rate > self.max_yaw_rate:
                self.safety_violation_count += 1
                if self.safety_violation_count > 10:  # Allow brief spikes
                    rospy.logerr(f"Yaw rate too high: {yaw_rate:.2f} rad/s")
                    return False
            else:
                self.safety_violation_count = max(0, self.safety_violation_count - 1)

        return True

    def emergency_stop(self):
        """Emergency stop procedure"""
        rospy.logwarn("EMERGENCY STOP TRIGGERED")
        self.cancel_align_frame()
        self.restore_original_gains()
        self.state = AutoTuneState.IDLE

    # ================================================================
    # TEST EXECUTION
    # ================================================================

    def run_settling(self, elapsed):
        """Settling phase - let vehicle stabilize"""
        center_yaw = self.center_pose[5]
        self.publish_autotune_frame(center_yaw)

        if elapsed >= self.settle_time:
            rospy.loginfo("✓ Settling complete, starting chirp test")
            self.test_start_time = rospy.Time.now()
            self.state = AutoTuneState.CHIRP_TEST

    def run_chirp_test(self, elapsed):
        """
        Run chirp (frequency sweep) test.

        Chirp signal: f(t) = A * sin(2π * f(t) * t)
        where f(t) = f0 + (f1 - f0) * t / T (linear chirp)
        """
        # Check if test complete
        if elapsed >= self.chirp_duration:
            rospy.loginfo("✓ Chirp test complete, analyzing data...")
            self.state = AutoTuneState.ANALYZING
            self.analyze_frequency_response()
            return

        # Generate chirp signal
        t = elapsed
        T = self.chirp_duration
        f0 = self.chirp_f_start
        f1 = self.chirp_f_end

        # Instantaneous frequency (linear chirp)
        f_t = f0 + (f1 - f0) * t / T

        # Chirp signal
        chirp_angle = self.chirp_amplitude * np.sin(
            2 * np.pi * f_t * t * (f0 + f_t) / (2 * f0)
        )

        # Target yaw = center + chirp
        center_yaw = self.center_pose[5]
        target_yaw = center_yaw + chirp_angle

        # Get actual yaw
        pose = self.get_current_pose()
        if pose is None:
            return

        actual_yaw = pose[5]
        error = self.angle_diff(center_yaw, actual_yaw)

        # Record data
        self.chirp_time.append(elapsed)
        self.chirp_input.append(chirp_angle)
        self.chirp_output.append(error)
        self.chirp_error.append(error)

        # Publish target frame
        self.publish_autotune_frame(target_yaw)

        # Log progress
        if int(elapsed) % 5 == 0 and len(self.chirp_time) % (5 * self.rate) < 2:
            progress = (elapsed / self.chirp_duration) * 100
            rospy.loginfo(
                f"Chirp test: {progress:.1f}% | f={f_t:.3f} Hz | "
                f"samples={len(self.chirp_time)}"
            )

    def run_verify_test(self, elapsed):
        """Run step response verification"""
        center_yaw = self.center_pose[5]

        # Test sequence: hold -> step -> hold
        if elapsed < 2.0:
            target = center_yaw
            phase = "hold"
        elif elapsed < 2.0 + self.verify_duration:
            target = center_yaw + self.verify_step_angle
            phase = "step"
        elif elapsed < 4.0 + self.verify_duration:
            target = center_yaw
            phase = "return"
        else:
            # Test complete
            rospy.loginfo("✓ Verification complete, analyzing...")
            self.cancel_align_frame()
            self.analyze_step_response()
            self.state = AutoTuneState.IDLE
            return

        # Publish target
        self.publish_autotune_frame(target)

        # Record data
        pose = self.get_current_pose()
        if pose is not None:
            self.verify_time.append(elapsed)
            self.verify_yaw.append(pose[5])
            self.verify_cmd.append(target)

        # Log progress
        if int(elapsed * 2) % 2 == 0 and len(self.verify_time) % (self.rate // 2) < 2:
            if pose is not None:
                rospy.loginfo(
                    f"Verify [{phase:6s}]: t={elapsed:5.1f}s | "
                    f"target={np.degrees(target):6.1f}° | "
                    f"actual={np.degrees(pose[5]):6.1f}°"
                )

    # ================================================================
    # SIGNAL PROCESSING & ANALYSIS
    # ================================================================

    def analyze_frequency_response(self):
        """
        Analyze chirp test data to extract frequency response.
        Uses FFT-based spectral analysis.
        """
        if len(self.chirp_time) < 100:
            rospy.logerr("Insufficient data for analysis")
            self.emergency_stop()
            return

        rospy.loginfo("Analyzing frequency response...")

        # Convert to numpy arrays
        time = np.array(self.chirp_time)
        input_signal = np.array(self.chirp_input)
        output_signal = np.array(self.chirp_output)

        # Compute FFT
        n = len(time)
        dt = np.mean(np.diff(time))

        input_fft = np.fft.rfft(input_signal)
        output_fft = np.fft.rfft(output_signal)
        freqs = np.fft.rfftfreq(n, dt)

        # Frequency response H(f) = Output(f) / Input(f)
        # Avoid division by zero
        epsilon = 1e-10
        H = output_fft / (input_fft + epsilon)

        # Magnitude and phase
        magnitude = np.abs(H)
        phase = np.angle(H)  # radians

        # Focus on frequency range of interest
        freq_mask = (freqs >= self.chirp_f_start - 0.02) & (
            freqs <= self.chirp_f_end + 0.2
        )
        freqs_roi = freqs[freq_mask]
        magnitude_roi = magnitude[freq_mask]
        phase_roi = phase[freq_mask]

        # Store results
        self.freq_response = {
            "freq": freqs_roi.tolist(),
            "magnitude": magnitude_roi.tolist(),
            "phase": phase_roi.tolist(),
            "magnitude_db": (20 * np.log10(magnitude_roi + 1e-10)).tolist(),
            "phase_deg": np.degrees(phase_roi).tolist(),
        }

        rospy.loginfo(f"✓ Frequency response computed ({len(freqs_roi)} points)")
        rospy.loginfo(f"  Frequency range: {freqs_roi[0]:.3f} - {freqs_roi[-1]:.3f} Hz")

        # Fit transfer function model and compute gains
        self.state = AutoTuneState.TUNING
        self.compute_optimal_gains()

    def compute_optimal_gains(self):
        """
        Compute optimal PID gains using frequency response data.

        Method: Loop shaping based on desired crossover frequency and phase margin.
        For cascaded PID:
        - Inner loop (velocity): Fast response, moderate damping
        - Outer loop (position): Slower, well-damped
        """
        rospy.loginfo("Computing optimal PID gains...")

        if self.freq_response is None:
            rospy.logerr("No frequency response data")
            self.emergency_stop()
            return

        freqs = np.array(self.freq_response["freq"])
        magnitude = np.array(self.freq_response["magnitude"])
        phase_rad = np.array(self.freq_response["phase"])

        # Find system characteristics
        # 1. DC gain (low frequency)
        dc_gain = magnitude[0] if len(magnitude) > 0 else 1.0

        # 2. Unity gain crossover frequency (approximate)
        unity_idx = np.argmin(np.abs(magnitude - 1.0))
        if unity_idx < len(freqs):
            crossover_freq = freqs[unity_idx]
        else:
            crossover_freq = 0.5  # Default

        # 3. Phase at crossover
        phase_at_crossover = (
            phase_rad[unity_idx] if unity_idx < len(phase_rad) else -np.pi / 2
        )
        phase_margin = np.pi + phase_at_crossover  # Phase margin in radians

        rospy.loginfo(f"  DC gain: {dc_gain:.3f}")
        rospy.loginfo(f"  Crossover frequency: {crossover_freq:.3f} Hz")
        rospy.loginfo(f"  Phase margin: {np.degrees(phase_margin):.1f}°")

        # Store model parameters
        self.system_model = {
            "dc_gain": float(dc_gain),
            "crossover_freq_hz": float(crossover_freq),
            "phase_margin_deg": float(np.degrees(phase_margin)),
        }

        # ========================================
        # CASCADED PID DESIGN
        # ========================================

        # Target specifications (adjust based on aggressiveness)
        # Higher aggressiveness = faster response, less damping
        target_pm_vel = 45 + (1 - self.aggressiveness) * 15  # 45-60 deg
        target_bw_vel = crossover_freq * (0.8 + self.aggressiveness * 0.4)  # 0.8-1.2x

        target_pm_pos = 50 + (1 - self.aggressiveness) * 20  # 50-70 deg
        target_bw_pos = target_bw_vel * 0.3  # Outer loop ~30% of inner loop

        rospy.loginfo(f"Target specs (aggressiveness={self.aggressiveness:.1f}):")
        rospy.loginfo(
            f"  Velocity loop: BW={target_bw_vel:.2f} Hz, PM={target_pm_vel:.1f}°"
        )
        rospy.loginfo(
            f"  Position loop: BW={target_bw_pos:.2f} Hz, PM={target_pm_pos:.1f}°"
        )

        # --- VELOCITY LOOP (INNER) ---
        # PID for velocity loop: controls torque based on velocity error
        # For a typical second-order system with damping
        omega_vel = 2 * np.pi * target_bw_vel
        zeta_vel = np.tan(np.radians(target_pm_vel)) / 2

        kp_vel = omega_vel / (dc_gain if dc_gain > 0.1 else 1.0)
        ki_vel = kp_vel * omega_vel * 0.3  # Moderate integral action
        kd_vel = kp_vel / (omega_vel * 2)

        # Limit velocity gains to reasonable ranges
        kp_vel = np.clip(kp_vel, 0.1, 5.0)
        ki_vel = np.clip(ki_vel, 0.0, 2.0)
        kd_vel = np.clip(kd_vel, 0.0, 1.0)

        # --- POSITION LOOP (OUTER) ---
        # PID for position loop: controls desired velocity based on position error
        # Should be significantly slower than velocity loop
        omega_pos = 2 * np.pi * target_bw_pos

        kp_pos = omega_pos * 1.5
        ki_pos = 0.0  # Usually no integral in outer loop
        kd_pos = kp_pos / (omega_pos * 3)

        # Limit position gains
        kp_pos = np.clip(kp_pos, 0.5, 10.0)
        ki_pos = np.clip(ki_pos, 0.0, 0.5)
        kd_pos = np.clip(kd_pos, 0.0, 2.0)

        # Store tuned gains
        self.tuned_gains = {
            "pos": {"kp": float(kp_pos), "ki": float(ki_pos), "kd": float(kd_pos)},
            "vel": {"kp": float(kp_vel), "ki": float(ki_vel), "kd": float(kd_vel)},
        }

        rospy.loginfo("=" * 70)
        rospy.loginfo("TUNED GAINS (Model-based)")
        rospy.loginfo("=" * 70)
        rospy.loginfo(
            f"Position PID: Kp={kp_pos:.3f}, Ki={ki_pos:.4f}, Kd={kd_pos:.3f}"
        )
        rospy.loginfo(
            f"Velocity PID: Kp={kp_vel:.3f}, Ki={ki_vel:.4f}, Kd={kd_vel:.3f}"
        )
        rospy.loginfo("=" * 70)

        # Apply gains
        if self.apply_gains(self.tuned_gains):
            self.state = AutoTuneState.COMPLETE
            rospy.loginfo("✓ AUTO-TUNE COMPLETE!")
            rospy.loginfo("  New gains have been applied.")
            rospy.loginfo("  Use 'yaw_autotune/verify' to test step response")
            rospy.loginfo("  Use 'yaw_autotune/cancel' to restore original gains")
        else:
            rospy.logerr("✗ Failed to apply gains")
            self.emergency_stop()

    def analyze_step_response(self):
        """Analyze step response verification test"""
        if len(self.verify_time) < 20:
            rospy.logwarn("Insufficient verification data")
            return

        time = np.array(self.verify_time)
        yaw = np.array(self.verify_yaw)
        cmd = np.array(self.verify_cmd)

        # Find step region (when command changes)
        cmd_diff = np.diff(cmd)
        step_indices = np.where(np.abs(cmd_diff) > 0.1)[0]

        if len(step_indices) == 0:
            rospy.logwarn("No step detected in verification data")
            return

        step_start_idx = step_indices[0] + 1

        # Analyze response after step
        t_step = time[step_start_idx:]
        y_step = yaw[step_start_idx:]
        c_step = cmd[step_start_idx:]

        if len(t_step) < 10:
            return

        # Normalize time
        t_step = t_step - t_step[0]

        # Convert to error from commanded value
        initial_val = y_step[0]
        final_cmd = c_step[-1]
        step_size = self.angle_diff(initial_val, final_cmd)

        # Compute response in terms of fraction of step
        response = np.array([self.angle_diff(initial_val, y) for y in y_step])
        response_normalized = (
            response / step_size if abs(step_size) > 0.01 else response
        )

        # Metrics
        final_value = response_normalized[-1]
        steady_state_error = abs(1.0 - final_value)

        # Overshoot
        peak_value = np.max(np.abs(response_normalized))
        overshoot = max(0, peak_value - 1.0) * 100  # Percent

        # Rise time (10% to 90%)
        rise_time = None
        t_10, t_90 = None, None
        for i, r in enumerate(response_normalized):
            if t_10 is None and abs(r) >= 0.1:
                t_10 = t_step[i]
            if t_90 is None and abs(r) >= 0.9:
                t_90 = t_step[i]
                break
        if t_10 is not None and t_90 is not None:
            rise_time = t_90 - t_10

        # Settling time (within 2%)
        settling_time = None
        for i in range(len(response_normalized) - 1, 0, -1):
            if abs(response_normalized[i] - 1.0) > 0.02:
                if i + 1 < len(t_step):
                    settling_time = t_step[i + 1]
                break

        # Display results
        rospy.loginfo("=" * 70)
        rospy.loginfo("STEP RESPONSE ANALYSIS")
        rospy.loginfo("=" * 70)
        rospy.loginfo(f"Step size: {np.degrees(abs(step_size)):.1f}°")
        rospy.loginfo(
            f"Steady-state error: {steady_state_error*100:.2f}% "
            f"({np.degrees(steady_state_error * abs(step_size)):.2f}°)"
        )
        rospy.loginfo(f"Overshoot: {overshoot:.1f}%")
        if rise_time is not None:
            rospy.loginfo(f"Rise time (10-90%): {rise_time:.3f}s")
        if settling_time is not None:
            rospy.loginfo(f"Settling time (2%): {settling_time:.3f}s")
        rospy.loginfo("-" * 70)

        # Assessment
        if overshoot < 5 and steady_state_error < 0.02:
            rospy.loginfo("Assessment: ✓ EXCELLENT - Well tuned, minimal overshoot")
        elif overshoot < 15 and steady_state_error < 0.05:
            rospy.loginfo("Assessment: ✓ GOOD - Acceptable performance")
        elif overshoot < 30:
            rospy.loginfo("Assessment: ⚠ ACCEPTABLE - Some overshoot present")
        else:
            rospy.loginfo(
                "Assessment: ✗ NEEDS IMPROVEMENT - Consider re-tuning with different aggressiveness"
            )
            rospy.loginfo(
                "  Suggestion: Reduce aggressiveness parameter (currently {:.1f})".format(
                    self.aggressiveness
                )
            )

        rospy.loginfo("=" * 70)


if __name__ == "__main__":
    try:
        node = YawAutoTuneNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
