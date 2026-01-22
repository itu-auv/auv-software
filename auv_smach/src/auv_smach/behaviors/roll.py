import py_trees
import rospy
import math
import tf2_ros
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from .actions import (
    SetBoolServiceBehavior,
    DelayBehavior,
    SetDepthBehavior,
    CancelAlignControllerBehavior,
    ResetOdometryBehavior,
    ClearObjectMapBehavior,
    SetDetectionFocusBehavior,
    RotateBehavior,
)
from .subtrees import create_search_subtree


class PitchCorrectionBehavior(py_trees.behaviour.Behaviour):
    """
    Corrects the vehicle's pitch using a fixed torque until it matches the criteria.
    Subscribes to odometry to monitor pitch and publishes wrench commands.
    Checks killswitch status to abort if motors are disabled.
    """

    def __init__(
        self,
        name: str,
        fixed_torque: float = 2.0,
        timeout: float = 5.0,
    ):
        super().__init__(name)
        self.fixed_torque = -abs(fixed_torque)  # Always negative logic from roll.py
        self.timeout = timeout

        self.odometry_topic = "odometry"
        self.killswitch_topic = "propulsion_board/status"
        self.wrench_topic = "wrench"
        self.frame_id = "taluy/base_link"

        # Runtime State
        self._current_pitch = 0.0
        self._odom_received = False
        self._killswitch_active = False
        self._start_time = None

    def setup(self, timeout=15, **kwargs):
        """Setup ROS connections."""
        try:
            self.sub_odom = rospy.Subscriber(
                self.odometry_topic, Odometry, self._odom_callback
            )
            self.sub_kill = rospy.Subscriber(
                self.killswitch_topic, Bool, self._killswitch_callback
            )
            self.pub_wrench = rospy.Publisher(
                self.wrench_topic, WrenchStamped, queue_size=1
            )
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {str(e)}")
            return False

    def initialise(self):
        """Reset state before execution."""
        self._current_pitch = 0.0
        self._odom_received = False
        self._killswitch_active = False
        self._start_time = rospy.Time.now()
        rospy.loginfo(f"[{self.name}] Initialized. Waiting for odometry...")

    def update(self):
        """Execute the behavior."""
        # 0. Check Killswitch
        if self._killswitch_active:
            rospy.logerr(f"[{self.name}] Aborted: Killswitch is active")
            self._stop_and_publish_zero()
            return py_trees.common.Status.FAILURE

        # 1. Wait for Odometry
        if not self._odom_received:
            elapsed = (rospy.Time.now() - self._start_time).to_sec()
            if elapsed > 5.0:
                rospy.logerr(f"[{self.name}] Timeout waiting for odometry")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        # 2. Check if already corrected (Success Condition)
        if self._current_pitch <= 0:
            rospy.loginfo(
                f"[{self.name}] Success: Pitch corrected ({math.degrees(self._current_pitch):.2f} deg)"
            )
            self._stop_and_publish_zero()
            return py_trees.common.Status.SUCCESS

        # 3. Check Timeout
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed > self.timeout:
            rospy.logwarn(
                f"[{self.name}] Timeout reached ({elapsed:.1f}s). Assuming success/continuing."
            )
            self._stop_and_publish_zero()
            return py_trees.common.Status.SUCCESS

        # 4. Apply Torque
        cmd = WrenchStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.frame_id
        cmd.wrench.torque.y = self.fixed_torque
        self.pub_wrench.publish(cmd)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Cleanup."""
        if new_status != py_trees.common.Status.RUNNING:
            self._stop_and_publish_zero()

    def _stop_and_publish_zero(self):
        """Publish zero torque to stop rotation."""
        stop_cmd = WrenchStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.header.frame_id = self.frame_id
        stop_cmd.wrench.torque.y = 0.0
        if hasattr(self, "pub_wrench"):
            self.pub_wrench.publish(stop_cmd)

    def _odom_callback(self, msg: Odometry):
        orientation = msg.pose.pose.orientation
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        try:
            _, pitch, _ = euler_from_quaternion(q)
            self._current_pitch = pitch
            self._odom_received = True
        except Exception:
            pass

    def _killswitch_callback(self, msg: Bool):
        if not msg.data:
            self._killswitch_active = True


class RollBehavior(py_trees.behaviour.Behaviour):
    """
    Rolls the vehicle by applying torque until a total rotation angle is achieved.
    Uses angular velocity from odometry to estimate total rotation.
    """

    def __init__(
        self,
        name: str,
        roll_torque: float = 50.0,
        target_angle: float = math.radians(660.0),
        timeout: float = 15.0,
    ):
        super().__init__(name)
        self.roll_torque = roll_torque
        self.target_angle = target_angle
        self.timeout = timeout

        self.wrench_topic = "wrench"
        self.odometry_topic = "odometry"
        self.killswitch_topic = "propulsion_board/status"
        self.frame_id = "taluy/base_link"

        self._total_roll = 0.0
        self._last_time = None
        self._killswitch_active = False
        self._start_time = None
        self._odom_received = False

    def setup(self, timeout=15, **kwargs):
        """Setup ROS connections."""
        try:
            self.sub_odom = rospy.Subscriber(
                self.odometry_topic, Odometry, self._odom_callback
            )
            self.sub_kill = rospy.Subscriber(
                self.killswitch_topic, Bool, self._killswitch_callback
            )
            self.pub_wrench = rospy.Publisher(
                self.wrench_topic, WrenchStamped, queue_size=1
            )
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {str(e)}")
            return False

    def initialise(self):
        """Reset state before execution."""
        self._total_roll = 0.0
        self._last_time = rospy.Time.now()
        self._killswitch_active = False
        self._start_time = rospy.Time.now()
        self._odom_received = False
        rospy.loginfo(
            f"[{self.name}] Initialized. Target Roll: {math.degrees(self.target_angle):.1f} deg"
        )

    def update(self):
        """Execute the behavior."""
        # 0. Check Killswitch
        if self._killswitch_active:
            rospy.logerr(f"[{self.name}] Aborted: Killswitch is active")
            self._stop_and_publish_zero()
            return py_trees.common.Status.FAILURE

        # 1. Wait for Odometry
        if not self._odom_received:
            elapsed = (rospy.Time.now() - self._start_time).to_sec()
            if elapsed > 5.0:
                rospy.logerr(f"[{self.name}] Timeout waiting for odometry")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        # 2. Check Success
        if self._total_roll >= self.target_angle:
            rospy.loginfo(
                f"[{self.name}] Success: Rolled {math.degrees(self._total_roll):.1f} deg"
            )
            self._stop_and_publish_zero()
            return py_trees.common.Status.SUCCESS

        # 3. Check Timeout
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed > self.timeout:
            rospy.logerr(
                f"[{self.name}] Timeout reached. Rolled only {math.degrees(self._total_roll):.1f} deg"
            )
            self._stop_and_publish_zero()
            return py_trees.common.Status.FAILURE

        # 4. Apply Torque & Log
        cmd = WrenchStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.frame_id
        cmd.wrench.torque.x = self.roll_torque
        self.pub_wrench.publish(cmd)

        rospy.loginfo_throttle(
            1.0,
            f"[{self.name}] Rolling: {math.degrees(self._total_roll):.1f}/{math.degrees(self.target_angle):.0f} deg",
        )

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if new_status != py_trees.common.Status.RUNNING:
            self._stop_and_publish_zero()

    def _stop_and_publish_zero(self):
        stop_cmd = WrenchStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.header.frame_id = self.frame_id
        if hasattr(self, "pub_wrench"):
            self.pub_wrench.publish(stop_cmd)

    def _odom_callback(self, msg: Odometry):
        now = rospy.Time.now()
        self._odom_received = True

        if self._last_time is None:
            self._last_time = now
            return

        dt = (now - self._last_time).to_sec()
        self._last_time = now

        # Calculate roll change
        omega_x = msg.twist.twist.angular.x
        delta_angle = omega_x * dt
        self._total_roll += abs(delta_angle)

    def _killswitch_callback(self, msg: Bool):
        if not msg.data:
            self._killswitch_active = True


def create_roll_subtree(name: str, gate_look_at_frame: str, roll_torque: float = 50.0):
    """
    Creates the 'California Roll' maneuver subtree.
    SMACH equivalent: TwoRollState (roll.py)
    """
    root = py_trees.composites.Sequence(name=name, memory=True)

    # 1. Disable DVL Odom
    root.add_child(
        SetBoolServiceBehavior(
            name="DisableDvlOdom", service_name="dvl_to_odom_node/enable", value=False
        )
    )

    # 2. Wait
    root.add_child(DelayBehavior("WaitForPitchCorrection", duration=3.0))

    # 3. Pitch Correction
    root.add_child(
        PitchCorrectionBehavior(name="PitchCorrection", fixed_torque=3.0, timeout=3.0)
    )

    # 4. Roll (Two Times - 660 deg in SMACH)
    root.add_child(
        RollBehavior(
            name="RollTwoTimes",
            roll_torque=roll_torque,
            target_angle=math.radians(660.0),
            timeout=15.0,
        )
    )

    # 5. Wait for Stabilization
    root.add_child(DelayBehavior("WaitForStabilization", duration=2.0))

    # 6. Enable DVL Odom
    root.add_child(
        SetBoolServiceBehavior(
            name="EnableDvlOdom", service_name="dvl_to_odom_node/enable", value=True
        )
    )

    # 7. Wait for DVL enable
    root.add_child(DelayBehavior("WaitForDvlOdomEnable", duration=3.0))

    # 8. Set Roll Depth (Set to -0.8 in SMACH? No, roll.py uses SetDepthState(depth=-0.8))
    root.add_child(SetDepthBehavior("SetRollDepth", depth=-0.8, sleep_duration=5.0))

    # 9. Align to Look at Gate (create_search_subtree)
    # SMACH uses SearchForPropState with full_rotation=True, duration=7.0, speed=0.25
    root.add_child(
        create_search_subtree(
            name="AlignToLookAtGateAfterRoll",
            source_frame="taluy/base_link",
            look_at_frame=gate_look_at_frame,
            alignment_frame="gate_search_after_roll",
            rotation_speed=0.25,
            duration=7.0,
            timeout=30.0,  # Standard
        )
    )

    # 10. Delay After Alignment
    root.add_child(DelayBehavior("DelayAfterAlignment", duration=1.0))

    # 11. Cancel Align Controller
    root.add_child(CancelAlignControllerBehavior("CancelAlignAfterRoll"))

    # 12. Reset Odometry
    root.add_child(ResetOdometryBehavior("ResetOdometryAfterRoll"))

    # 13. Delay After Reset
    root.add_child(DelayBehavior("DelayAfterReset", duration=1.0))

    # 14. Clear Object Map
    root.add_child(ClearObjectMapBehavior("ClearObjectMapAfterRoll"))

    return root


def create_yaw_subtree(name: str, yaw_frame: str):
    """
    Creates the 'Two Yaw' maneuver subtree.
    SMACH equivalent: TwoYawState (roll.py)
    """
    root = py_trees.composites.Sequence(name=name, memory=True)

    # 1. Focus None
    root.add_child(SetDetectionFocusBehavior("FocusNone", focus_object="none"))

    # 2. First 360 Spin
    root.add_child(
        RotateBehavior(
            name="YawFirst360",
            source_frame="taluy/base_link",
            look_at_frame=yaw_frame,
            rotation_speed=0.5,
            rotation_radian=2 * math.pi,
            timeout=45.0,
            full_rotation=True,
        )
    )

    # 3. Second 360 Spin
    root.add_child(
        RotateBehavior(
            name="YawSecond360",
            source_frame="taluy/base_link",
            look_at_frame=yaw_frame,
            rotation_speed=0.5,
            rotation_radian=2 * math.pi,
            timeout=45.0,
            full_rotation=True,
        )
    )

    # 4. Cancel Align Controller
    root.add_child(CancelAlignControllerBehavior("CancelAlignAfterYaw"))

    # 5. Reset Odometry
    root.add_child(ResetOdometryBehavior("ResetOdometryAfterYaw"))

    # 6. Delay After Reset
    root.add_child(DelayBehavior("DelayAfterResetYaw", duration=1.0))

    # 7. Clear Object Map
    root.add_child(ClearObjectMapBehavior("ClearObjectMapAfterYaw"))

    # 8. Focus Gate
    root.add_child(SetDetectionFocusBehavior("FocusGateAfterYaw", focus_object="gate"))

    return root
