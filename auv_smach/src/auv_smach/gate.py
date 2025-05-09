from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
import math
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePlannedPathsState,
    ClearObjectMapState,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_smach.initialize import DelayState, OdometryEnableState, ResetOdometryPoseState


import numpy as np

from geometry_msgs.msg import TransformStamped


class RollTwoTimes(smach.State):
    def __init__(
        self,
        odometry_topic="odometry",
        killswitch_topic="propulsion_board/status",
        wrench_topic="wrench",
        roll_rate=30.0,
        timeout=rospy.Duration(60),
        frame_id="taluy/base_link",
        pitch_Kp=10.0,
        pitch_Kd=10.0,
        yaw_Kp=10.0,
        yaw_Kd=10.0,
        max_pitch_torque=50.0,
        max_yaw_torque=20.0,
    ):
        super(RollTwoTimes, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        # Topics and frames
        self.odometry_topic = odometry_topic
        self.killswitch_topic = killswitch_topic
        self.wrench_topic = wrench_topic
        self.frame_id = frame_id

        # Motion and stabilization parameters
        self.roll_rate = roll_rate
        self.timeout = timeout
        self.pitch_Kp = pitch_Kp
        self.pitch_Kd = pitch_Kd
        self.yaw_Kp = yaw_Kp
        self.yaw_Kd = yaw_Kd
        self.max_pitch_torque = max_pitch_torque
        self.max_yaw_torque = max_yaw_torque

        self.rate = rospy.Rate(20)
        self.active = True
        self.odom_ready = False
        self.total_roll = 0.0
        self.last_time = None

        # Track reference and current orientation and rates
        self.initial_pitch = 0.0
        self.initial_yaw = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.omega_y = 0.0
        self.omega_z = 0.0

        # TF interfaces (reuse if needed for yaw frame)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ROS interfaces
        self.sub_odom = rospy.Subscriber(self.odometry_topic, Odometry, self.odom_cb)
        self.sub_kill = rospy.Subscriber(
            self.killswitch_topic, Bool, self.killswitch_cb
        )
        self.pub_wrench = rospy.Publisher(
            self.wrench_topic, WrenchStamped, queue_size=1
        )

    def odom_cb(self, msg: Odometry):
        now = rospy.Time.now()
        # Orientation
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # Angular rates
        self.omega_y = msg.twist.twist.angular.y
        self.omega_z = msg.twist.twist.angular.z

        self.current_pitch = pitch
        self.current_yaw = yaw

        if not self.odom_ready:
            # initialize reference angles
            self.initial_pitch = pitch
            self.initial_yaw = yaw
            self.last_time = now
            self.odom_ready = True
            return

        if self.last_time:
            dt = (now - self.last_time).to_sec()
            # accumulate roll from odometry x-rate
            omega_x = msg.twist.twist.angular.x
            self.total_roll += abs(omega_x * dt)
            self.last_time = now

    def killswitch_cb(self, msg: Bool):
        if not msg.data:
            self.active = False
            rospy.logwarn("ROLL_TWO_TIMES: propulsion board disabled → aborting")

    def calculate_stabilizing_torques(self):
        # Pitch error
        pitch_error = self.initial_pitch - self.current_pitch
        # Derivative: negative of current pitch rate
        pitch_d = -self.omega_y
        # PD control
        torque_pitch = self.pitch_Kp * pitch_error + self.pitch_Kd * pitch_d
        # Saturate
        torque_pitch = max(
            -self.max_pitch_torque, min(self.max_pitch_torque, torque_pitch)
        )

        # Yaw error (normalize)
        yaw_error = self.normalize_angle(self.initial_yaw - self.current_yaw)
        yaw_d = -self.omega_z
        torque_yaw = self.yaw_Kp * yaw_error + self.yaw_Kd * yaw_d
        torque_yaw = max(-self.max_yaw_torque, min(self.max_yaw_torque, torque_yaw))

        return torque_pitch, torque_yaw

    def normalize_angle(self, angle):
        # wrap to [-pi, pi]
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def execute(self, userdata):
        rospy.loginfo("ROLL_TWO_TIMES: waiting for odometry data…")
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and not self.odom_ready:
            if (rospy.Time.now() - t0).to_sec() > 5.0:
                rospy.logerr("No odom after 5s → abort")
                return "aborted"
            if self.preempt_requested():
                return "preempted"
            self.rate.sleep()

        # Reset
        self.total_roll = 0.0
        self.last_time = rospy.Time.now()
        start_time = rospy.Time.now()
        # target = 2 * 2 * math.pi  # two full revolutions in rad
        target = (660 * math.pi) / 180.0  # 660 degrees in radians
        rospy.loginfo(f"Rolling {math.degrees(target):.0f}° at {self.roll_rate} rad/s")

        while not rospy.is_shutdown() and self.total_roll < target and self.active:
            if self.preempt_requested():
                return self._stop_and("preempted")
            if (rospy.Time.now() - start_time) > self.timeout:
                rospy.logerr("Timeout after %.1f s", self.timeout.to_sec())
                return self._stop_and("aborted")

            # get stabilizing torques
            torque_pitch, torque_yaw = self.calculate_stabilizing_torques()
            cmd = WrenchStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = self.frame_id
            # roll
            cmd.wrench.torque.x = self.roll_rate
            # stabilization
            cmd.wrench.torque.y = torque_pitch
            cmd.wrench.torque.z = torque_yaw
            # no forces
            cmd.wrench.force.x = 0.0
            cmd.wrench.force.y = 0.0
            cmd.wrench.force.z = 0.0
            self.pub_wrench.publish(cmd)

            # log
            progress = self.total_roll / target * 100
            rospy.loginfo_throttle(
                1.0,
                f"Progress: {progress:.1f}%, pitch_t: {torque_pitch:.2f}, yaw_t: {torque_yaw:.2f}",
            )
            self.rate.sleep()

        return self._stop_and("succeeded")

    def _stop_and(self, outcome):
        stop = WrenchStamped()
        stop.header.stamp = rospy.Time.now()
        stop.header.frame_id = self.frame_id
        stop.wrench.torque.x = 0.0
        stop.wrench.torque.y = 0.0
        stop.wrench.torque.z = 0.0
        stop.wrench.force.x = 0.0
        stop.wrench.force.y = 0.0
        stop.wrench.force.z = 0.0
        self.pub_wrench.publish(stop)
        return outcome

    def _abort_on_shutdown(self):
        rospy.logwarn("ROS shutdown detected → aborting")
        return self._stop_and("aborted")


class PlanGatePathsState(smach.State):
    """State that plans the paths for the gate task"""

    def __init__(self, tf_buffer):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanGatePathsState] Preempt requested")
                return "preempted"

            path_planners = PathPlanners(self.tf_buffer)
            paths = path_planners.path_for_gate()
            if paths is None:
                return "aborted"

            userdata.planned_paths = paths
            return "succeeded"

        except Exception as e:
            rospy.logerr("[PlanGatePathsState] Error: %s", str(e))
            return "aborted"


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class DvlOdometryServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "/dvl_to_odom_node/enable",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TwoRollState(smach.StateMachine):
    def __init__(self):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        with self:
            smach.StateMachine.add(
                "DISABLE_DVL_ODOM",
                DvlOdometryServiceEnableState(req=False),
                transitions={
                    "succeeded": "WAIT_FOR_TWO_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TWO_ROLL",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ROLL_TWO_TIMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROLL_TWO_TIMES",
                RollTwoTimes(roll_rate=30.0),
                transitions={
                    "succeeded": "WAIT_FOR_ENABLE_DVL_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ENABLE_DVL_ODOM",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "ENABLE_DVL_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_DVL_ODOM",
                DvlOdometryServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_RESET_ODOMETRY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_RESET_ODOMETRY",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "ODOMETRY_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ODOMETRY_ENABLE",
                OdometryEnableState(),
                transitions={
                    "succeeded": "RESET_ODOMETRY_POSE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESET_ODOMETRY_POSE",
                ResetOdometryPoseState(),
                transitions={
                    "succeeded": "DELAY_AFTER_RESET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DELAY_AFTER_RESET",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "CLEAR_OBJECT_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLEAR_OBJECT_MAP",
                ClearObjectMapState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:

            smach.StateMachine.add(
                "SET_ROLL_DEPTH",
                SetDepthState(
                    depth=-0.7,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 4.0),
                ),
                transitions={
                    "succeeded": "TWO_ROLL_STATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_ROLL_STATE",
                TwoRollState(),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(
                    depth=gate_depth,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 5.0),
                ),
                transitions={
                    "succeeded": "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_GATE_TRAJECTORY_PUBLISHER",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "DISABLE_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "PLAN_GATE_PATHS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_GATE_PATHS",
                PlanGatePathsState(self.tf_buffer),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="dynamic_target"
                ),
                transitions={
                    "succeeded": "EXECUTE_GATE_PATHS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_GATE_PATHS",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughGateState] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
