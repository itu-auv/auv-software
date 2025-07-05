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
    SearchForPropState,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest

from tf.transformations import euler_from_quaternion

from auv_smach.initialize import (
    DelayState,
    OdometryEnableState,
    ResetOdometryPoseState,
    ResetOdometryState,
)


class PitchCorrectionState(smach.State):
    def __init__(
        self, pitch_torque=0.5, rate_hz=20, timeout_s=10.0, pitch_threshold=0.01
    ):
        super(PitchCorrectionState, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.odometry_topic = "odometry"
        self.killswitch_topic = "propulsion_board/status"
        self.wrench_topic = "wrench"
        self.frame_id = "taluy/base_link"

        self.pitch_torque = -pitch_torque
        self.timeout = rospy.Duration(timeout_s)
        self.rate = rospy.Rate(rate_hz)
        self.pitch_threshold = pitch_threshold

        self.odom_ready = False
        self.active = True
        self.last_time = None

        self.sub_odom = rospy.Subscriber(self.odometry_topic, Odometry, self.odom_cb)
        self.sub_kill = rospy.Subscriber(
            self.killswitch_topic, Bool, self.killswitch_cb
        )
        self.pub_wrench = rospy.Publisher(
            self.wrench_topic, WrenchStamped, queue_size=1
        )

    def odom_cb(self, msg: Odometry):
        self.odom_ready = True
        self.last_odom = msg

    def killswitch_cb(self, msg: Bool):
        self.active = msg.data

    def execute(self, userdata):
        start_time = rospy.Time.now()
        wrench_msg = WrenchStamped()
        wrench_msg.header.frame_id = self.frame_id
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.wrench.force.x = 0
        wrench_msg.wrench.force.y = 0
        wrench_msg.wrench.force.z = 0
        wrench_msg.wrench.torque.x = 0
        wrench_msg.wrench.torque.y = self.pitch_torque
        wrench_msg.wrench.torque.z = 0

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            if not self.active:
                return "aborted"

            if (rospy.Time.now() - start_time) > self.timeout:
                return "aborted"

            if self.odom_ready:
                # Get current pitch angle from quaternion
                orientation = self.last_odom.pose.pose.orientation
                _, pitch, _ = euler_from_quaternion(
                    [orientation.x, orientation.y, orientation.z, orientation.w]
                )

                # Check if pitch is positive (front up) and within threshold
                if pitch > self.pitch_threshold:
                    # Stop applying torque when pitch is within threshold
                    wrench_msg.wrench.torque.y = 0
                    self.pub_wrench.publish(wrench_msg)
                    rospy.sleep(1.0)  # Wait a bit to ensure torque is applied
                    return "succeeded"

            wrench_msg.header.stamp = rospy.Time.now()
            self.pub_wrench.publish(wrench_msg)
            self.rate.sleep()

        return "aborted"


class RollTwoTimes(smach.State):
    def __init__(self, roll_torque, rate_hz=20, timeout_s=15.0):
        super(RollTwoTimes, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.odometry_topic = "odometry"
        self.killswitch_topic = "propulsion_board/status"
        self.wrench_topic = "wrench"
        self.frame_id = "taluy/base_link"

        self.roll_torque = roll_torque
        self.timeout = rospy.Duration(timeout_s)
        self.rate = rospy.Rate(rate_hz)

        self.odom_ready = False
        self.active = True
        self.total_roll = 0.0
        self.last_time = None

        self.sub_odom = rospy.Subscriber(self.odometry_topic, Odometry, self.odom_cb)
        self.sub_kill = rospy.Subscriber(
            self.killswitch_topic, Bool, self.killswitch_cb
        )
        self.pub_wrench = rospy.Publisher(
            self.wrench_topic, WrenchStamped, queue_size=1
        )

    def odom_cb(self, msg: Odometry):
        now = rospy.Time.now()

        if not self.odom_ready:
            self.last_time = now
            self.odom_ready = True
            return

        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).to_sec()
        self.last_time = now

        omega_x = msg.twist.twist.angular.x
        delta_angle = omega_x * dt
        self.total_roll += abs(delta_angle)

    def killswitch_cb(self, msg: Bool):
        if not msg.data:
            self.active = False
            rospy.logwarn("ROLL_TWO_TIMES: propulsion board disabled → aborting")

    def execute(self, userdata):
        rospy.loginfo("ROLL_TWO_TIMES: waiting for odometry data…")
        start_wait = rospy.Time.now()
        while not rospy.is_shutdown() and not self.odom_ready:
            if (rospy.Time.now() - start_wait).to_sec() > 5.0:
                rospy.logerr("ROLL_TWO_TIMES: No odometry data after 5 s → abort")
                return "aborted"
            if self.preempt_requested():
                return "preempted"
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                return self._abort_on_shutdown()

        self.total_roll = 0.0
        self.last_time = rospy.Time.now()
        self.start_time = rospy.Time.now()
        target = math.radians(675.0)
        rospy.loginfo(
            "ROLL_TWO_TIMES: starting roll with torque %.2f Nm, target = %.2f rad",
            self.roll_torque,
            target,
        )

        try:
            while not rospy.is_shutdown() and self.total_roll < target and self.active:

                if self.preempt_requested():
                    return self._stop_and("preempted")

                if (rospy.Time.now() - self.start_time) > self.timeout:
                    rospy.logerr(
                        "ROLL_TWO_TIMES: timed out after %.1f s", self.timeout.to_sec()
                    )
                    return self._stop_and("aborted")

                cmd = WrenchStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.header.frame_id = self.frame_id
                cmd.wrench.torque.x = self.roll_torque
                self.pub_wrench.publish(cmd)

                rospy.loginfo_throttle(
                    1.0,
                    "ROLL_TWO_TIMES: total_roll = %.2f / %.2f",
                    self.total_roll,
                    target,
                )

                self.rate.sleep()

        except rospy.ROSInterruptException:
            return self._abort_on_shutdown()

        return self._stop_and("succeeded")

    def _stop_and(self, outcome):
        stop = WrenchStamped()
        stop.header.stamp = rospy.Time.now()
        stop.header.frame_id = self.frame_id
        stop.wrench.torque.x = 0.0
        self.pub_wrench.publish(stop)
        return outcome

    def _abort_on_shutdown(self):
        rospy.logwarn("ROLL_TWO_TIMES: ROS shutdown detected → aborting state")
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
            "toggle_gate_trajectory",
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
    def __init__(self, roll_torque=50.0):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        self.roll_torque = roll_torque
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
                    "succeeded": "PITCH_CORRECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PITCH_CORRECTION",
                PitchCorrectionState(
                    pitch_torque=0.5, rate_hz=20, timeout_s=10.0, pitch_threshold=0.01
                ),
                transitions={
                    "succeeded": "WAIT_FOR_TWO_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROLL_TWO_TIMES",
                RollTwoTimes(roll_torque=self.roll_torque, rate_hz=20, timeout_s=15.0),
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
                    "succeeded": "ALIGN_TO_LOOK_AT_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_LOOK_AT_GATE",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="gate_search"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNMENT",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER_BEFORE_ODOM_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER_BEFORE_ODOM_ENABLE",
                CancelAlignControllerState(),
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
                ResetOdometryState(),
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
    def __init__(self, gate_depth: float, gate_search_depth: float):
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
                    depth=gate_search_depth,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 4.0),
                ),
                transitions={
                    "succeeded": "FIND_AND_AIM_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_GATE",
                SearchForPropState(
                    look_at_frame="gate_blue_arrow_link",
                    alignment_frame="gate_search",
                    full_rotation=True,
                    set_frame_duration=8.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "TWO_ROLL_STATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_ROLL_STATE",
                TwoRollState(roll_torque=50.0),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=gate_depth, sleep_duration=3.0),
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
