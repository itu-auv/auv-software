from .initialize import *
import smach
import rospy
import tf2_ros
import math
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePlannedPathsState,
)
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from tf.transformations import euler_from_quaternion

from auv_smach.initialize import DelayState

class TwoRollState(smach.State):
    def __init__(self, roll_rate=100.0, rate_hz=20, timeout_s=15.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.imu_topic = "imu/data"
        self.cmd_vel_topic = "cmd_vel"
        self.roll_rate = roll_rate
        self.timeout = rospy.Duration(timeout_s)

        self.imu_ready = False
        self.roll = 0.0
        self.roll_prev = None
        self.total_roll = 0.0
        self.active = True

        self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self.imu_cb)
        self.sub_kill = rospy.Subscriber(
            "propulsion_board/status", Bool, self.killswitch_cb
        )
        self.pub_cmd = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.pub_enable = rospy.Publisher("enable", Bool, queue_size=1)

        self.rate = rospy.Rate(rate_hz)

    def imu_cb(self, msg):
        # convert quaternion → roll
        q = msg.orientation
        _, _, r = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.roll = r
        self.imu_ready = True

    def killswitch_cb(self, msg):
        if not msg.data:
            self.active = False
            rospy.logwarn("TwoRollState: propulsion board disabled → aborting")

    @staticmethod
    def normalize_angle(angle):
        # wrap to [-π, π]
        return math.atan2(math.sin(angle), math.cos(angle))

    def execute(self, userdata):
        rospy.loginfo("TwoRollState: waiting for IMU data…")
        start_wait = rospy.Time.now()
        while not rospy.is_shutdown() and not self.imu_ready:
            if (rospy.Time.now() - start_wait).to_sec() > 5.0:
                rospy.logerr("No IMU data after 5 s → abort")
                return "aborted"
            if self.preempt_requested():
                return "preempted"
            self.rate.sleep()

        self.roll_prev = self.roll
        self.total_roll = 0.0
        self.start_time = rospy.Time.now()

        twist = Twist()
        twist.angular.x = self.roll_rate

        target = 4 * math.pi
        rospy.loginfo(
            "TwoRollState: starting roll at %.2f rad/s, target = %.2f rad",
            self.roll_rate,
            target,
        )

        while (
            not rospy.is_shutdown()
            and self.total_roll < target
            and self.active
        ):
            if self.preempt_requested():
                twist.angular.x = 0.0
                self.pub_cmd.publish(twist)
                return "preempted"

            if rospy.Time.now() - self.start_time > self.timeout:
                rospy.logerr("TwoRollState: timed out after %.1f s", self.timeout.to_sec())
                twist.angular.x = 0.0
                self.pub_cmd.publish(twist)
                return "aborted"

            self.pub_enable.publish(Bool(data=True))
            self.pub_cmd.publish(twist)

            delta = self.normalize_angle(self.roll - self.roll_prev)
            self.total_roll += abs(delta)
            self.roll_prev = self.roll

            if rospy.get_time() % 1.0 < 0.05:
                rospy.logdebug(
                    "roll=%.2f, accumulated=%.2f/%.2f",
                    self.roll,
                    self.total_roll,
                    target,
                )

            self.rate.sleep()

        twist.angular.x = 0.0
        self.pub_cmd.publish(twist)

        if not self.active:
            return "aborted"

        rospy.loginfo(
            "TwoRollState: completed two rolls (%.2f rad) in %.2f s",
            self.total_roll,
            (rospy.Time.now() - self.start_time).to_sec(),
        )
        return "succeeded"


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

            path_planners = PathPlanners(
                self.tf_buffer
            )  # instance of PathPlanners with tf_buffer
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


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:

            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
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
                    "succeeded": "DISABLE_DVL_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_DVL_ODOM",
                smach_ros.ServiceState(
                    "/dvl_to_odom_node/enable",
                    SetBool,
                    request=SetBoolRequest(data=False),
                ),
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
                    "succeeded": "TWO_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_ROLL",
                TwoRollState(),
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
                smach_ros.ServiceState(
                    "/dvl_to_odom_node/enable",
                    SetBool,
                    request=SetBoolRequest(data=True),
                ),
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
            # 1 second delay after resetting odometry pose
            smach.StateMachine.add(
                "DELAY_AFTER_RESET",
                DelayState(delay_time=1.0),
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
                    "preempted": "CANCEL_ALIGN_CONTROLLER",  # if aborted or preempted, cancel the alignment request
                    "aborted": "CANCEL_ALIGN_CONTROLLER",  # to disable the controllers.
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "DISABLE_DVL_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughGateState] Starting state machine execution.")

        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:  # ctrl + c
            return "preempted"
        # Return the outcome of the state machine
        return outcome
