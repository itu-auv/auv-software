from auv_smach.tf_utils import get_tf_buffer, get_base_link
from .initialize import *
import math
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    AlignFrame,
    DynamicPathState,
    SetDetectionFocusState,
    LookAroundState,
    MonitorVisibilityState,
    AlignFrameWithVisibilityCheck,
)

from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.roll import PitchTwoTimes, TwoRollState, TwoYawState
from auv_smach.coin_flip import CoinFlipState
from auv_smach.acoustic import AcousticTransmitter
from std_msgs.msg import Bool


START_DIRECTION_TO_YAW = {
    "turn_right": -math.pi / 2.0,
    "turn_left": math.pi / 2.0,
    "turn_back": math.pi,
}


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_mini_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TransformServiceEnableStateTaluy(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


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


class NavigateThroughGateMiniState(smach.State):
    def __init__(
        self,
        gate_depth: float,
        gate_search_depth: float,
        gate_exit_angle: float = 0.0,
        roll_depth: float = -0.8,
        target_animal: str = "gate_survey_repair_link",
        pitch_torque: float = -90.0,
        pitch_timeout: float = 15.0,
        mini_coin_flip: str = "",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.base_link = get_base_link()
        self.roll = rospy.get_param("~roll", True)
        self.yaw = rospy.get_param("~yaw", False)
        self.coin_flip = rospy.get_param("~coin_flip", False)
        self.gate_look_at_frame = "mini_gate_middle_part"
        self.gate_search_frame = "gate_search"
        self.gate_exit_angle = gate_exit_angle
        self.roll_depth = roll_depth
        self.target_animal = target_animal
        self.pitch_torque = pitch_torque
        self.pitch_timeout = pitch_timeout
        self.start_frame_yaw = self.get_start_frame_yaw(mini_coin_flip)

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_START_FRAME",
                SetStartFrameState(
                    frame_name="mini_coin_flip",
                    rotation_yaw=self.start_frame_yaw,
                ),
                transitions={
                    "succeeded": "GATE_E_DON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "GATE_E_DON",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="mini_coin_flip",
                    dist_threshold=0.2,
                    yaw_threshold=0.2,
                    confirm_duration=2.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.15,
                    max_linear_velocity_y=0.05,
                ),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_GATE",
                SetDetectionFocusState(focus_object="gate"),
                transitions={
                    "succeeded": "SET_INITIAL_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_INITIAL_GATE_DEPTH",
                SetDepthState(
                    depth=-1.1,
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
                    "succeeded": "ENABLE_GATE_TRAJECTORY_PUBLISHER_taluy",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY_PUBLISHER_taluy",
                TransformServiceEnableStateTaluy(req=True),
                transitions={
                    "succeeded": "SABIT_BAKMA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SABIT_BAKMA",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="gate_entrance",
                    dist_threshold=0.2,
                    yaw_threshold=0.2,
                    confirm_duration=10.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.02,
                    max_linear_velocity_y=0.02,
                    max_angular_velocity=0.1,
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
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=-1.3),
                transitions={
                    "succeeded": "ALIGN_FRAME_TO_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_TO_GATE",
                AlignFrameWithVisibilityCheck(
                    source_frame=self.base_link,
                    target_frame=self.gate_look_at_frame,
                    prop_name=self.target_animal,
                    lost_timeout=6.0,
                    angle_offset=self.gate_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=20.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                    max_linear_velocity=0.15,
                    max_linear_velocity_y=0.05,
                ),
                transitions={
                    "succeeded": "a",
                    "target_lost": "a",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "a",
                SetDepthState(
                    depth=-1.0,
                    confirm_duration=5.0,
                    timeout=15.0,
                ),
                transitions={
                    "succeeded": "PITCH_TWO_TIMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PITCH_TWO_TIMES",
                PitchTwoTimes(
                    pitch_torque=self.pitch_torque,
                    timeout_s=self.pitch_timeout,
                ),
                transitions={
                    "succeeded": "SON_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SON_DEPTH",
                SetDepthState(
                    depth=-1.0,
                    confirm_duration=1.0,
                    timeout=15.0,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    @staticmethod
    def get_start_frame_yaw(mini_coin_flip: str) -> float:
        if not mini_coin_flip:
            return 0.0

        if mini_coin_flip not in START_DIRECTION_TO_YAW:
            rospy.logwarn(
                "Unknown mini_coin_flip '%s'. Using 0 yaw.",
                mini_coin_flip,
            )
            return 0.0

        return START_DIRECTION_TO_YAW[mini_coin_flip]

    def execute(self, userdata):
        rospy.logdebug(
            "[NavigateThroughGateMiniState] Starting state machine execution."
        )

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
