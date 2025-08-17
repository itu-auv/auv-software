from .initialize import *
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
)

from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.roll import TwoRollState, TwoYawState
from auv_smach.coin_flip import CoinFlipState
from auv_smach.acoustic import AcousticTransmitter


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PublishGateAngleState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "publish_gate_angle", Trigger, request=TriggerRequest()
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


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PublishGateAngleState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "publish_gate_angle", Trigger, request=TriggerRequest()
        )


class NavigateThroughGateState(smach.State):
    def __init__(
        self,
        gate_depth: float,
        gate_search_depth: float,
        gate_exit_angle: float = 0.0,
        roll_depth: float = -0.8,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.roll = rospy.get_param("~roll", True)
        self.yaw = rospy.get_param("~yaw", False)
        self.coin_flip = rospy.get_param("~coin_flip", False)
        self.gate_look_at_frame = "gate_middle_part"
        self.gate_search_frame = "gate_search"
        self.gate_exit_angle = gate_exit_angle
        self.roll_depth = roll_depth

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_INITIAL_GATE_DEPTH",
                SetDepthState(
                    depth=-0.5,
                    sleep_duration=3.0,
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
                    "succeeded": (
                        "COIN_FLIP_STATE"
                        if self.coin_flip
                        else "SET_DETECTION_FOCUS_GATE"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "COIN_FLIP_STATE",
                CoinFlipState(),
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
                    "succeeded": "SET_ROLL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ROLL_DEPTH",
                SetDepthState(
                    depth=self.roll_depth,
                    sleep_duration=3.0,
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
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": (
                        "CALIFORNIA_ROLL"
                        if self.roll
                        else (
                            "TWO_YAW_STATE" if self.yaw else "SET_GATE_TRAJECTORY_DEPTH"
                        )
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CALIFORNIA_ROLL",
                TwoRollState(
                    roll_torque=50.0, gate_look_at_frame=self.gate_look_at_frame
                ),
                transitions={
                    "succeeded": "SET_GATE_TRAJECTORY_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_YAW_STATE",
                TwoYawState(yaw_frame=self.gate_search_frame),
                transitions={
                    "succeeded": "SET_GATE_TRAJECTORY_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_TRAJECTORY_DEPTH",
                SetDepthState(depth=gate_search_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "LOOK_AT_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_GATE",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    set_frame_duration=3.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "SELAM_TO_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SELAM_TO_GATE",
                LookAroundState(
                    angle_offset=0.5,
                    max_angular_velocity=0.15,
                ),
                transitions={
                    "succeeded": "LOOK_AT_GATE_FOR_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_GATE_FOR_TRAJECTORY",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
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
                    "succeeded": "SET_DETECTION_TO_NONE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_TO_NONE",
                SetDetectionFocusState(focus_object="none"),
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
                    "succeeded": "DYNAMIC_PATH_TO_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_ENTRANCE",
                DynamicPathState(
                    plan_target_frame="gate_entrance",
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_EXIT",
                DynamicPathState(
                    plan_target_frame="gate_exit",
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_REQUEST_AFTER_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_REQUEST_AFTER_EXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="gate_exit",
                    angle_offset=self.gate_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "DELAY_FOR_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DELAY_FOR_PINGER",
                DelayState(delay_time=10.0),
                transitions={
                    "succeeded": "TRANSMIT_ACOUSTIC_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TRANSMIT_ACOUSTIC_1",
                AcousticTransmitter(acoustic_data=1),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
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
