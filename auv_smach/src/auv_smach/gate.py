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
from auv_smach.acoustic import AcousticTransmitter


class TransformServiceEnableState(smach_ros.ServiceState):
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


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
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
        self.coin_flip_direction = rospy.get_param("~coin_flip", "none")
        self.gate_look_at_frame = "gate_passage"
        self.gate_search_frame = "gate_search"
        self.gate_exit_angle = gate_exit_angle
        self.roll_depth = roll_depth

        self.tx, self.ty, self.yaw_val = 0.0, 0.0, 0.0
        if self.coin_flip_direction == "left":
            self.ty = 1.0
            self.yaw_val = 1.5708
        elif self.coin_flip_direction == "right":
            self.ty = -1.0
            self.yaw_val = -1.5708
        elif self.coin_flip_direction == "back":
            self.tx = -1.0
            self.yaw_val = 3.14159

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_INITIAL_GATE_DEPTH",
                SetDepthState(
                    depth=self.roll_depth,
                    sleep_duration=1.0,
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
                        "CREATE_COIN_FLIP_RESCUER_FRAME"
                        if self.coin_flip_direction != "none"
                        else "SET_DETECTION_FOCUS_GATE"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CREATE_COIN_FLIP_RESCUER_FRAME",
                SetStartFrameState(
                    frame_name="coin_flip_rescuer",
                    translation_x=self.tx,
                    translation_y=self.ty,
                    rotation_yaw=self.yaw_val,
                ),
                transitions={
                    "succeeded": "RESCUE_COIN_FLIP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESCUE_COIN_FLIP",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="coin_flip_rescuer",
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=30.0,
                    cancel_on_success=True,
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
                    "succeeded": "AIM_AT_GATE_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "AIM_AT_GATE_TARGET",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    timeout=30.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": (
                        "CALIFORNIA_ROLL"
                        if self.roll
                        else (
                            "TWO_YAW_STATE"
                            if self.yaw
                            else "DISABLE_GATE_TRAJECTORY_PUBLISHER"
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
                    "succeeded": "DISABLE_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_YAW_STATE",
                TwoYawState(yaw_frame=self.gate_search_frame),
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
                SetDepthState(depth=gate_depth, sleep_duration=2.0),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_GATE",
                DynamicPathState(
                    plan_target_frame="gate_passage",
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_REQUEST_AFTER_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_REQUEST_AFTER_GATE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="gate_passage",
                    angle_offset=self.gate_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=10.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
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
