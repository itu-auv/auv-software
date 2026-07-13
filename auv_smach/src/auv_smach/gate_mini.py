from auv_smach.tf_utils import get_tf_buffer, get_base_link
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
    MonitorVisibilityState,
    AlignFrameWithVisibilityCheck,
    ClearObjectMapState,
)

from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.roll import PitchTwoTimes, TwoRollState, TwoYawState
from auv_smach.coin_flip import CoinFlipState
from auv_smach.acoustic import AcousticTransmitter
from std_msgs.msg import Bool


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
        pitch_torque: float = -100.0,
        pitch_timeout: float = 15.0,
        pitch_depth: float = -0.65,
        after_pitch_depth: float = -0.45,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.base_link = get_base_link()
        self.roll = rospy.get_param("~roll", True)
        self.yaw = rospy.get_param("~yaw", False)
        self.coin_flip = rospy.get_param("~coin_flip", False)
        self.gate_look_at_frame = "mini_gate_exit"
        self.gate_search_frame = "gate_search"
        self.gate_exit_angle = gate_exit_angle
        self.roll_depth = roll_depth
        self.target_animal = target_animal
        self.pitch_torque = pitch_torque
        self.pitch_timeout = pitch_timeout
        self.pitch_depth = pitch_depth
        self.after_pitch_depth = after_pitch_depth

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "ilk_state_tir",
                ResetOdometryPositionState(),
                transitions={
                    "succeeded": "open_gate_detection",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "open_gate_detection",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "SET_INITIAL_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_INITIAL_GATE_DEPTH",
                SetDepthState(
                    depth=self.after_pitch_depth,
                    depth_threshold=0.1,
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
                    "succeeded": "GATE_E_DON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "GATE_E_DON",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="odom",
                    dist_threshold=100.0,
                    yaw_threshold=0.2,
                    confirm_duration=2.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.001,
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
                    "succeeded": "wait_for_gate_trajectory",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "wait_for_gate_trajectory",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ilk_entrance",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
########################################### sequence 1
            smach.StateMachine.add(
                "ilk_entrance",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="mini_gate_entrance",
                    dist_threshold=0.3,
                    yaw_threshold=0.2,
                    confirm_duration=1.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.05,
                    max_linear_velocity_y=0.001,
                    max_angular_velocity=0.4,
                ),
                transitions={
                    "succeeded": "yavas_entrance",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "yavas_entrance",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="mini_gate_entrance",
                    dist_threshold=100.0,
                    yaw_threshold=1.0,
                    confirm_duration=0.1,
                    timeout=10.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.0001,
                    max_linear_velocity_z=0.6,
                    max_angular_velocity=0.4,
                ),
                transitions={
                    "succeeded": "depth_denemesi",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "depth_denemesi",
                SetDepthState(
                    depth=self.pitch_depth,
                    depth_threshold=0.1,
                    max_velocity=0.4,
                ),
                transitions={
                    "succeeded": "depth_denemesi_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "depth_denemesi_2",
                SetDepthState(
                    depth=self.pitch_depth,
                    depth_threshold=0.1,
                    max_velocity=0.05,
                ),
                transitions={
                    "succeeded": "ilk_pitch",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ilk_pitch",
                PitchTwoTimes(
                    pitch_torque=self.pitch_torque,
                    timeout_s=self.pitch_timeout,
                ),
                transitions={
                    "succeeded": "bekle",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "bekle",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "RESET_ODOMETRY_POSITION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESET_ODOMETRY_POSITION",
                ResetOdometryPositionState(),
                transitions={
                    "succeeded": "depth_denemesi_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "depth_denemesi_3",
                SetDepthState(
                    depth=self.after_pitch_depth,
                    depth_threshold=0.1,
                ),
                transitions={
                    "succeeded": "pitch_arasi_bakis",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "pitch_arasi_bakis",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                    confirm_duration=1.0,
                ),
                transitions={
                    "succeeded": "ikinci_entrance",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
########################################### sequence 2
            smach.StateMachine.add(
                "ikinci_entrance",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="mini_gate_entrance",
                    dist_threshold=0.3,
                    yaw_threshold=0.2,
                    confirm_duration=1.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.03,
                    max_linear_velocity_y=0.001,
                    max_angular_velocity=0.4,
                ),
                transitions={
                    "succeeded": "yavas_entrance_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )            
            smach.StateMachine.add(
                "yavas_entrance_2",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="mini_gate_entrance",
                    dist_threshold=100.0,
                    yaw_threshold=1.0,
                    confirm_duration=0.1,
                    timeout=10.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.0001,
                    max_linear_velocity_z=0.6,
                    max_angular_velocity=0.4,
                ),
                transitions={
                    "succeeded": "depth_denemesi_5",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "depth_denemesi_5",
                SetDepthState(
                    depth=self.pitch_depth,
                    depth_threshold=0.1,
                    max_velocity=0.4,
                ),
                transitions={
                    "succeeded": "depth_denemesi_6",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "depth_denemesi_6",
                SetDepthState(
                    depth=self.pitch_depth,
                    depth_threshold=0.1,
                    max_velocity=0.05,
                ),
                transitions={
                    "succeeded": "ikinci_pitch",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ikinci_pitch",
                PitchTwoTimes(
                    pitch_torque=self.pitch_torque,
                    timeout_s=self.pitch_timeout,
                ),
                transitions={
                    "succeeded": "bekle_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "bekle_2",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "RESET_ODOMETRY_POSITION_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESET_ODOMETRY_POSITION_2",
                ResetOdometryPositionState(),
                transitions={
                    "succeeded": "depth_denemesi_7",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )          
            smach.StateMachine.add(
                "depth_denemesi_7",
                SetDepthState(
                    depth=self.after_pitch_depth,
                    depth_threshold=0.1,
                ),
                transitions={
                    "succeeded": "pitch_arasi_bakis_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )  
            smach.StateMachine.add(
                "pitch_arasi_bakis_2",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                    confirm_duration=1.0,
                ),
                transitions={
                    "succeeded": "m",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
#############################################################
            smach.StateMachine.add(
                "m",
                SetDepthState(
                    depth=-1.1,
                ),
                transitions={
                    "succeeded": "n",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "n",
                AlignFrameWithVisibilityCheck(
                    source_frame=self.base_link,
                    target_frame=self.gate_look_at_frame,
                    prop_name=self.target_animal,
                    lost_timeout=6.0,
                    angle_offset=self.gate_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=10.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                    max_linear_velocity=0.15,
                    max_linear_velocity_y=0.05,
                ),
                transitions={
                    "succeeded": "DISABLE_GATE_DETECTION",
                    "target_lost": "DISABLE_GATE_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_GATE_DETECTION",
                SetDetectionState(camera_name="front", enable=False),
                transitions={
                    "succeeded": "ENABLE_SLALOM_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_SLALOM_DETECTION",
                SetDetectionState(camera_name="slalom", enable=True),
                transitions={
                    "succeeded": "SET_SLALOM_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_SLALOM_FOCUS",
                SetDetectionFocusState(focus_object="slalom"),
                transitions={
                    "succeeded": "SEARCH_RED_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_RED_PIPE",
                SearchForPropState(
                    look_at_frame="slalom_red_pipe_link",
                    alignment_frame="slalom_mini_search",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=-0.2,
                ),
                transitions={
                    "succeeded": "kapa_gate_sonda",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "kapa_gate_sonda",
                TransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "succeeded",  # ev için eklendi robosubda sil
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # smach.StateMachine.add(
            #     "amerika",
            #     SetDepthState(
            #         depth=-1,
            #     ),
            #     transitions={
            #         "succeeded": "succeeded",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )

    def execute(self, userdata):
        rospy.logdebug(
            "[NavigateThroughGateMiniState] Starting state machine execution."
        )

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
