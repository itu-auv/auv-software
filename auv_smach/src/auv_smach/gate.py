from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from geometry_msgs.msg import PoseStamped
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePathState,
    ClearObjectMapState,
    SearchForPropState,
    AlignFrame,
    DynamicPathState,
    SetDetectionFocusState,
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.initialize import DelayState, OdometryEnableState, ResetOdometryState
from auv_smach.roll import TwoRollState


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
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.roll_mode = rospy.get_param("~roll_mode", False)

        # Gate task parameters
        gate_task_params = rospy.get_param("~gate_task", {})
        set_roll_depth_params = gate_task_params.get("set_roll_depth", {})
        find_and_aim_gate_params = gate_task_params.get("find_and_aim_gate", {})
        two_roll_state_params = gate_task_params.get("two_roll_state", {})
        set_gate_trajectory_depth_params = gate_task_params.get(
            "set_gate_trajectory_depth", {}
        )
        look_at_gate_params = gate_task_params.get("look_at_gate", {})
        set_gate_depth_params = gate_task_params.get("set_gate_depth", {})
        align_frame_after_exit_params = gate_task_params.get(
            "align_frame_after_exit", {}
        )

        # Mission selection parameters
        self.target_selection = rospy.get_param("~target_selection", "shark")
        if self.target_selection == "shark":
            self.gate_look_at_frame = "gate_shark_link"
        elif self.target_selection == "sawfish":
            self.gate_look_at_frame = "gate_sawfish_link"
        else:
            self.gate_look_at_frame = "gate_shark_link"  # fallback

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_GATE",
                SetDetectionFocusState(focus_object="gate"),
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
                    "succeeded": "SET_ROLL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ROLL_DEPTH",
                SetDepthState(
                    depth=set_roll_depth_params.get("depth", -0.7),
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
                    look_at_frame="gate_middle_part,
                    alignment_frame="gate_search",
                    full_rotation=find_and_aim_gate_params.get("full_rotation", False),
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=find_and_aim_gate_params.get("rotation_speed", 0.2),
                ),
                transitions={
                    "succeeded": (
                        "CALIFORNIA_ROLL"
                        if self.roll_mode
                        else "SET_GATE_TRAJECTORY_DEPTH"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CALIFORNIA_ROLL",
                TwoRollState(roll_torque=50.0),
                transitions={
                    "succeeded": "SET_GATE_TRAJECTORY_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_TRAJECTORY_DEPTH",
                SetDepthState(
                    depth=set_gate_trajectory_depth_params.get("depth", -0.7),
                    sleep_duration=3.0,
                ),
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
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "LOOK_LEFT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_LEFT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=self.gate_search_frame,
                    angle_offset=0.5,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.2,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.15,
                ),
                transitions={
                    "succeeded": "LOOK_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_RIGHT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=self.gate_search_frame,
                    angle_offset=-0.5,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.2,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
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
                    alignment_frame="gate_search",
                    full_rotation=look_at_gate_params.get("full_rotation", False),
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=look_at_gate_params.get("rotation_speed", 0.2),
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
                SetDepthState(
                    depth=set_gate_depth_params.get("depth", -1.5), sleep_duration=3.0
                ),
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
                    angle_offset=align_frame_after_exit_params.get("angle_offset", 0.0),
                    dist_threshold=align_frame_after_exit_params.get(
                        "dist_threshold", 0.1
                    ),
                    yaw_threshold=align_frame_after_exit_params.get(
                        "yaw_threshold", 0.1
                    ),
                    confirm_duration=align_frame_after_exit_params.get(
                        "confirm_duration", 0.0
                    ),
                    timeout=10.0,
                    cancel_on_success=align_frame_after_exit_params.get(
                        "cancel_on_success", True
                    ),
                    keep_orientation=align_frame_after_exit_params.get(
                        "keep_orientation", False
                    ),
                ),
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
