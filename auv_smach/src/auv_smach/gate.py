from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import PlanPath, PlanPathRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePathState,
    SearchForPropState,
    AlignFrame,
)
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.initialize import DelayState
from auv_smach.roll import TwoRollState


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class RescueCoinFlipServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_coin_flip_rescuer",
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


class SetPlanState(smach.State):
    """State that calls the /set_plan service"""

    def __init__(self, target_frame: str):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.target_frame = target_frame

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[SetPlanState] Preempt requested")
                return "preempted"

            rospy.wait_for_service("/set_plan")
            set_plan = rospy.ServiceProxy("/set_plan", PlanPath)

            req = PlanPathRequest(
                target_frame=self.target_frame,
            )
            set_plan(req)
            return "succeeded"

        except Exception as e:
            rospy.logerr("[SetPlanState] Error: %s", str(e))
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


class SetPlanningNotActive(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "/stop_planning", Trigger, request=TriggerRequest()
        )


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float, gate_search_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.sim_mode = rospy.get_param("~sim", False)

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "RESCUE_COIN_FLIP_SERVICE_ENABLE",
                RescueCoinFlipServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_RESCUE_COIN_FLIP_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_RESCUE_COIN_FLIP_FRAME",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "RESCUE_COIN_FLIP_SERVICE_DISABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESCUE_COIN_FLIP_SERVICE_DISABLE",
                RescueCoinFlipServiceEnableState(req=False),
                transitions={
                    "succeeded": "ALIGN_TO_RESCUE_COIN_FLIP_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_RESCUE_COIN_FLIP_FRAME",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="coin_flip_rescuer",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "SET_ROLL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ROLL_DEPTH",
                SetDepthState(depth=gate_search_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "ALIGN_ORIENTATION_TO_RESCUE_COIN_FLIP_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_ORIENTATION_TO_RESCUE_COIN_FLIP_FRAME",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="coin_flip_rescuer",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    keep_orientation=False,
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
                    look_at_frame="gate_shark_link",
                    alignment_frame="gate_search",
                    full_rotation=False,
                    set_frame_duration=3.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": (
                        "TWO_ROLL_STATE"
                        if not self.sim_mode
                        else "SET_GATE_TRAJECTORY_DEPTH"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_ROLL_STATE",
                TwoRollState(roll_torque=50.0),
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
                    "succeeded": "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "LOOK_AT_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_GATE",
                SearchForPropState(
                    look_at_frame="gate_shark_link",
                    alignment_frame="gate_search",
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "START_PLANNING_TO_GATE_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "START_PLANNING_TO_GATE_ENTRANCE",
                SetPlanState(target_frame="gate_entrance"),
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
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=gate_depth, sleep_duration=3.0),
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
                    "succeeded": "EXECUTE_GATE_PATH_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_GATE_PATH_ENTRANCE",
                ExecutePathState(),
                transitions={
                    "succeeded": "START_PLANNING_TO_GATE_EXIT",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",  # if aborted or preempted, cancel the alignment request
                    "aborted": "CANCEL_ALIGN_CONTROLLER",  # to disable the controllers.
                },
            )
            smach.StateMachine.add(
                "START_PLANNING_TO_GATE_EXIT",
                SetPlanState(target_frame="gate_exit"),
                transitions={
                    "succeeded": "EXECUTE_GATE_PATH_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_GATE_PATH_EXIT",
                ExecutePathState(),
                transitions={
                    "succeeded": "STOP_PLANNING",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",  # if aborted or preempted, cancel the alignment request
                    "aborted": "CANCEL_ALIGN_CONTROLLER",  # to disable the controllers.
                },
            )
            smach.StateMachine.add(
                "STOP_PLANNING",
                SetPlanningNotActive(),
                transitions={
                    "succeeded": "ALIGN_FRAME_REQUEST_AFTER_EXIT",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_REQUEST_AFTER_EXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="gate_exit",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.0,
                    timeout=10.0,
                    cancel_on_success=True,
                    keep_orientation=False,
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
