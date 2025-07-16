from .initialize import *
import smach
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import PlanPath, PlanPathRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from geometry_msgs.msg import PoseStamped

from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePathState,
    SearchForPropState,
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


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float, gate_search_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize the state machine

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        with self.state_machine:
            smach.StateMachine.add(
                "SET_GATE_SEARCH_DEPTH",
                SetDepthState(depth=gate_search_depth, sleep_duration=3.0),
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
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
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
                    "succeeded": "EXECUTE_PATH_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_PATH_EXIT",
                ExecutePathState(),
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
                    "succeeded": "succeeded",
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
