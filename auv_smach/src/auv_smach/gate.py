from .initialize import *
import smach
import rospy
import tf2_ros
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePlannedPathsState,
)

from auv_smach.red_buoy import SetFrameLookingAtState

from auv_smach.initialize import DelayState


class PlanGatePathsState(smach.State):
    """State that plans the paths for the gate task"""

    def __init__(self, tf_buffer, return_home: bool = False):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.return_home = return_home
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanGatePathsState] Preempt requested")
                return "preempted"

            path_planners = PathPlanners(
                self.tf_buffer
            )  # instance of PathPlanners with tf_buffer
            paths = path_planners.path_for_gate(
                return_home=self.return_home
            )  # return_home parametresi eklendi
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
    def __init__(self, gate_depth: float, return_home: bool = False):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.return_home = return_home
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
                    "succeeded": "SET_LOOK_AT_GATE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_LOOK_AT_GATE_FRAME",
                SetFrameLookingAtState(
                    base_frame="taluy/base_link",
                    look_at_frame="gate_blue_arrow_link",
                    target_frame="gate_travel_start",
                ),
                transitions={
                    "succeeded": "SET_GATE_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="gate_travel_start"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_TRAVEL_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_TRAVEL_START",
                DelayState(delay_time=5.0),
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
                PlanGatePathsState(self.tf_buffer, return_home=self.return_home),
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
