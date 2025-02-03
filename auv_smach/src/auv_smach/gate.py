from .initialize import *
import smach
import rospy
import tf2_ros
from auv_navigation.path_planners import PathPlanners
from auv_navigation import follow_path_action_client
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePlannedPathsState
)

class PlanGatePathsState(smach.State):
    """State that plans the paths for the gate task"""
    def __init__(self, tf_buffer):
        smach.State.__init__(
            self, 
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"]
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanGatePathsState] Preempt requested")
                return "preempted"
            
            path_planners = PathPlanners(self.tf_buffer)
            paths = path_planners.path_for_gate()
            rospy.loginfo("[PlanGatePathsState] Planned paths: %s", paths)
            rospy.logdebug("[PlanGatePathsState] Paths: %s", paths)
            if paths is None:
                return "aborted"
            
            userdata.planned_paths = paths
            return "succeeded"
            
        except Exception as e:
            rospy.logerr("[PlanGatePathsState] Error: %s", str(e))
            return "aborted"

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
                "SET_GATE_DEPTH",
                SetDepthState(depth=gate_depth),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_GATE_PATHS",
                PlanGatePathsState(self.tf_buffer),
                transitions={
                    "succeeded": "EXECUTE_GATE_PATHS",
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
                    "succeeded": "PLAN_GATE_PATHS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_GATE_PATHS",
                ExecutePlannedPathsState(),
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
        rospy.logdebug("[NavigateThroughGateState] Executing state machine")
        try:
            outcome = self.state_machine.execute()
            if outcome not in ["succeeded", "preempted", "aborted"]:
                rospy.logerr("[NavigateThroughGateState] Invalid outcome returned: %s", outcome)
                return "aborted"
            return outcome
        except Exception as e:
            rospy.logerr("[NavigateThroughGateState] Error: %s", str(e))
            return "aborted"
