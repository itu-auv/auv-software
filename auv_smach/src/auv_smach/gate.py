from .initialize import *
import smach
import rospy
import tf2_ros
from auv_navigation import path_planners
from auv_navigation import follow_path_action_client
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
)
class PlanGatePathsState(smach.State):
    """State that plans the paths for the gate task"""
    def __init__(self, tf_buffer):
        smach.State.__init__(
            self, 
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["gate_paths"]
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata):
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanGatePathsState] Preempt requested")
                return "preempted"
            
            paths = path_planners.path_for_gate(self.tf_buffer)
            if paths is None:
                return "aborted"
            
            userdata.gate_paths = paths
            return "succeeded"
            
        except Exception as e:
            rospy.logerr("[PlanGatePathsState] Error: %s", str(e))
            return "aborted"


class ExecuteGatePathsState(smach.State):
    """State that executes the planned paths for the gate task"""
    def __init__(self):
        smach.State.__init__(
            self, 
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=["gate_paths"]
        )
        self._client = None 

    def execute(self, userdata):
        # Lazy initialization of the action client
        if self._client is None:
            self._client = follow_path_action_client.FollowPathActionClient()
        try:
            if self.preempt_requested():
                rospy.logwarn("[ExecuteGatePathsState] Preempt requested")
                return "preempted"
            
            success = self._client.execute_paths(userdata.gate_paths)
            return "succeeded" if success else "aborted"
            
        except Exception as e:
            rospy.logerr("[ExecuteGatePathsState] Error: %s", str(e))
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
                SetDepthState(depth=gate_depth, sleep_duration=5.0),
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
                    "succeeded": "PLAN_GATE_PATHS",
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
                "EXECUTE_GATE_PATHS",
                ExecuteGatePathsState(),
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
        outcome = self.state_machine.execute()
        return outcome
