from .initialize import *
import smach
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest

from auv_smach.common import (
    SetDepthState,
    NavigateToFrameState,
)

from auv_smach.search_states import SearchForPropState

from auv_smach.alignment_states import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
)


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
                    full_rotation=True,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
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
                    "succeeded": "PUBLISH_GATE_ANGLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PUBLISH_GATE_ANGLE",
                PublishGateAngleState(),
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
                    "succeeded": "NAVIGATE_TO_GATE_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_GATE_EXIT",
                NavigateToFrameState(
                    start_frame="gate_entrance",
                    target_frame="dynamic_target",
                    end_frame="gate_exit",
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

        # Execute the state machine

        outcome = self.state_machine.execute()

        if outcome is None:  # ctrl + c
            return "preempted"
        # Return the outcome of the state machine
        return outcome
