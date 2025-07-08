from .initialize import *
import smach
from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
)
from auv_smach.initialize import DelayState


class OctagonTaskState(smach.State):
    def __init__(self, octagon_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        set_octagon_depth_params = rospy.get_param("~set_octagon_depth", {})
        wait_for_surfacing_params = rospy.get_param("~wait_for_surfacing", {})
        surfacing_params = rospy.get_param("~surfacing", {})

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(
                    depth=octagon_depth,
                    sleep_duration=set_octagon_depth_params.get("sleep_duration", 4.0),
                ),
                transitions={
                    "succeeded": "SET_OCTAGON_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="octagon_target",
                ),
                transitions={
                    "succeeded": "APPROACH_TO_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "APPROACH_TO_OCTAGON",
                NavigateToFrameState(
                    "taluy/base_link", "octagon_link", "octagon_target"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_SURFACING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SURFACING",
                DelayState(delay_time=wait_for_surfacing_params.get("delay_time", 3.0)),
                transitions={
                    "succeeded": "SURFACING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACING",
                SetDepthState(
                    depth=surfacing_params.get("depth", 0.3),
                    sleep_duration=surfacing_params.get("sleep_duration", 4.0),
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
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
