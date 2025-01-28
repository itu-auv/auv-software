import smach

from auv_smach.common import (
    CancelAlignControllerState,
    ClearObjectTransformsState,
    ClearPropTransformsState,
)


class DeinitializeState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "CLEAR_OBJECT_TRANSFORMS",
                    "preempted": "CLEAR_OBJECT_TRANSFORMS",
                    "aborted": "CLEAR_OBJECT_TRANSFORMS",
                },
            )
            smach.StateMachine.add(
                "CLEAR_OBJECT_TRANSFORMS",
                ClearObjectTransformsState(),
                transitions={
                    "succeeded": "CLEAR_PROP_TRANSFORMS",
                    "preempted": "CLEAR_PROP_TRANSFORMS",
                    "aborted": "CLEAR_PROP_TRANSFORMS",
                },
            )
            smach.StateMachine.add(
                "CLEAR_PROP_TRANSFORMS",
                ClearPropTransformsState(),
                transitions={
                    "succeeded": "aborted",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        # Return the outcome of the state machine
        return outcome
