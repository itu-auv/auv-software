from .initialize import *
import smach
import rospy
from auv_smach.common import (
    SetDepthState,
    VisualServoingCentering,
    VisualServoingNavigation,
)


class NavigateThroughGateStateVS(smach.State):
    def __init__(self, gate_depth: float, target_prop: str, wait_duration: float = 6.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:

            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(
                    depth=gate_depth,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 5.0),
                ),
                transitions={
                    "succeeded": "VISUAL_SERVOING_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_CENTERING",
                VisualServoingCentering(target_prop=target_prop),
                transitions={
                    "succeeded": "WAIT_FOR_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_CENTERING",
                DelayState(delay_time=wait_duration),
                transitions={
                    "succeeded": "VISUAL_SERVOING_NAVIGATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_NAVIGATION",
                VisualServoingNavigation(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughGateStateVS] Starting state machine execution.")

        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:  # ctrl + c
            return "preempted"
        # Return the outcome of the state machine
        return outcome
