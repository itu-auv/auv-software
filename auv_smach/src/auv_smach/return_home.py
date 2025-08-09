from .initialize import *
import smach
import rospy
from auv_smach.common import (
    SetDepthState,
    VisualServoingCentering,
    VisualServoingNavigation,
)
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest
from auv_smach.initialize import DelayState


class ReturnHomeState(smach.State):
    def __init__(self, gate_depth: float, target_prop: str, wait_duration: float = 6.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        # Open the container for adding states
        with self.state_machine:

            smach.StateMachine.add(
                "RETURN_SET_GATE_DEPTH",
                SetDepthState(
                    depth=gate_depth,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 5.0),
                ),
                transitions={
                    "succeeded": "RETURN_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RETURN_CENTERING",
                VisualServoingCentering(target_prop=target_prop),
                transitions={
                    "succeeded": "RETURN_WAIT_FOR_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RETURN_WAIT_FOR_CENTERING",
                DelayState(delay_time=wait_duration),
                transitions={
                    "succeeded": "RETURN_NAVIGATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RETURN_NAVIGATION",
                VisualServoingNavigation(),
                transitions={
                    "succeeded": "RETURN_WAIT_FOR_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RETURN_WAIT_FOR_EXIT",
                DelayState(delay_time=60.0),
                transitions={
                    "succeeded": "RETURN_CANCEL_NAVIGATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RETURN_CANCEL_NAVIGATION",
                VisualServoingCancelNavigation(),
                transitions={
                    "succeeded": "RETURN_CANCEL_VISUAL_SERVOING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RETURN_CANCEL_VISUAL_SERVOING",
                VisualServoingCancel(),
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


class VisualServoingCancelNavigation(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingCancelNavigation, self).__init__(
            "visual_servoing/cancel_navigation",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )


class VisualServoingCancel(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingCancel, self).__init__(
            "visual_servoing/cancel",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )
