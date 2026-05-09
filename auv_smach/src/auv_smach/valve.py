import math
from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SetDepthState,
)
from auv_smach.initialize import DelayState


class ValveApproachFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_approach_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ValveContactFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_contact_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ValveKeypointNodeEnableServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_valve_keypoint_enabled",
            SetBool,
            request=SetBoolRequest(data=req),
        )


def _build_approach_freeze_timer_sm(delay: float) -> smach.StateMachine:
    sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])
    with sm:
        smach.StateMachine.add(
            "WAIT",
            DelayState(delay_time=delay),
            transitions={
                "succeeded": "FREEZE",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        smach.StateMachine.add(
            "FREEZE",
            ValveApproachFramePublisherServiceState(req=False),
            transitions={
                "succeeded": "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
    return sm


class ValveTaskState(smach.State):
    def __init__(
        self,
        valve_depth,
        valve_approach_frame: str = "valve_approach_frame",
        valve_contact_frame: str = "valve_contact_frame",
        valve_frame: str = "tac/valve",
        approach_freeze_delay: float = 4.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        gripper_frame = "taluy/base_link/valve_gripper_front_link"
        base_link_frame = "taluy/base_link"

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.state_machine.set_initial_state(["SET_VALVE_DEPTH"])

        align_and_freeze_approach = smach.Concurrence(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            child_termination_cb=lambda om: om.get("ALIGN") is not None,
            outcome_cb=lambda om: om.get("ALIGN", "aborted"),
        )
        with align_and_freeze_approach:
            smach.Concurrence.add(
                "ALIGN",
                AlignFrame(
                    source_frame=base_link_frame,
                    target_frame=valve_approach_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    use_frame_depth=True,
                ),
            )
            smach.Concurrence.add(
                "FREEZE_TIMER",
                _build_approach_freeze_timer_sm(approach_freeze_delay),
            )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth),
                transitions={
                    "succeeded": "ENABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_APPROACH_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_AND_FREEZE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_AND_FREEZE_APPROACH",
                align_and_freeze_approach,
                transitions={
                    "succeeded": "DISABLE_VALVE_KEYPOINT_NODE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_VALVE_KEYPOINT_NODE",
                ValveKeypointNodeEnableServiceState(req=False),
                transitions={
                    "succeeded": "DISABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ENABLE_CONTACT_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_CONTACT_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_CONTACT_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_CONTACT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_CONTACT",
                AlignFrame(
                    source_frame=gripper_frame,
                    target_frame=valve_contact_frame,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=5.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "DISABLE_CONTACT_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ALIGN_TO_VALVE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # angle_offset=pi flips heading: tac/valve's X is the outward normal, we want to face into it.
            smach.StateMachine.add(
                "ALIGN_TO_VALVE",
                AlignFrame(
                    source_frame=gripper_frame,
                    target_frame=valve_frame,
                    angle_offset=math.pi,
                    dist_threshold=0.03,
                    yaw_threshold=0.05,
                    confirm_duration=3.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "REENABLE_VALVE_KEYPOINT_NODE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "REENABLE_VALVE_KEYPOINT_NODE",
                ValveKeypointNodeEnableServiceState(req=True),
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
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
