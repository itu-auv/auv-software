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


class _SetBoolServiceState(smach_ros.ServiceState):
    """Generic SetBool wrapper — used for both the trajectory publisher's
    approach-frame toggle and the keypoint node's enable toggle."""

    def __init__(self, service_name: str, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            service_name,
            SetBool,
            request=SetBoolRequest(data=req),
        )


def _build_approach_freeze_timer_sm(
    delay: float, approach_publisher_service: str
) -> smach.StateMachine:
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
            _SetBoolServiceState(approach_publisher_service, req=False),
            transitions={
                "succeeded": "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
    return sm


class ValveTaskState(smach.State):
    """Aligns to a valve and drives the gripper onto it.

    Parameterised so a single class can run against either the front or
    bottom valve — both share the keypoint pipeline; only the frames and
    per-instance service names differ.

    `align_angle_offset` is forwarded to the final AlignFrame: the front
    valve uses π so the AUV faces *into* the panel (its TF's +X is the
    outward flange normal). The bottom valve's geometry is different and
    the value is a placeholder until the bottom approach is tuned.
    """

    def __init__(
        self,
        valve_depth,
        valve_approach_frame: str = "valve_front_approach_frame",
        valve_frame: str = "tac/valve_front",
        gripper_frame: str = "taluy/base_link/valve_gripper_front_link",
        keypoint_enable_service: str = "valve_keypoint_node_front/set_enabled",
        approach_publisher_service: str = "set_transform_valve_front_approach_frame",
        align_angle_offset: float = math.pi,
        align_pitch_offset: float = 0.0,
        approach_freeze_delay: float = 4.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

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
                _build_approach_freeze_timer_sm(
                    approach_freeze_delay, approach_publisher_service
                ),
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
                _SetBoolServiceState(approach_publisher_service, req=True),
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
                _SetBoolServiceState(keypoint_enable_service, req=False),
                transitions={
                    "succeeded": "DISABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_APPROACH_PUBLISHER",
                _SetBoolServiceState(approach_publisher_service, req=False),
                transitions={
                    "succeeded": "ALIGN_TO_VALVE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_VALVE",
                AlignFrame(
                    source_frame=gripper_frame,
                    target_frame=valve_frame,
                    angle_offset=align_angle_offset,
                    pitch_offset=align_pitch_offset,
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
                _SetBoolServiceState(keypoint_enable_service, req=True),
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
