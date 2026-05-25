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
    """Generic SetBool wrapper for a trajectory publisher's
    frame-publishing toggle."""

    def __init__(self, service_name: str, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            service_name,
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ValveTaskState(smach.State):
    """Aligns to a valve in two phases: approach → engage.

    A single trajectory publisher node (one per valve) emits both target
    frames simultaneously (approach + engage).  All frames are enabled
    once at the start via a single service call.

    Approach phase uses a Concurrence to stabilise the valve TF:
      1. AlignFrame drives toward the approach target.
      2. After a settle delay the publisher is disabled ("frozen") — the
         object map keeps broadcasting the last-computed TF, giving the
         alignment a rock-solid static target to converge on.
      When AlignFrame succeeds the Concurrence exits, regardless of
      whether the freeze has fired yet.

    After approach, the publisher is re-enabled (unfrozen) and the
    engage phase aligns the valve gripper link to the engage target.

    Parameterised so a single class covers both front and bottom valves —
    they differ only in the frame names and gripper link.
    """

    def __init__(
        self,
        valve_depth,
        gripper_frame: str = "taluy/base_link/valve_gripper_front_link",
        approach_target_frame: str = "valve_front_approach_target",
        engage_target_frame: str = "valve_front_engage_target",
        publisher_service: str = "set_publishing_valve_front",
        freeze_delay: float = 5.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        base_link_frame = "taluy/base_link"

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.state_machine.set_initial_state(["SET_VALVE_DEPTH"])

        # -- Build the approach Concurrence --
        # Child 1: align to approach target (loose thresholds, long timeout)
        # Child 2: delay → freeze publisher (disable it so TF stops updating)
        # When AlignFrame succeeds, the whole Concurrence exits.
        approach_concurrence = smach.Concurrence(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            outcome_map={
                "succeeded": {"ALIGN_TO_APPROACH": "succeeded"},
                "preempted": {"ALIGN_TO_APPROACH": "preempted"},
                "aborted": {"ALIGN_TO_APPROACH": "aborted"},
            },
            child_termination_cb=lambda outcome_map: True,
        )
        with approach_concurrence:
            smach.Concurrence.add(
                "ALIGN_TO_APPROACH",
                AlignFrame(
                    source_frame=base_link_frame,
                    target_frame=approach_target_frame,
                    dist_threshold=0.15,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    use_frame_depth=True,
                ),
            )

            freeze_sequence = smach.StateMachine(
                outcomes=["succeeded", "preempted", "aborted"]
            )
            with freeze_sequence:
                smach.StateMachine.add(
                    "WAIT_FOR_SETTLE",
                    DelayState(delay_time=freeze_delay),
                    transitions={
                        "succeeded": "FREEZE_PUBLISHER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    "FREEZE_PUBLISHER",
                    _SetBoolServiceState(publisher_service, req=False),
                    transitions={
                        "succeeded": "WAIT_FOREVER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                # Sit here until the Concurrence kills us.
                smach.StateMachine.add(
                    "WAIT_FOREVER",
                    DelayState(delay_time=999.0),
                    transitions={
                        "succeeded": "succeeded",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.Concurrence.add("FREEZE_SEQUENCE", freeze_sequence)

        # -- Wire the top-level state machine --
        with self.state_machine:
            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth),
                transitions={
                    "succeeded": "ENABLE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Enable all trajectory frames at once.
            smach.StateMachine.add(
                "ENABLE_PUBLISHER",
                _SetBoolServiceState(publisher_service, req=True),
                transitions={
                    "succeeded": "WAIT_FOR_FRAMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRAMES",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "APPROACH_WITH_FREEZE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Approach phase (with concurrent freeze) --------
            smach.StateMachine.add(
                "APPROACH_WITH_FREEZE",
                approach_concurrence,
                transitions={
                    "succeeded": "UNFREEZE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Re-enable publisher so engage sees live TF.
            smach.StateMachine.add(
                "UNFREEZE_PUBLISHER",
                _SetBoolServiceState(publisher_service, req=True),
                transitions={
                    "succeeded": "ALIGN_TO_ENGAGE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Engage phase --------
            # Gripper frame aligns to the engage target so the gripper
            # tip (not base_link centre) lands on the valve.
            smach.StateMachine.add(
                "ALIGN_TO_ENGAGE",
                AlignFrame(
                    source_frame=gripper_frame,
                    target_frame=engage_target_frame,
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
                    "succeeded": "DISABLE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "DISABLE_PUBLISHER",
                _SetBoolServiceState(publisher_service, req=False),
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
