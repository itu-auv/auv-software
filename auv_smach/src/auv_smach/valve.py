from .initialize import *
import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_msgs.srv import RotateGripper, RotateGripperRequest
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


def _rotate_gripper_response_cb(userdata, response):
    """Map the service-level success flag onto the smach outcome (a refusal —
    e.g. not enough headroom — must abort the task, not 'succeed')."""
    if response.success:
        return "succeeded"
    rospy.logerr(f"RotateGripper service refused: {response.message}")
    return "aborted"


class SetGripperTurnDirectionState(smach_ros.ServiceState):
    """Tell a gripper roll tracker which way the valve handle will be turned
    ("cw"/"ccw" as seen facing the valve). From then on the tracker only parks
    the servo on branches with enough headroom to make that turn without a
    mid-grip finger flip. Call once, early — turn-readiness is a standing
    property of tracking, not a pre-engage ritual."""

    def __init__(
        self, tracker_namespace: str, direction: str, headroom_deg: float = 0.0
    ):
        smach_ros.ServiceState.__init__(
            self,
            f"{tracker_namespace}/set_turn_direction",
            RotateGripper,
            request=RotateGripperRequest(direction=direction, degrees=headroom_deg),
            response_cb=_rotate_gripper_response_cb,
        )


class RotateGripperState(smach_ros.ServiceState):
    """Turn the gripped valve handle: latches the tracked roll and ramps the
    servo by `degrees` in `direction` (handle sense, facing the valve). The
    service blocks until the ramp completes, so this state returns when the
    turn is physically done. Refusals (insufficient headroom, nothing tracked
    yet, turn already in flight) come back as 'aborted'."""

    def __init__(self, tracker_namespace: str, direction: str, degrees: float):
        smach_ros.ServiceState.__init__(
            self,
            f"{tracker_namespace}/rotate",
            RotateGripper,
            request=RotateGripperRequest(direction=direction, degrees=degrees),
            response_cb=_rotate_gripper_response_cb,
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

    If `turn_direction` is set ("cw"/"ccw", handle sense facing the valve),
    the gripper roll tracker is told the direction up front (so tracking
    stays turn-ready with enough servo headroom) and, after engaging, the
    handle is turned `turn_degrees` while the align controller holds the
    engage pose.

    Parameterised so a single class covers both front and bottom valves —
    they differ only in the frame names, gripper link, and tracker namespace.
    """

    def __init__(
        self,
        valve_depth,
        gripper_frame: str = "taluy/base_link/valve_gripper_front_link",
        approach_target_frame: str = "valve_front_approach_target",
        engage_target_frame: str = "valve_front_engage_target",
        publisher_service: str = "set_publishing_valve_front",
        freeze_delay: float = 5.0,
        tracker_namespace: str = "gripper_roll_tracker_front",
        turn_direction: str = "",
        turn_degrees: float = 90.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.state_machine.set_initial_state(
            ["SET_TURN_DIRECTION" if turn_direction else "SET_VALVE_DEPTH"]
        )

        # -- Build the approach Concurrence --
        # Child 1: align to approach target (loose thresholds, long timeout)
        # Child 2: delay → freeze publisher (disable it so TF stops updating)
        # The Concurrence exits only when ALIGN_TO_APPROACH finishes; the
        # freeze branch runs to completion (or is preempted if alignment
        # finishes before the settle delay) without ending the Concurrence.
        approach_concurrence = smach.Concurrence(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            outcome_map={
                "succeeded": {"ALIGN_TO_APPROACH": "succeeded"},
                "preempted": {"ALIGN_TO_APPROACH": "preempted"},
                "aborted": {"ALIGN_TO_APPROACH": "aborted"},
            },
            child_termination_cb=lambda outcomes: outcomes["ALIGN_TO_APPROACH"]
            is not None,
        )
        with approach_concurrence:
            smach.Concurrence.add(
                "ALIGN_TO_APPROACH",
                AlignFrame(
                    source_frame=gripper_frame,
                    target_frame=approach_target_frame,
                    dist_threshold=0.05,
                    yaw_threshold=0.01,
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
                # Once frozen, this branch is done — it just returns
                # "succeeded" and idles while ALIGN_TO_APPROACH keeps running;
                # the Concurrence's child_termination_cb ignores this outcome.
                smach.StateMachine.add(
                    "FREEZE_PUBLISHER",
                    _SetBoolServiceState(publisher_service, req=False),
                    transitions={
                        "succeeded": "succeeded",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.Concurrence.add("FREEZE_SEQUENCE", freeze_sequence)

        # -- Wire the top-level state machine --
        with self.state_machine:
            # Tell the roll tracker the turn direction first: from here on,
            # tracking permanently keeps enough headroom for the turn.
            if turn_direction:
                smach.StateMachine.add(
                    "SET_TURN_DIRECTION",
                    SetGripperTurnDirectionState(
                        tracker_namespace=tracker_namespace,
                        direction=turn_direction,
                        headroom_deg=turn_degrees,
                    ),
                    transitions={
                        "succeeded": "SET_VALVE_DEPTH",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

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
                    max_linear_velocity=0.05,
                    max_angular_velocity=0.05,
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
                    "succeeded": (
                        "ROTATE_VALVE" if turn_direction else "CANCEL_ALIGN_CONTROLLER"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Turn phase --------
            # The align controller is still holding the engage pose; the
            # rotate service latches the tracked roll and ramps the servo.
            # It blocks until the turn is physically complete.
            if turn_direction:
                smach.StateMachine.add(
                    "ROTATE_VALVE",
                    RotateGripperState(
                        tracker_namespace=tracker_namespace,
                        direction=turn_direction,
                        degrees=turn_degrees,
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
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
