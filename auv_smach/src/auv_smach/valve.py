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
    """Aligns to a valve in three phases: look → approach → engage.

    Each trajectory publisher emits a target frame that is:
      - Positioned along the valve's outward normal at a configured offset
      - Oriented level (roll=0, pitch=0) and yaw facing the valve

    The smach side is just: enable phase publisher, align base_link to
    its target frame, repeat for next phase.

    Phases:
      - look:     large offset — AUV sits back while keypoints settle
                  and tac/valve_* stabilises.
      - approach: medium offset — AUV moves to a standoff in front of
                  the valve.
      - engage:   zero offset — AUV docks the gripper onto the valve.

    Parameterised so a single class covers both front and bottom valves —
    they differ only in the frame names and per-phase service names.
    """

    def __init__(
        self,
        valve_depth,
        look_target_frame: str = "valve_front_look_target",
        approach_target_frame: str = "valve_front_approach_target",
        engage_target_frame: str = "valve_front_engage_target",
        look_publisher_service: str = "set_publishing_valve_front_look",
        approach_publisher_service: str = "set_publishing_valve_front_approach",
        engage_publisher_service: str = "set_publishing_valve_front_engage",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        base_link_frame = "taluy/base_link"

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.state_machine.set_initial_state(["SET_VALVE_DEPTH"])

        with self.state_machine:
            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth),
                transitions={
                    "succeeded": "ENABLE_LOOK_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Look phase --------
            # Sit back, point the camera at the valve, let keypoints settle.
            # Loose thresholds and a short confirm — we just need the
            # keypoint pipeline to lock on, not a precise pose.
            smach.StateMachine.add(
                "ENABLE_LOOK_PUBLISHER",
                _SetBoolServiceState(look_publisher_service, req=True),
                transitions={
                    "succeeded": "WAIT_FOR_LOOK_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_LOOK_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_LOOK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_LOOK",
                AlignFrame(
                    source_frame=base_link_frame,
                    target_frame=look_target_frame,
                    dist_threshold=0.3,
                    yaw_threshold=0.15,
                    confirm_duration=2.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "DISABLE_LOOK_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_LOOK_PUBLISHER",
                _SetBoolServiceState(look_publisher_service, req=False),
                transitions={
                    "succeeded": "ENABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Approach phase --------
            # Standoff in front of the valve, gripper axis pointed at the
            # valve. Tighter than look, looser than engage.
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
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "ALIGN_TO_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
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
                    "succeeded": "ENABLE_ENGAGE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Engage phase --------
            # Zero offset along the flange normal — co-located with the
            # valve, oriented so the gripper engages it. Tight thresholds
            # and low velocity for a clean dock.
            smach.StateMachine.add(
                "ENABLE_ENGAGE_PUBLISHER",
                _SetBoolServiceState(engage_publisher_service, req=True),
                transitions={
                    "succeeded": "WAIT_FOR_ENGAGE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ENGAGE_FRAME",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "ALIGN_TO_ENGAGE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_ENGAGE",
                AlignFrame(
                    source_frame=base_link_frame,
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
                    "succeeded": "DISABLE_ENGAGE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_ENGAGE_PUBLISHER",
                _SetBoolServiceState(engage_publisher_service, req=False),
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
