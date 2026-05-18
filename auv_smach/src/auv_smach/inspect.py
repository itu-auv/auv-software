import math
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    LookAroundState,
)
from auv_smach.initialize import DelayState


class InspectFramesPublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_inspect_frames",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class InspectTaskState(smach.State):
    def __init__(
        self,
        sweep_angle: float = math.pi / 2,
        num_points: int = 8,
        align_timeout: float = 40.0,
        align_dist_threshold: float = 0.15,
        align_yaw_threshold: float = 0.15,
        align_confirm_duration: float = 1.0,
        publisher_warmup_delay: float = 1.5,
        scan_timeout: float = 15.0,
        scan_confirm_duration: float = 0.2,
        scan_max_linear_velocity: float = 0.1,
        scan_max_angular_velocity: float = 0.25,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        base_link_frame = "taluy/base_link"
        # LookAroundState's angle_offset is half the total sweep (it swings
        # +offset, -offset, 0), so halve the user's full sweep here.
        half_sweep = sweep_angle / 2.0

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        def align_to(target_frame):
            return AlignFrame(
                source_frame=base_link_frame,
                target_frame=target_frame,
                dist_threshold=align_dist_threshold,
                yaw_threshold=align_yaw_threshold,
                confirm_duration=align_confirm_duration,
                timeout=align_timeout,
                cancel_on_success=False,
                use_frame_depth=True,
            )

        def scan_at(index):
            return LookAroundState(
                source_frame=base_link_frame,
                angle_offset=half_sweep,
                confirm_duration=scan_confirm_duration,
                timeout=scan_timeout,
                max_linear_velocity=scan_max_linear_velocity,
                max_angular_velocity=scan_max_angular_velocity,
                current_pose_frame=f"inspect_scan_anchor_{index}",
            )

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_INSPECT_PUBLISHER",
                InspectFramesPublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_FRAMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRAMES",
                DelayState(delay_time=publisher_warmup_delay),
                transitions={
                    "succeeded": "ALIGN_TO_START_TOP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_START_TOP",
                align_to("inspect_start_top"),
                transitions={
                    "succeeded": "ALIGN_TO_FRAME_0_TOP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_FRAME_0_TOP",
                align_to("inspect_frame_0_top"),
                transitions={
                    "succeeded": "ALIGN_TO_FRAME_0",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            for i in range(num_points):
                align_state = f"ALIGN_TO_FRAME_{i}"
                scan_state = f"SCAN_AT_FRAME_{i}"
                after_scan = (
                    f"ALIGN_TO_FRAME_{i + 1}"
                    if i + 1 < num_points
                    else "DISABLE_INSPECT_PUBLISHER"
                )

                smach.StateMachine.add(
                    align_state,
                    align_to(f"inspect_frame_{i}"),
                    transitions={
                        "succeeded": scan_state,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    scan_state,
                    scan_at(i),
                    transitions={
                        "succeeded": after_scan,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.StateMachine.add(
                "DISABLE_INSPECT_PUBLISHER",
                InspectFramesPublisherServiceState(req=False),
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
