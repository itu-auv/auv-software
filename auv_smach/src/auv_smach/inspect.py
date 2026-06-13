import math
import rospy
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
    """Toggle the inspect frame publisher. The valve side ('left'/'right')
    picks which of the node's two enable services is called, which in turn
    selects the mirrored rectangle geometry on the publisher side.
    """

    def __init__(self, req: bool, valve_side: str):
        smach_ros.ServiceState.__init__(
            self,
            f"set_transform_inspect_frames_{valve_side}",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class SetKeypointNodeEnabledState(smach_ros.ServiceState):
    """Toggle a valve_keypoint_node instance via its ~set_enabled service."""

    def __init__(self, service_name: str, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            service_name,
            SetBool,
            request=SetBoolRequest(data=req),
        )


class SetArucoDetectionEnabledState(smach_ros.ServiceState):
    """Toggle the ArUco detection node via its ~set_enabled service.

    The node launches dormant (start_enabled=false in tac_sea.launch); this
    switches it on for the duration of the inspection scan.
    """

    def __init__(self, service_name: str, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            service_name,
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
        keypoint_node_services=(
            "valve_keypoint_node_front/set_enabled",
            "valve_keypoint_node_bottom/set_enabled",
        ),
        aruco_enable_service="aruco_detection/set_enabled",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Which side of the desk the valve is on. Required, no default: the
        # operator must launch with valve_pos:=left|right. The smach calls the
        # matching enable service so the publisher mirrors its geometry.
        valve_side = rospy.get_param("~valve_pos")
        if valve_side not in ("left", "right"):
            raise ValueError(
                f"~valve_pos must be 'left' or 'right', got '{valve_side}'"
            )

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

        keypoint_node_services = list(keypoint_node_services)

        # First state after enabling the ArUco detector: the keypoint-disable
        # chain if any, otherwise straight to the inspect publisher.
        first_after_aruco = (
            "DISABLE_KEYPOINT_0"
            if keypoint_node_services
            else "ENABLE_INSPECT_PUBLISHER"
        )

        with self.state_machine:
            # ArUco scan-only detector launches dormant; enable it for the
            # inspection so it produces marker detections/debug images.
            smach.StateMachine.add(
                "ENABLE_ARUCO_DETECTION",
                SetArucoDetectionEnabledState(aruco_enable_service, req=True),
                transitions={
                    "succeeded": first_after_aruco,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Turn the valve keypoint producers off for the duration of the
            # inspection, then back on again at the end (see re-enable chain
            # after DISABLE_INSPECT_PUBLISHER).
            for i, service_name in enumerate(keypoint_node_services):
                next_state = (
                    f"DISABLE_KEYPOINT_{i + 1}"
                    if i + 1 < len(keypoint_node_services)
                    else "ENABLE_INSPECT_PUBLISHER"
                )
                smach.StateMachine.add(
                    f"DISABLE_KEYPOINT_{i}",
                    SetKeypointNodeEnabledState(service_name, req=False),
                    transitions={
                        "succeeded": next_state,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.StateMachine.add(
                "ENABLE_INSPECT_PUBLISHER",
                InspectFramesPublisherServiceState(req=True, valve_side=valve_side),
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
                InspectFramesPublisherServiceState(req=False, valve_side=valve_side),
                transitions={
                    "succeeded": "ENABLE_KEYPOINT_0",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Re-enable the valve keypoint producers now that inspection is done.
            for i, service_name in enumerate(keypoint_node_services):
                next_state = (
                    f"ENABLE_KEYPOINT_{i + 1}"
                    if i + 1 < len(keypoint_node_services)
                    else "CANCEL_ALIGN_CONTROLLER"
                )
                smach.StateMachine.add(
                    f"ENABLE_KEYPOINT_{i}",
                    SetKeypointNodeEnabledState(service_name, req=True),
                    transitions={
                        "succeeded": next_state,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "DISABLE_ARUCO_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Inspection done: return the ArUco detector to its dormant
            # launch state (mirrors ENABLE_ARUCO_DETECTION at the start).
            smach.StateMachine.add(
                "DISABLE_ARUCO_DETECTION",
                SetArucoDetectionEnabledState(aruco_enable_service, req=False),
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
