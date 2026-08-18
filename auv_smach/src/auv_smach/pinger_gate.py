import threading

import rospy
import smach
import smach_ros
from auv_msgs.srv import SetString, SetStringRequest
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyRequest, SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    CheckForTransformState,
    DynamicPathState,
    DynamicPathWithTransformCheck,
    SearchForPropState,
    SetAltitudeState,
    SetDetectionState,
)
from auv_smach.initialize import DelayState
from auv_smach.tf_utils import get_base_link


def require_success(_userdata, response):
    if response.success:
        rospy.loginfo("Pinger-gate service succeeded: %s", response.message)
        return "succeeded"
    rospy.logerr("Service rejected pinger-gate transition: %s", response.message)
    return "aborted"


class PingerTrajectoryPublisherState(smach_ros.ServiceState):
    def __init__(self, enable: bool):
        super().__init__(
            "toggle_pinger_teknofest_trajectory",
            SetBool,
            request=SetBoolRequest(data=enable),
        )


class VitposeDetectionState(smach_ros.ServiceState):
    """Enable or pause the pose-producing ViTPose node."""

    def __init__(self, enable: bool):
        super().__init__(
            "vitpose_detection_node/enable",
            SetBool,
            request=SetBoolRequest(data=enable),
            response_cb=require_success,
        )


class VitposeConfigState(smach_ros.ServiceState):
    def __init__(self, object_name: str):
        super().__init__(
            "vitpose_detection_node/set_config",
            SetString,
            request=SetStringRequest(data=object_name),
            response_cb=require_success,
        )


class VitposeModeState(smach_ros.ServiceState):
    """Set the ViTPose detection node mode to off, detect, or pose."""

    VALID_MODES = {"off", "detect", "pose"}

    def __init__(self, mode: str):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid ViTPose mode '{mode}'; expected one of "
                f"{sorted(self.VALID_MODES)}"
            )

        super().__init__(
            "vitpose_detection_node/set_mode",
            SetString,
            request=SetStringRequest(data=mode),
            response_cb=require_success,
        )


class VitposeScanState(smach_ros.ServiceState):
    def __init__(self, enable: bool):
        super().__init__(
            "vitpose_scan_node/enable",
            SetBool,
            request=SetBoolRequest(data=enable),
            response_cb=require_success,
        )


class TetraFrontCameraState(smach_ros.ServiceState):
    def __init__(self, enable: bool):
        super().__init__(
            "enable_tetra_front_camera_detections",
            SetBool,
            request=SetBoolRequest(data=enable),
            response_cb=require_success,
        )


class ResetTetraUnfoldState(smach_ros.ServiceState):
    def __init__(self):
        super().__init__(
            "tetra_unfold/reset",
            Empty,
            request=EmptyRequest(),
        )


class WaitForTetraUnfoldState(smach.State):
    """Wait for a fresh, locked unfold result and print the mission payload."""

    def __init__(self, topic="tetra/letter_colors", timeout=30.0):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])
        self.timeout = float(timeout)
        self.condition = threading.Condition()
        self.sequence = 0
        self.latest = None
        self.subscriber = rospy.Subscriber(topic, String, self._callback, queue_size=1)

    def _callback(self, message):
        with self.condition:
            self.sequence += 1
            self.latest = message.data
            self.condition.notify_all()

    def execute(self, _userdata):
        with self.condition:
            seen_sequence = self.sequence
        deadline = rospy.Time.now() + rospy.Duration(self.timeout)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            with self.condition:
                if self.sequence <= seen_sequence:
                    self.condition.wait(timeout=0.1)
                    continue
                seen_sequence = self.sequence
                result = self.latest

            rospy.loginfo("[TetraUnfold] %s", result)
            if result and "state=LOCKED" in result:
                return "succeeded"

        rospy.logwarn("Timed out waiting for a fresh LOCKED tetra unfold result")
        return "aborted"


class PingerGateTaskState(smach.State):
    """Initial, bbox-based approach phase of the Teknofest pinger gate task."""

    def __init__(
        self,
        pinger_bbox_frame: str = "pinger_bbox",
        close_approach_frame: str = "pinger_close_approach",
        gate_closer_frame: str = "gate_closer",
        gate_farther_frame: str = "gate_farther",
        gate_farthest_frame: str = "gate_farthest",
        tetra_forward_search: bool = False,
        tetra_front_frame: str = "tetra_further_link",
        tetra_bottom_frame: str = "tetra_bottom_link",
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        tetra_search_start = (
            "ENABLE_TETRA_FRONT_SCAN"
            if tetra_forward_search
            else "SET_VITPOSE_CONFIG_TETRA"
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_ALTITUDE",
                SetAltitudeState(altitude=1.3),
                transitions={
                    "succeeded": "ENABLE_PINGER_CAMERA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_PINGER_CAMERA",
                SetDetectionState(camera_name="pinger", enable=True),
                transitions={
                    "succeeded": "ENABLE_PINGER_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_PINGER_TRAJECTORY",
                PingerTrajectoryPublisherState(enable=True),
                transitions={
                    "succeeded": "WAIT_FOR_PINGER_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_PINGER_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_PINGER",
                SearchForPropState(
                    look_at_frame=pinger_bbox_frame,
                    alignment_frame="pinger_search",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.4,
                ),
                transitions={
                    "succeeded": "PATH_TO_PINGER_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_TO_PINGER_CLOSE_APPROACH",
                DynamicPathState(plan_target_frame=close_approach_frame),
                transitions={
                    "succeeded": "ALIGN_TO_PINGER_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_PINGER_CLOSE_APPROACH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=close_approach_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=10.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_VITPOSE_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ##CLOSE APPROCH SEQUANCE'I BİTİYOR ÜSTÜNÜ SİLEBİLİRİZ EN KÖTÜ.
            smach.StateMachine.add(
                "ENABLE_VITPOSE_DETECTION",
                VitposeDetectionState(enable=True),
                transitions={
                    "succeeded": "WAIT_FOR_GATE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_GATE_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "DISABLE_PINGER_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_PINGER_TRAJECTORY",
                PingerTrajectoryPublisherState(enable=False),
                transitions={
                    "succeeded": "ALIGN_TO_GATE_CLOSER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_GATE_CLOSER",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=gate_closer_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=10.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "PATH_THROUGH_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_THROUGH_GATE",
                DynamicPathState(
                    plan_target_frame=gate_farther_frame,
                    max_linear_velocity=0.2,
                ),
                transitions={
                    "succeeded": tetra_search_start,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_VITPOSE_CONFIG_TETRA",
                VitposeConfigState(object_name="tetra"),
                transitions={
                    "succeeded": "ENABLE_TETRA_BOTTOM_PIPELINE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TETRA_BOTTOM_PIPELINE",
                VitposeDetectionState(enable=True),
                transitions={
                    "succeeded": "PATH_GATE_FARTHEST_UNTIL_BOTTOM_TETRA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_GATE_FARTHEST_UNTIL_BOTTOM_TETRA",
                DynamicPathWithTransformCheck(
                    plan_target_frame=gate_farthest_frame,
                    transform_source_frame="odom",
                    transform_target_frame=tetra_bottom_frame,
                    max_linear_velocity=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_TETRA_BOTTOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ##---------------------------------------Front Scan Smach

            smach.StateMachine.add(
                "ENABLE_TETRA_FRONT_SCAN",
                VitposeScanState(enable=True),
                transitions={
                    "succeeded": "ENABLE_TETRA_FRONT_CAMERA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TETRA_FRONT_CAMERA",
                TetraFrontCameraState(enable=True),
                transitions={
                    "succeeded": "WAIT_FOR_FRONT_TETRA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRONT_TETRA",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "close_before_allinging_to_Exit",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "close_before_allinging_to_Exit",
                PingerTrajectoryPublisherState(enable=False),
                transitions={
                    "succeeded": "alling_to_exit",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "alling_to_exit",
                DynamicPathState(plan_target_frame=gate_farther_frame),
                transitions={
                    "succeeded": "WAIT_FOR_FRONT_TETRA_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRONT_TETRA_FRAME",
                CheckForTransformState(
                    source_frame="odom",
                    target_frame=tetra_front_frame,
                    timeout=15.0,
                ),
                transitions={
                    "succeeded": "PATH_FRONT_TETRA_UNTIL_BOTTOM_TETRA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_FRONT_TETRA_UNTIL_BOTTOM_TETRA",
                DynamicPathWithTransformCheck(
                    plan_target_frame=tetra_front_frame,
                    transform_source_frame="odom",
                    transform_target_frame=tetra_bottom_frame,
                    max_linear_velocity=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_TETRA_BOTTOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TETRA_BOTTOM",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=tetra_bottom_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=30.0,
                    keep_orientation=True,
                    max_linear_velocity=0.15,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "RESET_TETRA_UNFOLD",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESET_TETRA_UNFOLD",
                ResetTetraUnfoldState(),
                transitions={
                    "succeeded": "WAIT_AND_PRINT_TETRA_UNFOLD",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_AND_PRINT_TETRA_UNFOLD",
                WaitForTetraUnfoldState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def request_preempt(self):
        smach.State.request_preempt(self)
        self.state_machine.request_preempt()

    def execute(self, userdata):
        rospy.loginfo("Starting Teknofest pinger gate approach")
        outcome = self.state_machine.execute()
        return "preempted" if outcome is None else outcome
