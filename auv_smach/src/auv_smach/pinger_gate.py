import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    DynamicPathState,
    SearchForPropState,
    SetDetectionState,
)
from auv_smach.initialize import DelayState
from auv_smach.tf_utils import get_base_link


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
        )


class PingerGateTaskState(smach.State):
    """Initial, bbox-based approach phase of the Teknofest pinger gate task."""

    def __init__(
        self,
        pinger_bbox_frame: str = "pinger_bbox",
        close_approach_frame: str = "pinger_close_approach",
        gate_closer_frame: str = "gate_closer",
        gate_farther_frame: str = "gate_farther",
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
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
                    "succeeded": "FEVZI",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FEVZI",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_GATE_FARTHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_GATE_FARTHER",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=gate_farther_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=10.0,
                    cancel_on_success=False,
                ),
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
