from auv_smach.tf_utils import get_base_link
from .initialize import *
import smach
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_smach.common import (
    CancelAlignControllerState,
    SearchForPropState,
    SetDepthState,
    AlignFrame,
)
from auv_smach.initialize import DelayState


class PublishSearchPointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/publish_search_points",
            Trigger,
            request=TriggerRequest(),
        )


class StartPointSearchState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/start_point_search",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class StopPointSearchState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/stop_point_search",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class PublishWaypointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/publish_waypoints",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class NavigateThroughSlalomExpState(smach.State):
    def __init__(
        self,
        slalom_depth: float,
        slalom_direction: str = "left",
        slalom_exit_angle: float = 0.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.base_link = get_base_link()
        self.slalom_depth = slalom_depth
        self.slalom_direction = slalom_direction

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with self.sm:
            smach.StateMachine.add(
                "SET_SLALOM_DEPTH",
                SetDepthState(depth=self.slalom_depth),
                transitions={
                    "succeeded": "PUBLISH_SEARCH_POINTS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "PUBLISH_SEARCH_POINTS",
                PublishSearchPointsState(),
                transitions={
                    "succeeded": "START_POINT_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "START_POINT_SEARCH",
                StartPointSearchState(),
                transitions={
                    "succeeded": "ALIGN_SEQUENCE_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            first_side = "right" if self.slalom_direction == "left" else "left"
            second_side = "left" if self.slalom_direction == "left" else "right"

            smach.StateMachine.add(
                "ALIGN_SEQUENCE_START",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_search_start",
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_FIRST_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_FIRST_SEARCH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=f"slalom_search_{first_side}",
                    confirm_duration=2.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_SECOND_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_SECOND_SEARCH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=f"slalom_search_{second_side}",
                    confirm_duration=2.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "STOP_POINT_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "STOP_POINT_SEARCH",
                StopPointSearchState(),
                transitions={
                    "succeeded": "PUBLISH_WAYPOINTS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "PUBLISH_WAYPOINTS",
                PublishWaypointsState(),
                transitions={
                    "succeeded": "WAIT_FOR_TF",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # just in case
            smach.StateMachine.add(
                "WAIT_FOR_TF",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "ALIGN_WP_0",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_WP_0",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_wp_0",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "LOOK_WP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "LOOK_WP_1",
                SearchForPropState(
                    look_at_frame="slalom_wp_1",
                    alignment_frame=f"sus",
                    full_rotation=False,
                    set_frame_duration=3.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_WP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_WP_1",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_wp_1",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "LOOK_WP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "LOOK_WP_2",
                SearchForPropState(
                    look_at_frame="slalom_wp_2",
                    alignment_frame=f"sus",
                    full_rotation=False,
                    set_frame_duration=6.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_WP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_WP_2",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_wp_2",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    keep_orientation=True,
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
        return self.sm.execute()
