from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import (
    PlanPath,
    PlanPathRequest,
    SetDetectionFocus,
    SetDetectionFocusRequest,
)
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePathState,
    AlignFrame,
    DynamicPathState,
    SetDetectionFocusState,
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
        self.slalom_depth = slalom_depth
        self.slalom_direction = slalom_direction
        self.slalom_exit_angle = slalom_exit_angle

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
                    source_frame="taluy/base_link",
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
                    source_frame="taluy/base_link",
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
                    source_frame="taluy/base_link",
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

            first_wp_side = self.slalom_direction
            first_wp_state = f"ALIGN_WP_{first_wp_side}_0"

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
                    "succeeded": first_wp_state,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                f"ALIGN_WP_{self.slalom_direction}_0",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=f"slalom_wp_{self.slalom_direction}_0",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": f"ALIGN_WP_{self.slalom_direction}_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                f"ALIGN_WP_{self.slalom_direction}_1",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=f"slalom_wp_{self.slalom_direction}_1",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": f"ALIGN_WP_{self.slalom_direction}_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                f"ALIGN_WP_{self.slalom_direction}_2",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=f"slalom_wp_{self.slalom_direction}_2",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
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
