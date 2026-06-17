#!/usr/bin/env python3

import smach
import smach_ros
import rospy
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

from auv_smach.tf_utils import get_base_link
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SearchForPropState,
    DynamicPathState,
)
from auv_smach.initialize import DelayState


class PublishPingerWaypointState(smach.State):
    def __init__(self, direction="forward"):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.direction = direction
        self.pub = rospy.Publisher(
            "pinger_waypoint_direction", String, queue_size=1, latch=True
        )

    def execute(self, userdata):
        self.pub.publish(String(data=self.direction))
        rospy.sleep(0.5)
        return "succeeded"


class TogglePingerCollection(smach_ros.ServiceState):
    def __init__(self, data):
        smach_ros.ServiceState.__init__(
            self, "toggle_pinger_collection", SetBool, request=SetBoolRequest(data)
        )


class ResetPingerData(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "clear_pinger_data", Trigger, request=TriggerRequest()
        )


class ComputePingerPosition(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "compute_pinger_position", Trigger, request=TriggerRequest()
        )


class PingerSearchState(smach.StateMachine):
    def __init__(
        self, direction="forward", waypoint_frame="pinger_waypoint", wait_for=20.0
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        with self:
            smach.StateMachine.add(
                "PUBLISH_WAYPOINT",
                PublishPingerWaypointState(direction=direction),
                transitions={
                    "succeeded": "AIM_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "AIM_TO_WAYPOINT",
                SearchForPropState(
                    look_at_frame=waypoint_frame,
                    alignment_frame="waypoint_aim",
                    full_rotation=False,
                    source_frame=get_base_link(),
                    rotation_speed=0.4,
                ),
                transitions={
                    "succeeded": "DYNAMIC_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_TO_WAYPOINT",
                DynamicPathState(
                    plan_target_frame=waypoint_frame,
                    max_linear_velocity=0.4,
                    max_angular_velocity=0.3,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_WAYPOINT",
                AlignFrame(
                    source_frame=get_base_link(),
                    target_frame=waypoint_frame,
                    dist_threshold=0.15,
                    yaw_threshold=0.2,
                    timeout=30.0,
                    confirm_duration=10.0,
                    cancel_on_success=True,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CANCEL_CONTROL",
                    "preempted": "preempted",
                    "aborted": "CANCEL_CONTROL",
                },
            )

            smach.StateMachine.add(
                "CANCEL_CONTROL",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "WAIT_STABILISE",
                    "preempted": "preempted",
                    "aborted": "WAIT_STABILISE",
                },
            )

            smach.StateMachine.add(
                "WAIT_STABILISE",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "START_COLLECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "START_COLLECTION",
                TogglePingerCollection(True),
                transitions={
                    "succeeded": "COLLECTING_DELAY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "COLLECTING_DELAY",
                DelayState(delay_time=wait_for),
                transitions={
                    "succeeded": "STOP_COLLECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "STOP_COLLECTION",
                TogglePingerCollection(False),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class PingerTaskState(smach.State):
    def __init__(
        self,
        pinger_frame="pinger_frame",
        waypoint_frame="pinger_waypoint",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.pinger_frame = pinger_frame
        self.waypoint_frame = waypoint_frame

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with self.sm:
            smach.StateMachine.add(
                "RESET_PINGER_DATA",
                ResetPingerData(),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_PINGER_1",
                PingerSearchState(
                    direction="forward",
                    waypoint_frame=self.waypoint_frame,
                    wait_for=20,
                ),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_PINGER_2",
                PingerSearchState(
                    direction="right",
                    waypoint_frame=self.waypoint_frame,
                    wait_for=20,
                ),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_PINGER_3",
                PingerSearchState(
                    direction="backward",
                    waypoint_frame=self.waypoint_frame,
                    wait_for=20,
                ),
                transitions={
                    "succeeded": "COMPUTE_POSITION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "COMPUTE_POSITION",
                ComputePingerPosition(),
                transitions={
                    "succeeded": "LOOK_AT_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "LOOK_AT_PINGER",
                SearchForPropState(
                    look_at_frame=self.pinger_frame,
                    alignment_frame="look_at_pinger",
                    full_rotation=False,
                    source_frame=get_base_link(),
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo("Starting Pinger Localisation SMACH Task")
        outcome = self.sm.execute()
        if outcome is None:
            return "preempted"
        return outcome
