#!/usr/bin/env python3

import smach
import smach_ros
import rospy
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

from auv_smach.tf_utils import get_base_link
from auv_smach.common import (
    AlignFrame,
    DynamicPathState,
    SetDepthState,
)
from auv_smach.initialize import DelayState


class PingerAlignTaskState(smach.State):
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
                "DYNAMIC_PATH_TO_PINGER",
                DynamicPathState(plan_target_frame=self.pinger_frame),
                transitions={
                    "succeeded": "ALIGN_TO_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_PINGER",
                AlignFrame(
                    source_frame=get_base_link(),
                    target_frame=self.pinger_frame,
                    keep_orientation=True,
                    dist_threshold=0.3,
                    timeout=30.0,
                    confirm_duration=2.0,
                    cancel_on_success=True,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "IN_ASSAGI",
                SetDepthState(
                    depth=-5,
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
