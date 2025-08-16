#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import smach
from auv_smach.common import (
    SetDepthState,
    DynamicPathState,
    AlignFrame,
    SetDetectionFocusState,
    LookAroundState,
)
from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import PlanPath, PlanPathRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from geometry_msgs.msg import PoseStamped
from auv_smach.initialize import DelayState

from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.initialize import DelayState, OdometryEnableState, ResetOdometryState

class PublishSlalomWaypointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_slalom_waypoints",
            SetBool,
            request=SetBoolRequest(data=True),
        )
class PoolCheckpointState(smach.State):
    def __init__(self, depth=-1.5, align_horizontal_offset=0.5, look_around_angles=[-15, 15], pool_checkpoint_exit_angle=0.0):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted', 'preempted'])
        self.pool_checkpoint_exit_angle = pool_checkpoint_exit_angle
        self.state_machine = smach.StateMachine(outcomes=['succeeded', 'aborted', 'preempted'])

        with self.state_machine:
            smach.StateMachine.add(
                "PUBLISH_SLALOM_WAYPOINTS",
                PublishSlalomWaypointsState(),
                transitions={
                    "succeeded": "SET_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DEPTH",
                SetDepthState(depth=depth),
                transitions={
                    "succeeded": "WAIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "DYNAMIC_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH",
                DynamicPathState(plan_target_frame='pool_checkpoint'),
                transitions={
                    "succeeded": "ALIGN_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME",
                AlignFrame(target_frame='pool_checkpoint',source_frame="taluy/base_link",angle_offset=self.pool_checkpoint_exit_angle),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS",
                SetDetectionFocusState(focus_object='bin,torpedo,octagon'),
                transitions={
                    "succeeded": "LOOK_AROUND",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AROUND",
                LookAroundState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo("Executing Pool Checkpoint Smach State")
        outcome = self.state_machine.execute()
        if outcome is None:
            return "preempted"
        return outcome
