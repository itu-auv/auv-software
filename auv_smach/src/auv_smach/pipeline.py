#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    DynamicPathState,
    SetDepthState,
    AlignFrame,
)
from auv_smach.initialize import DelayState


# ---- ServiceState wrappers with FRIENDLY service names ----

class PipelineTrajectoryEnableState(smach_ros.ServiceState):
    """Enable/disable the pipeline trajectory publisher."""
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "/pipeline/trajectory/enable",   # RENAMED
            SetBool,
            request=SetBoolRequest(data=req),
        )


class AnnotatorEnableState(smach_ros.ServiceState):
    """Enable/disable the on-screen annotator."""
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "/vision/annotator/enable",       # RENAMED
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PipeFollowerEnableState(smach_ros.ServiceState):
    """Enable/disable the pipe follower node."""
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "/pipe_follower/enable",          # NEW: matches the node above
            SetBool,
            request=SetBoolRequest(data=req),
        )


class NavigateThroughPipelineState(smach.State):
    def __init__(self, pipeline_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Pipeline corner frame names (in navigation order)
        self.pipeline_frames = [
            "pipe_corner_1",
            "pipe_corner_2",
            "pipe_corner_3",
            "pipe_corner_4",
            "pipe_corner_5",
            "pipe_corner_6",
            "pipe_corner_7",
            "pipe_corner_8",
            "pipe_corner_9",
        ]

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_PIPELINE_DEPTH",
                SetDepthState(depth=pipeline_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "ENABLE_PIPE_FOLLOWER",   # <--- INSERTED HERE
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 1.5) Enable pipe follower (NEW)
            smach.StateMachine.add(
                "ENABLE_PIPE_FOLLOWER",
                PipeFollowerEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_STATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_STATE",
                DelayState(delay_time=100.0),
                transitions={
                    "succeeded": "DISABLE_PIPE_FOLLOWER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 1.5) Enable pipe follower (NEW)
            smach.StateMachine.add(
                "DISABLE_PIPE_FOLLOWER",
                PipeFollowerEnableState(req=False),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughPipelineState] Starting state machine execution.")
        outcome = self.state_machine.execute()
        if outcome is None:
            return "preempted"
        return outcome
