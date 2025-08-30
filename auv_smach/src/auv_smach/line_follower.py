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


class VisiualServoingServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "/pipe_follower/enable",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ToggleAnnotatorServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "/yolo_image_annotator/toggle_annotator",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class FollowPipelineState(smach.State):
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
                "ENABLE_ANNOTATOR",
                ToggleAnnotatorServiceState(req=True),
                transitions={
                    "succeeded": "SET_PIPELINE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_PIPELINE_DEPTH",
                SetDepthState(depth=pipeline_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "ENABLE_VISIUAL_SERVOING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_VISIUAL_SERVOING",
                VisiualServoingServiceEnableState(req=True),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[FollowPipelineState] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
