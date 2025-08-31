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


class SetStartFrameState(smach_ros.ServiceState):
    def __init__(self, frame_name: str):
        transform_request = SetObjectTransformRequest()
        transform_request.transform.header.frame_id = "mission_start_link"
        transform_request.transform.child_frame_id = frame_name
        transform_request.transform.transform.translation.x = (
            1.0  # 1 metre ahead of mission_start_link
        )
        transform_request.transform.transform.rotation.w = 1.0

        smach_ros.ServiceState.__init__(
            self,
            "set_object_transform",
            SetObjectTransform,
            request=transform_request,
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
            # smach.StateMachine.add(
            #     "SET_PIPE_START_FRAME",
            #     SetStartFrameState(frame_name="pipe_mission_start"),
            #     transitions={
            #         "succeeded": "ALIGN_TO_PIPE_MISSION_START",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            # smach.StateMachine.add(
            #     "ALIGN_TO_PIPE_MISSION_START",
            #     AlignFrame(
            #         source_frame="taluy/base_link",
            #         target_frame="pipe_mission_start",
            #         confirm_duration=1.0,
            #         timeout=15.0,
            #         cancel_on_success=True,
            #         max_linear_velocity=0.3,
            #     ),
            #     transitions={
            #         "succeeded": "ENABLE_ANNOTATOR",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
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
                SetDepthState(depth=-3.5, sleep_duration=3.0),
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
                    "succeeded": "WAIT_FOR_SECOND_SET_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SECOND_SET_DEPTH",
                DelayState(delay_time=50.0),
                transitions={
                    "succeeded": "SET_SECOND_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_SECOND_DEPTH",
                SetDepthState(depth=-2.0, sleep_duration=3.0),
                transitions={
                    "succeeded": "WAIT_FOR_THIRD_SET_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_THIRD_SET_DEPTH",
                DelayState(delay_time=20.0),
                transitions={
                    "succeeded": "SET_THIRD_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_THIRD_DEPTH",
                SetDepthState(depth=-0.7, sleep_duration=3.0),
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
