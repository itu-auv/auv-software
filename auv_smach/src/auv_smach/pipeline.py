from auv_smach.tf_utils import get_tf_buffer
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


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_pipeline_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ToggleAnnotatorServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_image_annotator",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class NavigateThroughPipelineState(smach.State):
    def __init__(self, pipeline_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()

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
                SetDepthState(depth=pipeline_depth),
                transitions={
                    "succeeded": "ENABLE_PIPELINE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_PIPELINE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_TRAJECTORY_GENERATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TRAJECTORY_GENERATION",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "DISABLE_PIPELINE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_PIPELINE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 1
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_1",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[0],
                    max_linear_velocity=0.3,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_1",
                AlignFrame(
                    target_frame=self.pipeline_frames[0],
                    source_frame="taluy/base_link",
                    angle_offset=1.57,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 2
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_2",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[1],
                    max_linear_velocity=0.3,
                    angle_offset=1.57,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_2",
                AlignFrame(
                    target_frame=self.pipeline_frames[1],
                    source_frame="taluy/base_link",
                    angle_offset=3.14,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 3
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_3",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[2],
                    max_linear_velocity=0.3,
                    angle_offset=3.14,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_3",
                AlignFrame(
                    target_frame=self.pipeline_frames[2],
                    source_frame="taluy/base_link",
                    angle_offset=-1.57,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_4",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 4
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_4",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[3],
                    max_linear_velocity=0.3,
                    angle_offset=-1.57,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_4",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_4",
                AlignFrame(
                    target_frame=self.pipeline_frames[3],
                    source_frame="taluy/base_link",
                    angle_offset=3.14,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_5",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 5
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_5",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[4],
                    max_linear_velocity=0.3,
                    angle_offset=3.14,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_5",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_5",
                AlignFrame(
                    target_frame=self.pipeline_frames[4],
                    source_frame="taluy/base_link",
                    angle_offset=1.57,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_6",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 6
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_6",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[5],
                    max_linear_velocity=0.3,
                    angle_offset=1.57,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_6",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_6",
                AlignFrame(
                    target_frame=self.pipeline_frames[5],
                    source_frame="taluy/base_link",
                    angle_offset=3.14,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_7",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 7
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_7",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[6],
                    max_linear_velocity=0.3,
                    angle_offset=3.14,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_7",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_7",
                AlignFrame(
                    target_frame=self.pipeline_frames[6],
                    source_frame="taluy/base_link",
                    angle_offset=1.57,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_8",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 8
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_8",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[7],
                    max_linear_velocity=0.3,
                    angle_offset=1.57,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_8",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_8",
                AlignFrame(
                    target_frame=self.pipeline_frames[7],
                    source_frame="taluy/base_link",
                    angle_offset=0.0,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CORNER_9",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            ### 9
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CORNER_9",
                DynamicPathState(
                    plan_target_frame=self.pipeline_frames[8],
                    max_linear_velocity=0.3,
                    angle_offset=0.0,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_CORNER_9",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_CORNER_9",
                AlignFrame(
                    target_frame=self.pipeline_frames[8],
                    source_frame="taluy/base_link",
                    angle_offset=0.0,
                    confirm_duration=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "DISABLE_ANNOTATOR",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_ANNOTATOR",
                ToggleAnnotatorServiceState(req=False),
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
        rospy.logdebug(
            "[NavigateThroughPipelineState] Starting state machine execution."
        )

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
