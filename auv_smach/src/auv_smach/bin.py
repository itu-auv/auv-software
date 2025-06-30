from .initialize import *
import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from robot_localization.srv import SetPose, SetPoseRequest, SetPoseResponse
from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
    SetObjectTransformResponse,
    AlignFrameController,
    AlignFrameControllerRequest,
    AlignFrameControllerResponse,
)
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_ros
import numpy as np
import tf.transformations as transformations

from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
)
from auv_smach.red_buoy import SetRedBuoyRotationStartFrame

from auv_smach.initialize import DelayState

from auv_smach.common import (
    DropBallState,
)


class BinTaskState(smach.State):
    def __init__(self, bin_whole_depth, bin_target_frame="bin_whole_link"):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.bin_whole_depth = bin_whole_depth
        self.bin_target_frame = bin_target_frame

        # Open the container for adding states
        with self.state_machine:
            # smach.StateMachine.add(
            #     "SET_BIN_DEPTH",
            #     SetDepthState(depth=bin_whole_depth, sleep_duration=3.0),
            #     transitions={
            #         "succeeded": "SET_BIN_TRAVEL_START",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            smach.StateMachine.add(
                "FIND_AND_AIM_BIN",
                SearchForPropState(
                    look_at_frame="bin_whole_link",
                    alignment_frame="bin_search",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "SET_BIN_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_APPROACH_FRAME",
                SetRedBuoyRotationStartFrame(
                    base_frame="taluy/base_link",
                    center_frame="bin_whole_link",
                    target_frame="bin_whole_approach_start",
                    radius=1.0,
                ),
                transitions={
                    "succeeded": "SET_BIN_WHOLE_TRAVEL_APPROACH_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_WHOLE_TRAVEL_APPROACH_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="bin_whole_approach_start",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_APPROACH_ALIGNING_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_APPROACH_ALIGNING_START",
                DelayState(delay_time=40.0),
                transitions={
                    "succeeded": "SET_BIN_WHOLE_TRAVEL_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_WHOLE_TRAVEL_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="bin_whole_link",
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_TARGET_AREA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_TARGET_AREA",
                DelayState(delay_time=7.0),
                transitions={
                    "succeeded": "SET_BIN_TARGET_TRAVEL_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_TARGET_TRAVEL_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame=self.bin_target_frame
                ),
                transitions={
                    "succeeded": "SET_BIN_DROP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DROP_DEPTH",
                SetDepthState(depth=self.bin_whole_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_START",
                DelayState(delay_time=12.0),
                transitions={
                    "succeeded": "DROP_BALL_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_1",
                DropBallState(),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_1",
                DelayState(delay_time=8.0),
                transitions={
                    "succeeded": "DROP_BALL_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_2",
                DropBallState(),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_2",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # smach.StateMachine.add(
            #     "ROTATE_AROUND_BUOY",
            #     RotateAroundCenterState(
            #         "taluy/base_link",
            #         "red_buoy_link",
            #         "red_buoy_target",
            #         radius=radius,
            #         direction=direction,
            #     ),
            #     transitions={
            #         "succeeded": "CANCEL_ALIGN_CONTROLLER",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
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
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
