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
    LaunchTorpedoState,
)


class TorpedoTaskState(smach.State):
    def __init__(
        self,
        torpedo_map_radius,
        torpedo_map_depth,
        torpedo_target_frame="torpedo_map_target",
    ):

        self.torpedo_map_radius = torpedo_map_radius
        self.torpedo_map_depth = torpedo_map_depth
        self.torpedo_target_frame = torpedo_target_frame

        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_TORPEDO_DEPTH",
                SetDepthState(depth=self.torpedo_map_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "FIND_AND_AIM_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_TORPEDO",
                SearchForPropState(
                    look_at_frame="torpedo_map_link",
                    alignment_frame="torpedo_search",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "SET_TORPEDO_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_APPROACH_FRAME",
                SetRedBuoyRotationStartFrame(
                    base_frame="taluy/base_link",
                    center_frame="torpedo_map_link",
                    target_frame="torpedo_approach_start",
                    radius=self.torpedo_map_radius,
                ),
                transitions={
                    "succeeded": "SET_TORPEDO_TRAVEL_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_TRAVEL_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="torpedo_map_target"
                ),
                transitions={
                    "succeeded": "APPROACH_TO_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "APPROACH_TO_TORPEDO",
                NavigateToFrameState(
                    "taluy/base_link", "torpedo_approach_start", "torpedo_map_target"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_START",
                DelayState(delay_time=4.0),
                transitions={
                    "succeeded": "SET_TORPEDO_CLOSE_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_CLOSE_APPROACH_FRAME",
                SetRedBuoyRotationStartFrame(
                    base_frame="taluy/base_link",
                    center_frame="torpedo_map_link",
                    target_frame="torpedo_close_approach_start",
                    radius=0.4,
                ),
                transitions={
                    "succeeded": "CLOSE_APPROACH_TO_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_APPROACH_TO_TORPEDO",
                NavigateToFrameState(
                    "taluy/base_link",
                    "torpedo_close_approach_start",
                    "torpedo_map_target",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_CLOSE_APPROACH_COMPLETE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_CLOSE_APPROACH_COMPLETE",
                DelayState(delay_time=7.0),
                transitions={
                    "succeeded": "APPROACH_TO_TARGET_WHOLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "APPROACH_TO_TARGET_WHOLE",
                NavigateToFrameState(
                    "taluy/base_link",
                    self.torpedo_target_frame,
                    "torpedo_map_target",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_TARGET_APPROACH_COMPLETE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TARGET_APPROACH_COMPLETE",
                DelayState(delay_time=7.0),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_1",
                LaunchTorpedoState(id=1),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_LAUNCH",
                DelayState(delay_time=6.0),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_2",
                LaunchTorpedoState(id=2),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_2_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_2_LAUNCH",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "MOVE_BACK_TO_APPROACH_POSE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "MOVE_BACK_TO_APPROACH_POSE",
                NavigateToFrameState(
                    "taluy/base_link", "torpedo_approach_start", "torpedo_map_target"
                ),
                transitions={
                    "succeeded": "SET_TORPEDO_EXIT_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_EXIT_DEPTH",
                SetDepthState(depth=-0.7, sleep_duration=3.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # smach.StateMachine.add(
            #     "CANCEL_ALIGN_CONTROLLER",
            #     CancelAlignControllerState(),
            #     transitions={
            #         "succeeded": "succeeded",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )

    def execute(self, userdata):
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
