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
    PlanPathToSingleFrameState,
    ExecutePlannedPathsState,
)


class TorpedoTransformServiceNode(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_torpedo_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TorpedoTaskState(smach.State):
    def __init__(self, torpedo_map_radius, torpedo_map_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_TORPEDO_DEPTH",
                SetDepthState(depth=torpedo_map_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoTransformServiceNode(req=True),
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
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "PLAN_PATH_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_PATH_TO_CLOSE_APPROACH",
                PlanPathToSingleFrameState(
                    tf_buffer=self.tf_buffer,
                    target_frame="torpedo_close",
                    source_frame="taluy/base_link",
                ),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_TO_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_TO_PATH",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="dynamic_target"
                ),
                transitions={
                    "succeeded": "EXECUTE_TORPEDO_PATH_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_TORPEDO_PATH_TO_CLOSE_APPROACH",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "WAIT_FOR_ORIENTATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ORIENTATION",
                DelayState(delay_time=7.0),
                transitions={
                    "succeeded": "PLAN_PATH_TO_FRONT_VIEW",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_PATH_TO_FRONT_VIEW",
                PlanPathToSingleFrameState(
                    tf_buffer=self.tf_buffer,
                    target_frame="torpedo_front_view",
                    source_frame="taluy/base_link",
                ),
                transitions={
                    "succeeded": "EXECUTE_TORPEDO_PATH_TO_FRONT_VIEW",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_TORPEDO_PATH_TO_FRONT_VIEW",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "WAIT_FOR_FRONT_VIEW",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRONT_VIEW",
                DelayState(delay_time=7.0),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_TO_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_TO_LAUNCH",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link/torpedo_upper_link",
                    target_frame="torpedo_launch",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_LAUNCH_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_LAUNCH_1",
                DelayState(delay_time=12.0),
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
                    "succeeded": "WAIT_FOR_TORPEDO_LAUNCH_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_LAUNCH_2",
                DelayState(delay_time=5.0),
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
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
