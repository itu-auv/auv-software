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
)
from auv_smach.red_buoy import SetRedBuoyRotationStartFrame, SetFrameLookingAtState

from auv_smach.initialize import DelayState

from auv_smach.common import (
    DropBallState,
    ExecutePlannedPathsState,
    CancelAlignControllerState,
)

from auv_navigation.path_planning.path_planners import PathPlanners

from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse


class BinTransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_bin_frames",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PlanBinPathState(smach.State):
    """State that plans the path for the bin task"""

    def __init__(self, tf_buffer):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanTorpedoPathState] Preempt requested")
                return "preempted"

            path_planners = PathPlanners(
                self.tf_buffer
            )  # instance of PathPlanners with tf_buffer
            paths = path_planners.path_for_bin()

            if paths is None:
                return "aborted"

            userdata.planned_paths = paths
            return "succeeded"

        except Exception as e:
            rospy.logerr("[PlanTorpedoPathState] Error: %s", str(e))
            return "aborted"


class BinTaskState(smach.State):
    def __init__(self, bin_task_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "SET_BIN_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DEPTH",
                SetDepthState(depth=bin_task_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SET_BIN_AIMER_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_AIMER_FRAME",
                SetFrameLookingAtState(
                    base_frame="taluy/base_link",
                    look_at_frame="bin_whole_link",
                    target_frame="bin_whole_travel_start",
                ),
                transitions={
                    "succeeded": "ALIGN_BIN_AIMER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_BIN_AIMER",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="bin_whole_travel_start",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_AIMER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_AIMER",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "PLAN_BIN_PATHS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_BIN_PATHS",
                PlanBinPathState(self.tf_buffer),
                transitions={
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=False),
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
                    "succeeded": "EXECUTE_BIN_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_BIN_PATH",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_TO_DROP_AREA",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_TO_DROP_AREA",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link/ball_dropper_link",
                    target_frame="bin/blue_link",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_DROP_AREA",
                DelayState(delay_time=15.0),
                transitions={
                    "succeeded": "SET_BIN_DROP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DROP_DEPTH",
                SetDepthState(depth=-1.2, sleep_duration=3.0),
                transitions={
                    "succeeded": "WAIT_FOR_BIN_DROP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BIN_DROP_DEPTH",
                DelayState(delay_time=10.0),
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
                    "succeeded": "SET_BIN_EXIT_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_EXIT_DEPTH",
                SetDepthState(depth=-0.7, sleep_duration=3.0),
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
