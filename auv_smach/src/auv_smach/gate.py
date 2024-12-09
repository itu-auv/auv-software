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
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np
import tf.transformations as transformations

from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
)

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from auv_smach.initialize import DelayState

class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req:bool):
        smach_ros.ServiceState.__init__(
            self,
            "taluy/set_gate_trajectory_enable",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Service to control the enable state of the TransformServiceNode 
        self.set_enable_service = rospy.ServiceProxy('taluy/set_gate_trajectory_enable', SetBool)

        # Open the container for adding states
        with self.state_machine:
            
            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=gate_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="gate_target"
                ),
                transitions={
                    "succeeded": "NAVIGATE_TO_GATE_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_GATE_START",
                NavigateToFrameState(
                    "taluy/base_link", "gate_enterance", "gate_target"
                ),
                transitions={
                    "succeeded": "NAVIGATE_TO_GATE_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_GATE_EXIT",
                NavigateToFrameState(
                    "gate_enterance", "gate_exit", "gate_target", n_turns=-1
                ),
                transitions={
                    "succeeded": "DISABLE_GATE_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "DISABLE_GATE_TRAJECTORY",
                TransformServiceEnableState(req=False),
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
