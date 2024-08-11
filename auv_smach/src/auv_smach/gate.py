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


class BarrelRollState(smach.State):
    def __init__(self, duration: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.cmd_vel_pub = rospy.Publisher("/taluy/cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("/taluy/enable", Bool, queue_size=10)
        self.rate = rospy.Rate(20)  # 20 Hz
        self.duration = duration

    def execute(self, userdata):
        rospy.loginfo("Executing Barrel Roll State")

        # Create and publish the Twist message
        twist_msg = Twist()
        twist_msg.angular.x = 3000.0

        # Create and publish the Bool message
        enable_msg = Bool()
        enable_msg.data = True

        duration = rospy.Duration(self.duration)
        start_time = rospy.Time.now()

        while rospy.Time.now() - start_time < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist_msg)
            self.enable_pub.publish(enable_msg)
            self.rate.sleep()

        for _ in range(5):
            self.cmd_vel_pub.publish(Twist())
            self.rate.sleep()

        return "succeeded"


class DoBarrelRoll(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "WAIT_FOR_SETTLE_DOWN_0",
                DelayState(delay_time=7.0),
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
                    "succeeded": "DO_BARREL_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DO_BARREL_ROLL",
                BarrelRollState(2.5),
                transitions={
                    "succeeded": "WAIT_FOR_SETTLE_DOWN_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SETTLE_DOWN_1",
                DelayState(delay_time=7.0),
                transitions={
                    "succeeded": "DO_BARREL_ROLL_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DO_BARREL_ROLL_2",
                BarrelRollState(2.5),
                transitions={
                    "succeeded": "WAIT_FOR_SETTLE_DOWN_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SETTLE_DOWN_2",
                DelayState(delay_time=7.0),
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


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=gate_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "BARREL_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BACK_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="mission_start_link"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_SETTLE_DOWN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SETTLE_DOWN",
                DelayState(delay_time=5.0),
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
                    "succeeded": "BARREL_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "BARREL_ROLL",
                DoBarrelRoll(),
                transitions={
                    "succeeded": "NAVIGATE_TO_GATE_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_GATE_EXIT",
                NavigateToFrameState(
                    "gate_enterance", "gate_exit", "gate_target", n_turns=0
                ),
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
