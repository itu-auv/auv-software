import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest
from std_srvs.srv import Empty, EmptyRequest
from std_srvs.srv import Trigger, TriggerRequest
from robot_localization.srv import SetPose, SetPoseRequest
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from std_msgs.msg import Bool
from auv_smach.common import (
    CancelAlignControllerState,
    ClearObjectMapState,
)
from typing import Optional, Literal
from dataclasses import dataclass


class ResetOdometryState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "reset_odometry",
            Trigger,
            request=TriggerRequest(),
        )


class WaitForKillswitchEnabledState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.enabled = False
        self.killswitch_subscriber = rospy.Subscriber(
            "propulsion_board/status", Bool, self.killswitch_callback
        )

    def killswitch_callback(self, msg):
        self.enabled = msg.data

    def execute(self, userdata):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.enabled:
                return "succeeded"
            rate.sleep()
        return "aborted"


class DVLEnableState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "dvl/enable",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class DelayState(smach.State):
    def __init__(self, delay_time):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.delay_time = delay_time

    def execute(self, userdata):
        rospy.sleep(self.delay_time)
        return "succeeded"


class OdometryEnableState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "localization_enable",
            Empty,
            request=EmptyRequest(),
        )


class ResetOdometryPoseState(smach_ros.ServiceState):
    def __init__(self):
        initial_pose_request = SetPoseRequest()
        initial_pose_request.pose.pose.pose.orientation.w = 1.0

        smach_ros.ServiceState.__init__(
            self,
            "set_pose",
            SetPose,
            request=initial_pose_request,
        )


class SetStartFrameState(smach_ros.ServiceState):
    def __init__(self, frame_name: str):
        transform_request = SetObjectTransformRequest()
        transform_request.transform.header.frame_id = "taluy/base_link"
        transform_request.transform.child_frame_id = frame_name
        transform_request.transform.transform.rotation.w = 1.0

        smach_ros.ServiceState.__init__(
            self,
            "set_object_transform",
            SetObjectTransform,
            request=transform_request,
        )


class InitializeState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "WAIT_FOR_KILLSWITCH_ENABLED",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_KILLSWITCH_ENABLED",
                WaitForKillswitchEnabledState(),
                transitions={
                    "succeeded": "DVL_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DVL_ENABLE",
                DVLEnableState(),
                transitions={
                    "succeeded": "DELAY_FOR_DVL_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DELAY_FOR_DVL_ENABLE",
                DelayState(delay_time=4.4),
                transitions={
                    "succeeded": "ODOMETRY_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ODOMETRY_ENABLE",
                OdometryEnableState(),
                transitions={
                    "succeeded": "CLEAR_OBJECT_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLEAR_OBJECT_MAP",
                ClearObjectMapState(),
                transitions={
                    "succeeded": "SET_START_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_START_FRAME",
                SetStartFrameState(frame_name="mission_start_link"),
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

        # Return the outcome of the state machine
        return outcome
