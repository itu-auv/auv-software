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
import tf2_geometry_msgs

from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SetFrameAtRadiusFacingFrameState,
    SetFrameLookingAtState,
)

from auv_smach.initialize import DelayState


class RotateAroundCenterState(smach.State):
    def __init__(
        self, base_frame, center_frame, target_frame, radius=0.2, direction="ccw"
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_frame = base_frame
        self.center_frame = center_frame
        self.target_frame = target_frame
        self.radius = radius
        self.direction = direction
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(10)

        self.linear_velocity = 0.8  # rospy.get_param("/smach/max_linear_velocity")
        self.angular_velocity = 0.8  # rospy.get_param("/smach/max_angular_velocity")

    def execute(self, userdata):
        try:
            # Lookup the initial transform from center_frame to base_frame
            center_to_base_transform = self.tf_buffer.lookup_transform(
                self.center_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(100000.0),
            )

            # Calculate the initial angle from the center to the base
            base_position = np.array(
                [
                    center_to_base_transform.transform.translation.x,
                    center_to_base_transform.transform.translation.y,
                ]
            )
            initial_angle = np.arctan2(base_position[1], base_position[0])

            # Calculate the duration for one full rotation (2 * pi radians)
            duration = 2 * np.pi * self.radius / self.linear_velocity
            num_steps = int(duration * 10)  # Number of steps based on the rate
            angular_step = 2 * np.pi / num_steps  # Angle step per iteration

            # Adjust the direction based on the input argument
            if self.direction == "cw":
                angular_step = -angular_step

            for i in range(num_steps):
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                # Calculate the current angle
                angle = initial_angle + angular_step * i

                # Calculate the new position using the radius and angle
                interp_pos = self.radius * np.array([np.cos(angle), np.sin(angle)])

                # The orientation should be such that the target_frame is always facing the center_frame
                facing_angle = (
                    angle + np.pi
                )  # The orientation should face the center (180 degrees offset)
                quaternion = transformations.quaternion_from_euler(0, 0, facing_angle)

                # Broadcast the new transform relative to center_frame
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.center_frame
                t.child_frame_id = self.target_frame
                t.transform.translation.x = interp_pos[0]
                t.transform.translation.y = interp_pos[1]
                t.transform.translation.z = 0.0  # Assume 2D movement (z = 0)
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]

                self.tf_broadcaster.sendTransform(t)
                self.rate.sleep()

            return "succeeded"

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup exception: {e}")
            return "aborted"


class RotateAroundBuoyState(smach.State):
    def __init__(self, radius, direction, red_buoy_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_RED_BUOY_DEPTH",
                SetDepthState(depth=red_buoy_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SET_RED_BUOY_TRAVEL_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_RED_BUOY_TRAVEL_START",
                SetFrameLookingAtState(
                    base_frame="taluy/base_link",
                    look_at_frame="red_buoy_link",
                    target_frame="red_buoy_travel_start",
                ),
                transitions={
                    "succeeded": "SET_RED_BUOY_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_RED_BUOY_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="red_buoy_travel_start"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_TRAVEL_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_TRAVEL_START",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "SET_RED_BUOY_ROTATION_START_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_RED_BUOY_ROTATION_START_FRAME",
                SetFrameAtRadiusFacingFrameState(
                    base_frame="taluy/base_link",
                    center_frame="red_buoy_link",
                    target_frame="red_buoy_rotation_start",
                    radius=radius,
                ),
                transitions={
                    "succeeded": "SET_RED_BUOY_TRAVEL_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_RED_BUOY_TRAVEL_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="red_buoy_target"
                ),
                transitions={
                    "succeeded": "NAVIGATE_TO_RED_BUOY_ROTATION_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_RED_BUOY_ROTATION_START",
                NavigateToFrameState(
                    "taluy/base_link", "red_buoy_rotation_start", "red_buoy_target"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_ROTATION_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_ROTATION_START",
                DelayState(delay_time=4.0),
                transitions={
                    "succeeded": "ROTATE_AROUND_BUOY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROTATE_AROUND_BUOY",
                RotateAroundCenterState(
                    "taluy/base_link",
                    "red_buoy_link",
                    "red_buoy_target",
                    radius=radius,
                    direction=direction,
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
