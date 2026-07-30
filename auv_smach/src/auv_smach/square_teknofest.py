import math

import rospy
import smach
import tf2_ros
import tf.transformations as transformations

from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    NavigateToFrameState,
    SetAlignControllerTargetState,
    SetDepthState,
)
from auv_smach.initialize import DelayState
from auv_smach.red_buoy import RotateAroundCenterState
from auv_smach.square import CreateSquareFramesState
from auv_smach.tf_utils import get_base_link


class CreateTeknofestSquareFramesState(CreateSquareFramesState):
    """Create the original square plus the orbit starting frame."""

    def __init__(self, side_length=10.0, circle_radius=1.0):
        super().__init__(
            side_length=side_length,
            start_frame="teknofest_square_start",
            forward_frame="teknofest_square_forward",
            forward_right_frame="teknofest_square_forward_right",
            right_frame="teknofest_square_right",
        )
        self.circle_radius = circle_radius

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        try:
            rospy.wait_for_service("set_object_transform", timeout=5.0)
            odom_to_base = self.tf_buffer.lookup_transform(
                "odom", self.base_link, rospy.Time(0), rospy.Duration(4.0)
            )
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn(
                "[CreateTeknofestSquareFramesState] Service unavailable: %s", error
            )
            return "aborted"
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn(
                "[CreateTeknofestSquareFramesState] TF lookup failed: %s", error
            )
            return "aborted"

        translation = odom_to_base.transform.translation
        rotation = odom_to_base.transform.rotation
        _, _, start_yaw = transformations.euler_from_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        start_x = translation.x
        start_y = translation.y
        start_z = translation.z
        side = self.side_length

        square_specs = [
            (self.start_frame, 0.0, 0.0, 0.0),
            (self.forward_frame, side, 0.0, 0.0),
            (self.forward_right_frame, side, -side, -math.pi / 2.0),
            (self.right_frame, 0.0, -side, math.pi),
            (
                "teknofest_circle_start",
                side,
                -side + self.circle_radius,
                -math.pi / 2.0,
            ),
        ]
        for frame_name, forward, left, yaw_offset in square_specs:
            x, y = self._offset_from_start(
                start_x, start_y, start_yaw, forward, left
            )
            transform = self._make_transform(
                "odom", frame_name, x, y, start_z, start_yaw + yaw_offset
            )
            if not self._publish_frame(transform):
                return "aborted"

        return "succeeded"


class NavigateTeknofestSquarePathState(smach.State):
    """The original square with a RedBuoy-style continuous orbit before b."""

    def __init__(
        self,
        side_length=10.0,
        circle_radius=1.0,
        depth=-1.0,
        max_linear_velocity=None,
        max_angular_velocity=None,
        dist_threshold=0.2,
        yaw_threshold=0.15,
        timeout_per_side=45.0,
        confirm_duration=0.5,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.source_frame = get_base_link()
        self.start_frame = "teknofest_square_start"
        self.forward_frame = "teknofest_square_forward"
        self.forward_right_frame = "teknofest_square_forward_right"
        self.right_frame = "teknofest_square_right"
        self.circle_start_frame = "teknofest_circle_start"
        self.circle_target_frame = "teknofest_circle_target"

        if max_linear_velocity is None:
            max_linear_velocity = rospy.get_param("/smach/max_linear_velocity", 0.3)
        if max_angular_velocity is None:
            max_angular_velocity = rospy.get_param(
                "/smach/max_angular_velocity", 0.45
            )

        align_args = {
            "source_frame": self.source_frame,
            "dist_threshold": dist_threshold,
            "yaw_threshold": yaw_threshold,
            "timeout": timeout_per_side,
            "confirm_duration": confirm_duration,
            "max_linear_velocity": max_linear_velocity,
            "max_angular_velocity": max_angular_velocity,
        }
        outcomes = {"preempted": "preempted", "aborted": "aborted"}
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "CANCEL_ACTIVE_ALIGNMENT",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "CREATE_SQUARE_FRAMES",
                    **outcomes,
                },
            )
            smach.StateMachine.add(
                "CREATE_SQUARE_FRAMES",
                CreateTeknofestSquareFramesState(
                    side_length=side_length,
                    circle_radius=circle_radius,
                ),
                transitions={
                    "succeeded": "SET_INITIAL_DEPTH",
                    **outcomes,
                },
            )
            smach.StateMachine.add(
                "SET_INITIAL_DEPTH",
                SetDepthState(depth=depth),
                transitions={"succeeded": "ALIGN_FORWARD", **outcomes},
            )
            smach.StateMachine.add(
                "ALIGN_FORWARD",
                AlignFrame(target_frame=self.forward_frame, **align_args),
                transitions={"succeeded": "a", **outcomes},
            )
            smach.StateMachine.add(
                "a",
                AlignFrame(
                    target_frame=self.forward_frame,
                    angle_offset=-math.pi / 2.0,
                    **align_args,
                ),
                transitions={"succeeded": "ALIGN_FORWARD_RIGHT", **outcomes},
            )
            smach.StateMachine.add(
                "ALIGN_FORWARD_RIGHT",
                AlignFrame(target_frame=self.forward_right_frame, **align_args),
                transitions={
                    "succeeded": "SET_CIRCLE_ALIGN_CONTROLLER_TARGET",
                    **outcomes,
                },
            )
            smach.StateMachine.add(
                "SET_CIRCLE_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame=self.source_frame,
                    target_frame=self.circle_target_frame,
                ),
                transitions={
                    "succeeded": "NAVIGATE_TO_CIRCLE_START",
                    **outcomes,
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_CIRCLE_START",
                NavigateToFrameState(
                    self.source_frame,
                    self.circle_start_frame,
                    self.circle_target_frame,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_CIRCLE_START_ALIGNMENT",
                    **outcomes,
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_CIRCLE_START_ALIGNMENT",
                DelayState(delay_time=4.0),
                transitions={"succeeded": "ROTATE_AROUND_CORNER", **outcomes},
            )
            smach.StateMachine.add(
                "ROTATE_AROUND_CORNER",
                RotateAroundCenterState(
                    base_frame=self.source_frame,
                    center_frame=self.forward_right_frame,
                    target_frame=self.circle_target_frame,
                    radius=circle_radius,
                    direction="cw",
                ),
                transitions={"succeeded": "b", **outcomes},
            )

            smach.StateMachine.add(
                "b",
                AlignFrame(
                    target_frame=self.forward_right_frame,
                    angle_offset=-math.pi / 2.0,
                    **align_args,
                ),
                transitions={"succeeded": "ALIGN_RIGHT", **outcomes},
            )
            smach.StateMachine.add(
                "ALIGN_RIGHT",
                AlignFrame(target_frame=self.right_frame, **align_args),
                transitions={"succeeded": "c", **outcomes},
            )
            smach.StateMachine.add(
                "c",
                AlignFrame(
                    target_frame=self.right_frame,
                    angle_offset=-math.pi / 2.0,
                    **align_args,
                ),
                transitions={"succeeded": "ALIGN_START", **outcomes},
            )
            smach.StateMachine.add(
                "ALIGN_START",
                AlignFrame(
                    target_frame=self.start_frame,
                    angle_offset=math.pi / 2.0,
                    **align_args,
                ),
                transitions={"succeeded": "ALIGN_START_NO_OFFSET", **outcomes},
            )
            smach.StateMachine.add(
                "ALIGN_START_NO_OFFSET",
                AlignFrame(
                    target_frame=self.start_frame,
                    cancel_on_success=True,
                    **align_args,
                ),
                transitions={"succeeded": "SET_FINAL_DEPTH", **outcomes},
            )
            smach.StateMachine.add(
                "SET_FINAL_DEPTH",
                SetDepthState(depth=0.0),
                transitions={"succeeded": "succeeded", **outcomes},
            )

    def execute(self, userdata):
        return self.state_machine.execute(userdata)
