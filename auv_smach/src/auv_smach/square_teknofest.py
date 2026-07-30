import math

import rospy
import smach
import tf2_ros
import tf.transformations as transformations

from auv_smach.common import AlignFrame, CancelAlignControllerState, SetDepthState
from auv_smach.square import CreateSquareFramesState
from auv_smach.tf_utils import get_base_link


class CreateTeknofestSquareFramesState(CreateSquareFramesState):
    """Create the original square plus a circle around forward_right."""

    def __init__(self, side_length=10.0, circle_radius=1.0):
        super().__init__(
            side_length=side_length,
            start_frame="teknofest_square_start",
            forward_frame="teknofest_square_forward",
            forward_right_frame="teknofest_square_forward_right",
            right_frame="teknofest_square_right",
        )
        self.circle_radius = circle_radius
        self.circle_frames = [
            "teknofest_circle_{:02d}".format(index) for index in range(8)
        ]

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

        # The centre is forward_right. circle_00 is on its upper side and its
        # orientation matches the vehicle after ALIGN_FORWARD_RIGHT.
        angle_step = 2.0 * math.pi / len(self.circle_frames)
        for index, frame_name in enumerate(self.circle_frames):
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            circle_angle = math.pi / 2.0 + index * angle_step
            forward = side + self.circle_radius * math.cos(circle_angle)
            left = -side + self.circle_radius * math.sin(circle_angle)
            x, y = self._offset_from_start(
                start_x, start_y, start_yaw, forward, left
            )
            inward_yaw = start_yaw + circle_angle + math.pi
            transform = self._make_transform(
                "odom", frame_name, x, y, start_z, inward_yaw
            )
            if not self._publish_frame(transform):
                return "aborted"

        return "succeeded"


class NavigateTeknofestSquarePathState(smach.State):
    """The original square state machine with one circle inserted before b."""

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
        self.circle_frames = [
            "teknofest_circle_{:02d}".format(index) for index in range(8)
        ]

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
                transitions={"succeeded": "ALIGN_CIRCLE_00", **outcomes},
            )

            # This is the only addition to the original navigation sequence.
            # Revisit circle_00 once to close the full 360-degree orbit.
            circle_route = self.circle_frames + [self.circle_frames[0]]
            for index, target_frame in enumerate(circle_route):
                state_name = "ALIGN_CIRCLE_{:02d}".format(index)
                next_state = (
                    "ALIGN_CIRCLE_{:02d}".format(index + 1)
                    if index + 1 < len(circle_route)
                    else "b"
                )
                smach.StateMachine.add(
                    state_name,
                    AlignFrame(target_frame=target_frame, **align_args),
                    transitions={"succeeded": next_state, **outcomes},
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
