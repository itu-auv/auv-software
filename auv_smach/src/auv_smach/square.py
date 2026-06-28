import math

import rospy
import smach
import tf2_ros
import tf.transformations as transformations
from geometry_msgs.msg import TransformStamped

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from auv_smach.common import AlignFrame, CancelAlignControllerState, SetDepthState
from auv_smach.tf_utils import get_base_link, get_tf_buffer


class CreateSquareFramesState(smach.State):
    def __init__(
        self,
        side_length: float = 10.0,
        start_frame: str = "square_start",
        forward_frame: str = "square_forward",
        forward_right_frame: str = "square_forward_right",
        right_frame: str = "square_right",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.side_length = side_length
        self.start_frame = start_frame
        self.forward_frame = forward_frame
        self.forward_right_frame = forward_right_frame
        self.right_frame = right_frame
        self.base_link = get_base_link()
        self.tf_buffer = get_tf_buffer()
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

    @staticmethod
    def _make_transform(parent_frame, child_frame, x, y, z, yaw):
        quat = transformations.quaternion_from_euler(0.0, 0.0, yaw)

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        return transform

    @staticmethod
    def _offset_from_start(start_x, start_y, start_yaw, forward, left):
        x = start_x + math.cos(start_yaw) * forward - math.sin(start_yaw) * left
        y = start_y + math.sin(start_yaw) * forward + math.cos(start_yaw) * left
        return x, y

    def _publish_frame(self, transform):
        request = SetObjectTransformRequest()
        request.transform = transform
        response = self.set_object_transform_service(request)
        if not response.success:
            rospy.logwarn(
                "[CreateSquareFramesState] Failed to set %s: %s",
                transform.child_frame_id,
                response.message,
            )
            return False

        rospy.loginfo(
            "[CreateSquareFramesState] Set %s at x=%.2f y=%.2f yaw=%.2f",
            transform.child_frame_id,
            transform.transform.translation.x,
            transform.transform.translation.y,
            transformations.euler_from_quaternion(
                (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
            )[2],
        )
        return True

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        try:
            rospy.wait_for_service("set_object_transform", timeout=5.0)
            odom_to_base = self.tf_buffer.lookup_transform(
                "odom", self.base_link, rospy.Time(0), rospy.Duration(4.0)
            )
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logwarn("[CreateSquareFramesState] Service unavailable: %s", e)
            return "aborted"
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("[CreateSquareFramesState] TF lookup failed: %s", e)
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

        frame_specs = [
            (self.start_frame, 0.0, 0.0, 0.0),
            (self.forward_frame, side, 0.0, 0.0),
            (self.forward_right_frame, side, -side, -math.pi / 2.0),
            (self.right_frame, 0.0, -side, math.pi),
        ]

        for frame_name, forward, left, yaw_offset in frame_specs:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            x, y = self._offset_from_start(start_x, start_y, start_yaw, forward, left)
            transform = self._make_transform(
                "odom", frame_name, x, y, start_z, start_yaw + yaw_offset
            )
            if not self._publish_frame(transform):
                return "aborted"

        return "succeeded"


class NavigateSquarePathState(smach.State):
    def __init__(
        self,
        side_length: float = 10.0,
        depth: float = -1.0,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        dist_threshold: float = 0.2,
        yaw_threshold: float = 0.15,
        timeout_per_side: float = 45.0,
        confirm_duration: float = 0.5,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.source_frame = get_base_link()
        self.start_frame = "square_start"
        self.forward_frame = "square_forward"
        self.forward_right_frame = "square_forward_right"
        self.right_frame = "square_right"

        if max_linear_velocity is None:
            max_linear_velocity = rospy.get_param("/smach/max_linear_velocity", 0.3)
        if max_angular_velocity is None:
            max_angular_velocity = rospy.get_param("/smach/max_angular_velocity", 0.45)

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "CANCEL_ACTIVE_ALIGNMENT",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "CREATE_SQUARE_FRAMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CREATE_SQUARE_FRAMES",
                CreateSquareFramesState(
                    side_length=side_length,
                    start_frame=self.start_frame,
                    forward_frame=self.forward_frame,
                    forward_right_frame=self.forward_right_frame,
                    right_frame=self.right_frame,
                ),
                transitions={
                    "succeeded": "SET_INITIAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_INITIAL_DEPTH",
                SetDepthState(depth=depth),
                transitions={
                    "succeeded": "ALIGN_FORWARD",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FORWARD",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.forward_frame,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "a",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "a",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.forward_frame,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    angle_offset=-math.pi / 2.0,
                ),
                transitions={
                    "succeeded": "ALIGN_FORWARD_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FORWARD_RIGHT",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.forward_right_frame,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "b",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "b",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.forward_right_frame,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    angle_offset=-math.pi / 2.0,
                ),
                transitions={
                    "succeeded": "ALIGN_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_RIGHT",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.right_frame,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "c",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "c",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.right_frame,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    angle_offset=-math.pi / 2.0,
                ),
                transitions={
                    "succeeded": "ALIGN_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_START",
                AlignFrame(
                    source_frame=self.source_frame,
                    target_frame=self.start_frame,
                    angle_offset=math.pi / 2.0,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout_per_side,
                    confirm_duration=confirm_duration,
                    cancel_on_success=True,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "SET_FINAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FINAL_DEPTH",
                SetDepthState(depth=0.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        return self.state_machine.execute(userdata)
