import smach
import smach_ros
import rospy
import threading
import numpy as np
import tf2_ros
import tf.transformations as transformations
import math
import angles

from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import AlignFrameController, AlignFrameControllerRequest
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped

from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetObjectTransform,
    SetObjectTransformRequest,
    SetObjectTransformResponse,
)

from auv_msgs.srv import SetDepth, SetDepthRequest
from auv_msgs.srv import VisualServoing, VisualServoingRequest

from auv_navigation.follow_path_action import follow_path_client

from auv_navigation.path_planning.path_planners import PathPlanners

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
    quaternion_from_euler,
    euler_from_quaternion,
)


class CancelAlignControllerState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "align_frame/cancel",
            Trigger,
            request=TriggerRequest(),
        )


class SetAlignControllerTargetState(smach_ros.ServiceState):
    def __init__(
        self,
        source_frame: str,
        target_frame: str,
        keep_orientation: bool = False,
        angle_offset: float = 0.0,
    ):
        align_request = AlignFrameControllerRequest()
        align_request.source_frame = source_frame
        align_request.target_frame = target_frame
        align_request.angle_offset = angle_offset
        align_request.keep_orientation = keep_orientation

        smach_ros.ServiceState.__init__(
            self,
            "align_frame/start",
            AlignFrameController,
            request=align_request,
        )


class CheckAlignmentState(smach.State):
    def __init__(
        self,
        source_frame,
        target_frame,
        dist_threshold,
        yaw_threshold,
        timeout,
        angle_offset=0.0,
        confirm_duration=0.0,
        keep_orientation=False,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "aborted", "preempted"])
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.dist_threshold = dist_threshold
        self.yaw_threshold = yaw_threshold
        self.timeout = timeout
        self.angle_offset = angle_offset
        self.confirm_duration = confirm_duration
        self.keep_orientation = keep_orientation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.rate = rospy.Rate(10)

    def get_error(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation

            dist_error = math.sqrt(trans.x**2 + trans.y**2)

            _, _, yaw = transformations.euler_from_quaternion(
                (rot.x, rot.y, rot.z, rot.w)
            )
            yaw_error = abs(angles.normalize_angle(yaw + self.angle_offset))

            return dist_error, yaw_error
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"CheckAlignmentState: TF lookup failed: {e}")
            return None, None

    def is_aligned_distance_only(self, dist_error):
        return dist_error < self.dist_threshold

    def is_aligned_distance_and_yaw(self, dist_error, yaw_error):
        return dist_error < self.dist_threshold and yaw_error < self.yaw_threshold

    def execute(self, userdata):
        start_time = rospy.Time.now()
        first_success_time = None

        while (rospy.Time.now() - start_time).to_sec() < self.timeout:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            dist_error, yaw_error = self.get_error()

            if dist_error is not None and yaw_error is not None:
                rospy.loginfo_throttle(
                    1.0,
                    f"Alignment check: dist_error={dist_error:.2f}m, yaw_error={yaw_error:.2f}rad",
                )
                if self.keep_orientation:
                    aligned = self.is_aligned_distance_only(dist_error)
                else:
                    aligned = self.is_aligned_distance_and_yaw(dist_error, yaw_error)

                if aligned:
                    if self.confirm_duration == 0.0:
                        rospy.loginfo("CheckAlignmentState: Alignment successful.")
                        return "succeeded"
                    if first_success_time is None:
                        first_success_time = rospy.Time.now()
                    if (
                        rospy.Time.now() - first_success_time
                    ).to_sec() >= self.confirm_duration:
                        rospy.loginfo(
                            f"CheckAlignmentState: Alignment successful for {self.confirm_duration} seconds."
                        )
                        return "succeeded"
                else:
                    first_success_time = None

            self.rate.sleep()

        rospy.logwarn("CheckAlignmentState: Timeout reached.")
        return "succeeded"


class AlignFrame(smach.StateMachine):
    def __init__(
        self,
        source_frame,
        target_frame,
        angle_offset=0.0,
        dist_threshold=0.1,
        yaw_threshold=0.1,
        timeout=30.0,
        cancel_on_success=False,
        confirm_duration=0.0,
        keep_orientation=False,
    ):
        super().__init__(outcomes=["succeeded", "aborted", "preempted"])

        with self:
            smach.StateMachine.add(
                "REQUEST_ALIGNMENT",
                SetAlignControllerTargetState(
                    source_frame=source_frame,
                    target_frame=target_frame,
                    angle_offset=angle_offset,
                    keep_orientation=keep_orientation,
                ),
                transitions={
                    "succeeded": "WATCH_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "WATCH_ALIGNMENT",
                CheckAlignmentState(
                    source_frame,
                    target_frame,
                    dist_threshold,
                    yaw_threshold,
                    timeout,
                    angle_offset,
                    confirm_duration,
                    keep_orientation=keep_orientation,
                ),
                transitions={
                    "succeeded": (
                        "CANCEL_ALIGNMENT_ON_SUCCESS"
                        if cancel_on_success
                        else "succeeded"
                    ),
                    "aborted": "CANCEL_ALIGNMENT_ON_FAIL",
                    "preempted": "CANCEL_ALIGNMENT_ON_PREEMPT",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGNMENT_ON_SUCCESS",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGNMENT_ON_FAIL",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "aborted",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGNMENT_ON_PREEMPT",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "preempted",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
