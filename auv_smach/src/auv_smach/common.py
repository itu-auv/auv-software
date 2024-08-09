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
import tf2_geometry_msgs
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse

from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
)


def transform_to_matrix(transform):
    trans = translation_matrix(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
    )
    rot = quaternion_matrix(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )
    return np.dot(trans, rot)


def multiply_transforms(transform1, transform2):
    matrix1 = transform_to_matrix(transform1)
    matrix2 = transform_to_matrix(transform2)
    return np.dot(matrix1, matrix2)


def matrix_to_transform(matrix):
    trans = translation_from_matrix(matrix)
    rot = quaternion_from_matrix(matrix)

    transform = TransformStamped()
    transform.transform.translation.x = trans[0]
    transform.transform.translation.y = trans[1]
    transform.transform.translation.z = trans[2]
    transform.transform.rotation.x = rot[0]
    transform.transform.rotation.y = rot[1]
    transform.transform.rotation.z = rot[2]
    transform.transform.rotation.w = rot[3]

    return transform


def concatenate_transforms(transform1, transform2):
    combined_matrix = multiply_transforms(transform1.transform, transform2.transform)
    return matrix_to_transform(combined_matrix)


class SetDepthState(smach_ros.ServiceState):
    def __init__(self, depth: float):
        set_depth_request = SetDepthRequest()
        set_depth_request.target_depth = depth

        smach_ros.ServiceState.__init__(
            self,
            "/taluy/set_depth",
            SetDepth,
            request=set_depth_request,
        )


class SetGateTransformsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "/create_gate_frames",
            Trigger,
            request=TriggerRequest(),
        )


class CancelAlignControllerState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "/taluy/control/align_frame/cancel",
            Trigger,
            request=TriggerRequest(),
        )


class SetAlignControllerTargetState(smach_ros.ServiceState):
    def __init__(self, source_frame: str, target_frame: str):
        align_request = AlignFrameControllerRequest()
        align_request.source_frame = source_frame
        align_request.target_frame = target_frame
        align_request.angle_offset = 0.0

        smach_ros.ServiceState.__init__(
            self,
            "/taluy/control/align_frame/start",
            AlignFrameController,
            request=align_request,
        )


class NavigateToFrameState(smach.State):
    def __init__(self, start_frame, end_frame, target_frame, n_turns=0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_frame = target_frame
        self.n_turns = n_turns
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(10)

        self.linear_velocity = rospy.get_param("/max_linear_velocity")
        self.angular_velocity = rospy.get_param("/max_angular_velocity")

    def execute(self, userdata):
        try:
            odom_to_start_transform = self.tf_buffer.lookup_transform(
                "odom", self.start_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            # Lookup the initial transform from start_frame to end_frame
            start_transform = self.tf_buffer.lookup_transform(
                self.start_frame, self.end_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            start_pos = np.array(
                [
                    0,  # Start position is the origin in the start_frame
                    0,
                ]
            )

            end_pos = np.array(
                [
                    start_transform.transform.translation.x,
                    start_transform.transform.translation.y,
                ]
            )

            start_orientation = 0.0  # Start with no rotation relative to start_frame

            end_orientation = transformations.euler_from_quaternion(
                [
                    start_transform.transform.rotation.x,
                    start_transform.transform.rotation.y,
                    start_transform.transform.rotation.z,
                    start_transform.transform.rotation.w,
                ]
            )[2]

            # Incorporate full rotations into the angular difference
            angular_diff = (end_orientation - start_orientation) + (
                2 * np.pi * self.n_turns
            )

            # Calculate the distance and ensure the angular velocity limit is respected
            distance = np.linalg.norm(end_pos - start_pos)
            duration_linear = distance / self.linear_velocity
            duration_angular = abs(angular_diff) / self.angular_velocity

            # The overall duration is the maximum of the two
            duration = max(duration_linear, duration_angular)
            num_steps = int(duration * 10)  # Number of steps based on the rate

            for i in range(num_steps):
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                # Linear interpolation for position
                t = float(i) / num_steps
                interp_pos = (1 - t) * start_pos + t * end_pos

                # Linear interpolation for orientation (in 2D, yaw only)
                interp_orientation = start_orientation + t * angular_diff
                quaternion = transformations.quaternion_from_euler(
                    0, 0, interp_orientation
                )

                # Broadcast the new transform
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.start_frame
                t.child_frame_id = self.target_frame
                t.transform.translation.x = interp_pos[0]
                t.transform.translation.y = interp_pos[1]
                t.transform.translation.z = 0.0  # Assume 2D movement (z = 0)
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]

                combined_transform = concatenate_transforms(odom_to_start_transform, t)
                combined_transform.header.stamp = rospy.Time.now()
                combined_transform.header.frame_id = "odom"
                combined_transform.child_frame_id = self.target_frame

                self.tf_broadcaster.sendTransform(combined_transform)
                self.rate.sleep()

            return "succeeded"

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup exception: {e}")
            return "aborted"
