import smach
import smach_ros
import rospy
import threading
import numpy as np
import tf2_ros
import tf.transformations as transformations

from std_srvs.srv import Trigger, TriggerRequest
from auv_msgs.srv import AlignFrameController, AlignFrameControllerRequest
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from auv_msgs.srv import SetDepth, SetDepthRequest

from auv_navigation.follow_path_action import follow_path_client

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


# ------------------- STATES -------------------


class SetDepthState(smach_ros.ServiceState):
    """
    Calls /taluy/set_depth with the requested depth.
    continuously publishes True to /taluy/enable topic
    whilst the state is running.

    Outcomes:
        - succeeded: The service call returned success.
        - preempted: The state was preempted.
        - aborted: The service call failed.
    """

    def __init__(self, depth: float, sleep_duration: float = 5.0):
        set_depth_request = SetDepthRequest()
        set_depth_request.target_depth = depth
        self.sleep_duration = sleep_duration

        super(SetDepthState, self).__init__(
            "set_depth",
            SetDepth,
            request=set_depth_request,
            outcomes=["succeeded", "preempted", "aborted"],
        )

        self._stop_publishing = threading.Event()

        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

    def _publish_enable_loop(self):
        publish_rate = rospy.get_param("~enable_rate", 20)
        rate = rospy.Rate(publish_rate)
        while not rospy.is_shutdown() and not self._stop_publishing.is_set():
            self.enable_pub.publish(Bool(True))
            rate.sleep()

    def execute(self, userdata):
        # if there's an immediate preempt
        if self.preempt_requested():
            rospy.logwarn("[SetDepthState] Preempt requested before execution.")
            self.service_preempt()
            return "preempted"

        # Call the service
        result = super(SetDepthState, self).execute(userdata)

        # Clear the stop flag
        self._stop_publishing.clear()
        # start publishing in the background thread
        pub_thread = threading.Thread(target=self._publish_enable_loop)
        pub_thread.start()
        # Wait for the specified sleep duration
        if self.sleep_duration > 0:
            rospy.sleep(self.sleep_duration)

        # signal the publishing thread to stop
        self._stop_publishing.set()
        pub_thread.join()

        return result


class LaunchTorpedoState(smach_ros.ServiceState):
    def __init__(self, id: int):
        smach_ros.ServiceState.__init__(
            self,
            f"torpedo_{id}/launch",
            Trigger,
            request=TriggerRequest(),
        )


class DropBallState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "ball_dropper/drop",
            Trigger,
            request=TriggerRequest(),
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
    def __init__(self, source_frame: str, target_frame: str):
        align_request = AlignFrameControllerRequest()
        align_request.source_frame = source_frame
        align_request.target_frame = target_frame
        align_request.angle_offset = 0.0

        smach_ros.ServiceState.__init__(
            self,
            "align_frame/start",
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

        self.linear_velocity = rospy.get_param("/smach/max_linear_velocity")
        self.angular_velocity = rospy.get_param("/smach/max_angular_velocity")

    def execute(self, userdata):
        try:
            odom_to_start_transform = self.tf_buffer.lookup_transform(
                "odom", self.start_frame, rospy.Time(0), rospy.Duration(1000.0)
            )
            # Lookup the initial transform from start_frame to end_frame
            start_transform = self.tf_buffer.lookup_transform(
                self.start_frame, self.end_frame, rospy.Time(0), rospy.Duration(1000.0)
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


class ClearObjectMapState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "clear_object_transforms",
            Trigger,
            request=TriggerRequest(),
        )


class ExecutePlannedPathsState(smach.State):
    """
    Uses the follow path action client to follow a set of planned paths.
    """

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=[
                "planned_paths"
            ],  # expects the input value under the name "planned_paths"
        )
        self._client = None

    def execute(self, userdata) -> str:
        """
        Args:
            userdata (smach.UserData): Contains `planned_paths` from the planning state.

        Returns:
            str: "succeeded" if execution was successful, otherwise "aborted" or "preempted".
        """
        if self._client is None:
            rospy.logdebug(
                "[ExecutePlannedPathsState] Initializing the FollowPathActionClient"
            )
            self._client = follow_path_client.FollowPathActionClient()

        # Check for preemption before proceeding
        if self.preempt_requested():
            rospy.logwarn("[ExecutePlannedPathsState] Preempt requested")
            return "preempted"
        try:
            planned_paths = userdata.planned_paths
            success = self._client.execute_paths(planned_paths)
            if success:
                rospy.logdebug(
                    "[ExecutePlannedPathsState] Planned paths executed successfully"
                )
                return "succeeded"
            else:
                rospy.logwarn(
                    "[ExecutePlannedPathsState] Execution of planned paths failed"
                )
                return "aborted"

        except Exception as e:
            rospy.logerr("[ExecutePlannedPathsState] Exception occurred: %s", str(e))
            return "aborted"


class SetFrameAtRadiusFacingFrameState(smach.State):
    def __init__(self, base_frame, center_frame, target_frame, radius=0.2):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_frame = base_frame
        self.center_frame = center_frame
        self.target_frame = target_frame
        self.radius = radius
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(10)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

    def execute(self, userdata):
        try:
            # Lookup the transform from center_frame to base_frame
            center_to_base_transform = self.tf_buffer.lookup_transform(
                self.center_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(10000.0),
            )

            # Calculate the position from the center to the base
            base_position = np.array(
                [
                    center_to_base_transform.transform.translation.x,
                    center_to_base_transform.transform.translation.y,
                ]
            )

            # Calculate the initial angle and the target position
            initial_angle = np.arctan2(base_position[1], base_position[0])
            interp_pos = self.radius * np.array(
                [np.cos(initial_angle), np.sin(initial_angle)]
            )

            # Calculate the orientation so the X-axis faces the center_frame
            facing_angle = initial_angle + np.pi  # Facing towards the center
            quaternion = transformations.quaternion_from_euler(0, 0, facing_angle)

            # Lookup the transform from odom to center_frame
            # odom_to_center_transform = self.tf_buffer.lookup_transform(
            #     "odom", self.center_frame, rospy.Time(0), rospy.Duration(1.0)
            # )

            # Convert the calculated position to a PointStamped in the center_frame
            point_in_center_frame = PointStamped()
            point_in_center_frame.header.frame_id = self.center_frame
            point_in_center_frame.header.stamp = center_to_base_transform.header.stamp
            point_in_center_frame.point.x = interp_pos[0]
            point_in_center_frame.point.y = interp_pos[1]
            point_in_center_frame.point.z = 0.0

            # Transform the point to the odom frame
            point_in_odom = self.tf_buffer.transform(point_in_center_frame, "odom")

            # Create the TransformStamped message for odom -> target_frame
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = self.target_frame
            t.transform.translation.x = point_in_odom.point.x
            t.transform.translation.y = point_in_odom.point.y
            t.transform.translation.z = point_in_odom.point.z
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]

            rospy.loginfo(f"Transform from odom to {self.target_frame}: {t}")

            # Call the set_object_transform service
            req = SetObjectTransformRequest()
            req.transform = t
            res = self.set_object_transform_service(req)

            if res.success:
                rospy.loginfo(f"SetObjectTransform succeeded: {res.message}")
            else:
                rospy.logwarn(f"SetObjectTransform failed: {res.message}")

            return "succeeded" if res.success else "aborted"

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup exception: {e}")
            return "aborted"
        except rospy.ServiceException as e:
            rospy.logwarn(f"Service call failed: {e}")
            return "aborted"


class SetFrameLookingAtState(smach.State):
    def __init__(self, base_frame, look_at_frame, target_frame):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_frame = base_frame
        self.look_at_frame = look_at_frame
        self.target_frame = target_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(10)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

    def execute(self, userdata):
        try:
            # Lookup the transform from base_frame to look_at_frame
            base_to_look_at_transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.look_at_frame,
                rospy.Time(0),
                rospy.Duration(10000.0),
            )

            # Calculate the direction vector from base_frame to look_at_frame
            direction_vector = np.array(
                [
                    base_to_look_at_transform.transform.translation.x,
                    base_to_look_at_transform.transform.translation.y,
                ]
            )

            # Calculate the angle to face the look_at_frame
            facing_angle = np.arctan2(direction_vector[1], direction_vector[0])
            quaternion = transformations.quaternion_from_euler(0, 0, facing_angle)

            # Create the TransformStamped message for base_frame -> target_frame
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self.base_frame
            t.child_frame_id = self.target_frame
            t.transform.translation.x = 0.0  # Placed exactly on the base_frame
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]

            rospy.loginfo(
                f"Transform from {self.base_frame} to {self.target_frame}: {t}"
            )

            # Call the set_object_transform service
            req = SetObjectTransformRequest()
            req.transform = t
            res = self.set_object_transform_service(req)

            if res.success:
                rospy.loginfo(f"SetObjectTransform succeeded: {res.message}")
            else:
                rospy.logwarn(f"SetObjectTransform failed: {res.message}")

            return "succeeded" if res.success else "aborted"

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup exception: {e}")
            return "aborted"
        except rospy.ServiceException as e:
            rospy.logwarn(f"Service call failed: {e}")
            return "aborted"