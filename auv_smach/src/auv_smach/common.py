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

from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetObjectTransform,
    SetObjectTransformRequest,
    SetObjectTransformResponse,
)

from auv_msgs.srv import SetDepth, SetDepthRequest

from auv_navigation.follow_path_action import follow_path_client


from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
    quaternion_from_euler,
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


class SetFrameLookingAtState(smach.State):
    def __init__(
        self,
        base_frame="taluy/base_link",
        target_frame="gate_search",
        look_at_frame="gate_blue_arrow_link",
        rotation_rate=0.5,
        full_rotation=False,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_frame = base_frame
        self.target_frame = target_frame
        self.look_at_frame = look_at_frame
        self.rotation_rate = rotation_rate
        self.full_rotation = full_rotation
        self.angle_threshold = np.radians(5)  # 5 degrees as threshold for alignment

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.set_object_transform = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.rate = rospy.Rate(10)

    def execute(self, userdata):
        start_time = rospy.Time.now().to_sec()
        start_angle = None
        completed_full_rotation = False

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            try:
                base_to_look = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.look_at_frame,
                    rospy.Time(0),
                    rospy.Duration(0.1),
                )

                # If full_rotation is true and we haven't completed a full rotation yet,
                # we continue rotating even if we found the look_at_frame
                if self.full_rotation and not completed_full_rotation:
                    elapsed = rospy.Time.now().to_sec() - start_time
                    current_angle = -self.rotation_rate * elapsed

                    if start_angle is None:
                        start_angle = current_angle

                    if abs(current_angle - start_angle) >= 2 * np.pi:
                        completed_full_rotation = True
                        rospy.loginfo("Completed full rotation as requested")
                    else:
                        # Continue with rotation (similar to the code in except block)
                        try:
                            odom_to_base = self.tf_buffer.lookup_transform(
                                "odom",
                                self.base_frame,
                                rospy.Time(0),
                                rospy.Duration(0.1),
                            )
                        except Exception as e:
                            rospy.logwarn(
                                "TF lookup failed for %s: %s", self.base_frame, e
                            )
                            return "aborted"

                        t = TransformStamped()
                        t.header.stamp = rospy.Time.now()
                        t.header.frame_id = "odom"
                        t.child_frame_id = self.target_frame
                        t.transform.translation.x = odom_to_base.transform.translation.x
                        t.transform.translation.y = odom_to_base.transform.translation.y
                        t.transform.translation.z = odom_to_base.transform.translation.z
                        q = quaternion_from_euler(0, 0, current_angle)
                        (
                            t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                            t.transform.rotation.w,
                        ) = q

                        req = SetObjectTransformRequest(transform=t)
                        try:
                            res = self.set_object_transform(req)
                            if not res.success:
                                rospy.logwarn(
                                    "SetObjectTransform failed: %s", res.message
                                )
                        except rospy.ServiceException as e:
                            rospy.logwarn("Service call failed: %s", e)
                            return "aborted"

                        self.rate.sleep()
                        continue

                # Normal behavior when look_at_frame is found and we're not doing a full rotation
                # or we've already completed the full rotation
                while not rospy.is_shutdown():
                    if self.preempt_requested():
                        self.service_preempt()
                        return "preempted"

                    vec = np.array(
                        [
                            base_to_look.transform.translation.x,
                            base_to_look.transform.translation.y,
                        ]
                    )
                    facing_angle = np.arctan2(vec[1], vec[0])
                    q = quaternion_from_euler(0, 0, facing_angle)

                    t = TransformStamped()
                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = self.base_frame
                    t.child_frame_id = self.target_frame
                    t.transform.translation.x = 0.0
                    t.transform.translation.y = 0.0
                    t.transform.translation.z = 0.0
                    (
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ) = q

                    req = SetObjectTransformRequest(transform=t)
                    try:
                        res = self.set_object_transform(req)
                        if not res.success:
                            rospy.logwarn("SetObjectTransform failed: %s", res.message)
                            return "aborted"
                    except rospy.ServiceException as e:
                        rospy.logwarn("Service call failed: %s", e)
                        return "aborted"

                    # Get current robot orientation
                    try:
                        base_to_target = self.tf_buffer.lookup_transform(
                            self.base_frame,
                            self.target_frame,
                            rospy.Time(0),
                            rospy.Duration(0.05),
                        )

                        # Extract current rotation of the robot
                        current_quat = [
                            base_to_target.transform.rotation.x,
                            base_to_target.transform.rotation.y,
                            base_to_target.transform.rotation.z,
                            base_to_target.transform.rotation.w,
                        ]
                        _, _, current_yaw = euler_from_quaternion(current_quat)

                        # Calculate angle difference
                        angle_diff = abs(current_yaw - facing_angle)
                        angle_diff = min(
                            angle_diff, 2 * np.pi - angle_diff
                        )  # Take the smaller angle

                        # If angle difference is less than threshold, we're aligned
                        if angle_diff < self.angle_threshold:
                            rospy.loginfo(
                                "Target aligned within %s radians (%s degrees)",
                                self.angle_threshold,
                                np.degrees(self.angle_threshold),
                            )
                            return "succeeded"
                    except Exception as e:
                        rospy.logwarn("Error checking alignment: %s", e)
                        break

                    try:
                        base_to_look = self.tf_buffer.lookup_transform(
                            self.base_frame,
                            self.look_at_frame,
                            rospy.Time(0),
                            rospy.Duration(0.05),
                        )
                    except Exception:
                        rospy.logdebug(
                            "look_at_frame cannot found anymore, trying again..."
                        )
                        break

                    self.rate.sleep()

                return "succeeded"

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                elapsed = rospy.Time.now().to_sec() - start_time
                current_angle = -self.rotation_rate * elapsed

                try:
                    odom_to_base = self.tf_buffer.lookup_transform(
                        "odom", self.base_frame, rospy.Time(0), rospy.Duration(0.1)
                    )
                except Exception as e:
                    rospy.logwarn("TF lookup failed for %s: %s", self.base_frame, e)
                    return "aborted"

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "odom"
                t.child_frame_id = self.target_frame
                t.transform.translation.x = odom_to_base.transform.translation.x
                t.transform.translation.y = odom_to_base.transform.translation.y
                t.transform.translation.z = odom_to_base.transform.translation.z
                q = quaternion_from_euler(0, 0, current_angle)
                (
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ) = q

                req = SetObjectTransformRequest(transform=t)
                try:
                    res = self.set_object_transform(req)
                    if not res.success:
                        rospy.logwarn("SetObjectTransform failed: %s", res.message)
                except rospy.ServiceException as e:
                    rospy.logwarn("Service call failed: %s", e)
                    return "aborted"

                if abs(current_angle) >= 2 * np.pi:
                    rospy.loginfo(
                        "completed a full rotation, could not find look_at_frame"
                    )
                    return "succeeded"

                self.rate.sleep()


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
