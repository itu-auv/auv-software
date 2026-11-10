import smach
import smach_ros
import rclpy
from auv_common_lib.ros import get_param, service_client, get_node, ROSException, Rate
from rclpy.duration import Duration
from rclpy.time import Time
import threading
import numpy as np
import tf2_ros
import tf_transformations as transformations
import math
import angles
from auv_smach.tf_utils import get_tf_buffer, get_base_link, reset_tf_buffer
from auv_common_lib.transform import lookup_fresh_transform

from std_srvs.srv import Trigger, SetBool
from auv_msgs.srv import AlignFrameController
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped, PoseStamped
from auv_msgs.action import FollowPath

from auv_msgs.srv import (
    PlanPath,
    SetDepth,
    SetObjectTransform,
    SetDetectionFocus,
)

from geometry_msgs.msg import TransformStamped
from auv_msgs.srv import SetDepth

from auv_navigation.follow_path_action import follow_path_client
from auv_msgs.srv import PlanPath
from auv_navigation.path_planning.path_planners import PathPlanners

from auv_msgs.srv import PlanPath
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from tf_transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
    quaternion_from_euler,
    euler_from_quaternion,
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


class SetDepthState(smach.State):
    """
    Calls /taluy/set_depth with the requested depth and waits until
    the robot reaches the target depth within a threshold for a confirm duration.

    Uses TF to get current robot depth by looking up base_link in the target frame.

    Args:
        depth: Target depth in meters
        depth_threshold: How close to target depth to consider "reached" (default: 0.1m)
        confirm_duration: How long to stay within threshold to confirm (default: 1.0s)
        timeout: Maximum time to wait (default: 30.0s), returns succeeded on timeout
        frame_id: Frame ID for the depth (default: "odom")
        max_velocity: Optional max velocity for Z axis (default: 0.0, uses default)

    Outcomes:
        - succeeded: Robot reached target depth or timeout reached
        - preempted: The state was preempted
        - aborted: Service call failed
    """

    def __init__(
        self,
        depth: float,
        depth_threshold: float = 0.1,
        confirm_duration: float = 1.0,
        timeout: float = 20.0,
        frame_id: str = "odom",
        max_velocity: float = 0.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.target_depth = depth
        self.depth_threshold = depth_threshold
        self.confirm_duration = confirm_duration
        self.timeout = timeout
        self.frame_id = frame_id
        self.max_velocity = max_velocity

        self._stop_publishing = threading.Event()
        self.enable_pub = get_node().create_publisher(Bool, "enable", 1)

        self.tf_buffer = get_tf_buffer()

        self.base_frame = get_base_link()

    def _publish_enable_loop(self):
        publish_rate = get_param(get_node(), "~enable_rate", 20)
        rate = Rate(get_node(), publish_rate)
        while rclpy.ok() and not self._stop_publishing.is_set():
            self.enable_pub.publish(Bool(data=True))
            rate.sleep()

    def _get_current_depth(self):
        """Get current robot depth by looking up base_link in frame_id."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frame_id, self.base_frame, Time(), Duration(seconds=4.0)
            )
            return transform.transform.translation.z
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            get_node().get_logger().warning(
                f"[SetDepthState] TF lookup failed: {e}", throttle_duration_sec=5.0
            )
            return None

    def execute(self, userdata):
        if self.preempt_requested():
            get_node().get_logger().warning(
                "[SetDepthState] Preempt requested before execution."
            )
            self.service_preempt()
            return "preempted"

        try:
            set_depth_service = service_client(get_node(), SetDepth, "set_depth")
            set_depth_service.wait_for_service(timeout=5.0)
            request = SetDepth.Request()
            request.target_depth = self.target_depth
            request.frame_id = self.frame_id
            request.max_velocity = self.max_velocity
            response = set_depth_service(request)

            if not response.success:
                get_node().get_logger().error(
                    f"[SetDepthState] Service call failed: {response.message}"
                )
                return "aborted"
        except (Exception, ROSException) as e:
            get_node().get_logger().error(f"[SetDepthState] Service call failed: {e}")
            return "aborted"

        # Start publishing enable in background
        self._stop_publishing.clear()
        pub_thread = threading.Thread(target=self._publish_enable_loop)
        pub_thread.start()

        # Wait for depth to be reached
        start_time = get_node().get_clock().now()
        confirm_start_time = None
        rate = Rate(get_node(), 20)

        get_node().get_logger().info(
            f"[SetDepthState] Waiting for depth {self.target_depth}m in {self.frame_id} "
            f"(threshold={self.depth_threshold}m, timeout={self.timeout}s)"
        )

        while rclpy.ok():
            # Check for preempt
            if self.preempt_requested():
                get_node().get_logger().warning(
                    "[SetDepthState] Preempt requested during execution."
                )
                self._stop_publishing.set()
                pub_thread.join()
                self.service_preempt()
                return "preempted"

            # Check for timeout
            elapsed = (get_node().get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > self.timeout:
                get_node().get_logger().warning(
                    f"[SetDepthState] Timeout after {self.timeout}s. Continuing anyway."
                )
                self._stop_publishing.set()
                pub_thread.join()
                return "succeeded"

            current_depth = self._get_current_depth()

            # Check if depth is within threshold
            if current_depth is not None:
                depth_error = abs(current_depth - self.target_depth)

                get_node().get_logger().info(
                    f"[SetDepthState] current={current_depth:.3f}m, "
                    f"target={self.target_depth:.3f}m, error={depth_error:.3f}m",
                    throttle_duration_sec=1.0,
                )

                if depth_error <= self.depth_threshold:
                    if confirm_start_time is None:
                        confirm_start_time = get_node().get_clock().now()
                        get_node().get_logger().info(
                            f"[SetDepthState] Depth within threshold, confirming..."
                        )

                    confirm_elapsed = (
                        get_node().get_clock().now() - confirm_start_time
                    ).nanoseconds / 1e9
                    if confirm_elapsed >= self.confirm_duration:
                        get_node().get_logger().info(
                            f"[SetDepthState] Depth confirmed at {current_depth:.3f}m"
                        )
                        self._stop_publishing.set()
                        pub_thread.join()
                        return "succeeded"
                else:
                    if confirm_start_time is not None:
                        get_node().get_logger().info(
                            f"[SetDepthState] Drifted out of threshold, resetting confirm timer"
                        )
                    confirm_start_time = None

            rate.sleep()

        self._stop_publishing.set()
        pub_thread.join()
        return "succeeded"


class DropBallState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            get_node(),
            "ball_dropper/drop",
            Trigger,
            request=Trigger.Request(),
        )


class CancelAlignControllerState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            get_node(),
            "align_frame/cancel",
            Trigger,
            request=Trigger.Request(),
        )


class SetAlignControllerTargetState(smach_ros.ServiceState):
    def __init__(
        self,
        source_frame: str,
        target_frame: str,
        keep_orientation: bool = False,
        angle_offset: float = 0.0,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        use_depth: bool = False,
        closest_yaw: bool = False,
        closest_yaw_180: bool = False,
    ):
        if closest_yaw and closest_yaw_180:
            raise ValueError("closest_yaw and closest_yaw_180 cannot both be enabled")
        align_request = AlignFrameController.Request()
        align_request.source_frame = source_frame
        align_request.target_frame = target_frame
        align_request.angle_offset = angle_offset
        align_request.keep_orientation = keep_orientation
        align_request.use_depth = use_depth
        align_request.closest_yaw = closest_yaw
        align_request.closest_yaw_180 = closest_yaw_180
        if max_linear_velocity is not None:
            align_request.max_linear_velocity = max_linear_velocity
        if max_angular_velocity is not None:
            align_request.max_angular_velocity = max_angular_velocity

        smach_ros.ServiceState.__init__(
            self,
            get_node(),
            "align_frame/start",
            AlignFrameController,
            request=align_request,
            response_cb=self.response_cb,
        )

    @staticmethod
    def response_cb(userdata, response):
        if not response.success:
            get_node().get_logger().warning(
                "SetAlignControllerTargetState failed: %s" % (response.message,)
            )
            return "aborted"

        return "succeeded"


class NavigateToFrameState(smach.State):
    def __init__(self, start_frame, end_frame, target_frame, n_turns=0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_frame = target_frame
        self.n_turns = n_turns
        self.tf_buffer = get_tf_buffer()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(get_node())
        self.rate = Rate(get_node(), 10)

        self.linear_velocity = get_param(get_node(), "/smach/max_linear_velocity")
        self.angular_velocity = get_param(get_node(), "/smach/max_angular_velocity")

    def execute(self, userdata):
        try:
            odom_to_start_transform = self.tf_buffer.lookup_transform(
                "odom", self.start_frame, Time(), Duration(seconds=1000.0)
            )
            # Lookup the initial transform from start_frame to end_frame
            start_transform = self.tf_buffer.lookup_transform(
                self.start_frame, self.end_frame, Time(), Duration(seconds=1000.0)
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
                t.header.stamp = get_node().get_clock().now().to_msg()
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
                combined_transform.header.stamp = get_node().get_clock().now().to_msg()
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
            get_node().get_logger().warning(f"TF lookup exception: {e}")
            return "aborted"


class RotationState(smach.State):
    def __init__(
        self,
        source_frame,
        look_at_frame,
        rotation_speed=0.2,
        full_rotation=False,
        full_rotation_timeout=25.0,
        rate_hz=10,
        rotation_radian=2 * math.pi,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.odom_topic = "odometry"
        self.cmd_vel_topic = "cmd_vel"
        self.rotation_speed = rotation_speed
        self.rotation_radian = rotation_radian
        self.odom_data = False
        self.yaw = None
        self.yaw_prev = None
        self.total_yaw = 0.0
        self.rate = Rate(get_node(), rate_hz)
        self.active = True

        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.full_rotation = full_rotation
        self.full_rotation_timeout = full_rotation_timeout
        self.tf_buffer = get_tf_buffer()

        self.sub = get_node().create_subscription(
            Odometry, self.odom_topic, self.odom_cb, 10
        )
        self.pub = get_node().create_publisher(Twist, self.cmd_vel_topic, 1)

        self.enable_pub = get_node().create_publisher(Bool, "enable", 1)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        orientation_list = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.odom_data = True
        self.yaw = yaw

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def is_transform_available(self):
        try:
            lookup_fresh_transform(
                self.tf_buffer,
                get_node(),
                self.source_frame,
                self.look_at_frame,
                Duration(seconds=get_param(get_node(), "~tf_lookup_timeout", 0.2)),
                Duration(seconds=get_param(get_node(), "~tf_freshness_threshold", 0.4)),
            )
            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            get_node().get_logger().debug(f"RotationState: Transform check failed: {e}")
            return False

    def execute(self, userdata):
        while rclpy.ok() and not self.odom_data:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            self.rate.sleep()

        self.yaw_prev = self.yaw
        self.total_yaw = 0.0
        twist = Twist()
        twist.angular.z = self.rotation_speed
        self.active = True
        rotation_start_time = get_node().get_clock().now()

        transform_found = self.is_transform_available()
        if transform_found and not self.full_rotation:
            get_node().get_logger().info(
                "RotationState: transform already available, no need to rotate"
            )
            return "succeeded"

        while rclpy.ok() and self.total_yaw < self.rotation_radian:
            if self.preempt_requested():
                twist.angular.z = 0.0
                self.pub.publish(twist)
                self.service_preempt()
                return "preempted"

            if (
                self.full_rotation
                and (get_node().get_clock().now() - rotation_start_time).nanoseconds
                / 1e9
                > self.full_rotation_timeout
            ):
                get_node().get_logger().warning(
                    f"RotationState: Timeout reached after {self.full_rotation_timeout} seconds during full rotation."
                )
                twist.angular.z = 0.0
                self.pub.publish(twist)
                break

            self.enable_pub.publish(Bool(data=True))

            if not self.full_rotation and self.is_transform_available():
                twist.angular.z = 0.0
                self.pub.publish(twist)
                get_node().get_logger().info(
                    "RotationState: transform found during rotation, stopping rotation"
                )
                return "succeeded"

            self.pub.publish(twist)

            if self.yaw is not None and self.yaw_prev is not None:
                dyaw = RotationState.normalize_angle(self.yaw - self.yaw_prev)
                self.total_yaw += abs(dyaw)
                self.yaw_prev = self.yaw

            self.rate.sleep()

        twist.angular.z = 0.0
        self.pub.publish(twist)

        get_node().get_logger().info(
            f"RotationState: completed full rotation. Total yaw: {self.total_yaw}"
        )

        if self.is_transform_available():
            return "succeeded"
        else:
            get_node().get_logger().warning(
                "RotationState: completed full rotation but no transform found between %s and %s"
                % (self.source_frame, self.look_at_frame)
            )
            return "aborted"


class SetFrameLookingAtState(smach.State):
    def __init__(self, source_frame, look_at_frame, alignment_frame, duration_time=3.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.alignment_frame = alignment_frame
        self.duration_time = duration_time
        self.tf_buffer = get_tf_buffer()
        self.rate = Rate(get_node(), 10)
        self.set_object_transform_service = service_client(
            get_node(), SetObjectTransform, "set_object_transform"
        )

    def execute(self, userdata):
        start_time = get_node().get_clock().now()
        end_time = start_time + Duration(seconds=self.duration_time)

        while rclpy.ok() and get_node().get_clock().now() < end_time:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            try:
                base_to_look_at_transform = self.tf_buffer.lookup_transform(
                    self.source_frame,
                    self.look_at_frame,
                    Time(),
                    Duration(seconds=4.0),
                )

                direction_vector = np.array(
                    [
                        base_to_look_at_transform.transform.translation.x,
                        base_to_look_at_transform.transform.translation.y,
                    ]
                )

                facing_angle = np.arctan2(direction_vector[1], direction_vector[0])
                quaternion = transformations.quaternion_from_euler(0, 0, facing_angle)

                t = TransformStamped()
                t.header.stamp = get_node().get_clock().now().to_msg()
                t.header.frame_id = self.source_frame
                t.child_frame_id = self.alignment_frame
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]

                req = SetObjectTransform.Request()
                req.transform = t
                res = self.set_object_transform_service(req)

                if not res.success:
                    get_node().get_logger().warning(
                        f"SetObjectTransform failed: {res.message}"
                    )

                time_remaining = (
                    end_time - get_node().get_clock().now()
                ).nanoseconds / 1e9
                get_node().get_logger().info(
                    f"Looking at {self.look_at_frame}. Time remaining: {time_remaining:.2f}s",
                    throttle_duration_sec=1.0,
                )

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                get_node().get_logger().warning(f"TF lookup exception: {e}")

            except Exception as e:
                get_node().get_logger().warning(f"Service call failed: {e}")
                return "aborted"

            self.rate.sleep()

        get_node().get_logger().info(
            f"Successfully looked at {self.look_at_frame} for {self.duration_time} seconds"
        )
        return "succeeded"


class ClearObjectMapState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            get_node(),
            "clear_object_transforms",
            Trigger,
            request=Trigger.Request(),
            response_cb=self.response_cb,
        )

    @staticmethod
    def response_cb(userdata, response):
        if not response.success:
            get_node().get_logger().warning(
                "ClearObjectMapState: clear_object_transforms failed"
            )
            return "aborted"

        reset_tf_buffer()
        return "succeeded"


class ClearKDEMapState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            get_node(),
            "kde_map/clear",
            Trigger,
            request=Trigger.Request(),
            response_cb=self.response_cb,
        )

    @staticmethod
    def response_cb(userdata, response):
        if not response.success:
            get_node().get_logger().warning("ClearKDEMapState: kde_map/clear failed")
            return "aborted"

        return "succeeded"


class SetDetectionState(smach_ros.ServiceState):
    """
    Calls the service to enable or disable camera detections.
    """

    def __init__(self, camera_name: str, enable: bool):
        if camera_name not in ["front", "bottom", "torpedo", "segment"]:
            raise ValueError(
                "camera_name must be 'front', 'bottom', 'torpedo', or 'segment'"
            )

        service_name = f"enable_{camera_name}_camera_detections"
        request = SetBool.Request(data=enable)

        super(SetDetectionState, self).__init__(
            get_node(),
            service_name,
            SetBool,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class SetDetectionFocusBottomState(smach_ros.ServiceState):
    """
    Calls the service to set the focus for the bottom camera detections.
    """

    def __init__(self, focus_object: str):
        service_name = "set_bottom_camera_focus"
        request = SetDetectionFocus.Request(focus_object=focus_object)

        super(SetDetectionFocusBottomState, self).__init__(
            get_node(),
            service_name,
            SetDetectionFocus,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class SetDetectionFocusState(smach_ros.ServiceState):
    """
    Calls the service to set the focus for the front camera detections.
    """

    def __init__(self, focus_object: str):
        service_name = "set_front_camera_focus"
        request = SetDetectionFocus.Request(focus_object=focus_object)

        super(SetDetectionFocusState, self).__init__(
            get_node(),
            service_name,
            SetDetectionFocus,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class ExecutePathState(smach.State):
    """
    Uses the follow path action client to follow a set of planned paths.
    """

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self._client = None

    def execute(self, userdata) -> str:
        """
        Returns:
            str: "succeeded" if execution was successful, otherwise "aborted" or "preempted".
        """
        if self._client is None:
            get_node().get_logger().debug(
                "[ExecutePathState] Initializing the FollowPathActionClient"
            )
            self._client = follow_path_client.FollowPathActionClient(get_node())

        # Check for preemption before proceeding
        if self.preempt_requested():
            get_node().get_logger().warning(
                "[ExecutePathState] Preempt requested before execution"
            )
            self.service_preempt()
            return "preempted"

        try:
            goal = FollowPath.Goal()
            goal.paths = []
            get_node().get_logger().debug("[ExecutePathState] Sending paths goal...")
            send_goal_future = self._client.client.send_goal_async(goal)
            goal_handle = None
            result_future = None

            while rclpy.ok():
                if self.preempt_requested():
                    get_node().get_logger().warning(
                        "[ExecutePathState] Preempt requested, cancelling goal"
                    )
                    if goal_handle is not None:
                        goal_handle.cancel_goal_async()
                    else:
                        send_goal_future.add_done_callback(
                            lambda future: future.result().cancel_goal_async()
                        )
                    self.service_preempt()
                    return "preempted"

                if goal_handle is None and send_goal_future.done():
                    goal_handle = send_goal_future.result()
                    if not goal_handle.accepted:
                        get_node().get_logger().warning(
                            "[ExecutePathState] Execution of planned paths failed"
                        )
                        return "aborted"
                    result_future = goal_handle.get_result_async()

                if result_future is not None and result_future.done():
                    result = result_future.result().result
                    if result and result.success:
                        get_node().get_logger().debug(
                            "[ExecutePathState] Planned paths executed successfully"
                        )
                        return "succeeded"
                    else:
                        get_node().get_logger().warning(
                            "[ExecutePathState] Execution of planned paths failed"
                        )
                        return "aborted"

                get_node().get_clock().sleep_for(Duration(seconds=0.1))

            return "aborted"

        except Exception as e:
            get_node().get_logger().error(
                "[ExecutePathState] Exception occurred: %s" % (str(e),)
            )
            return "aborted"


class SearchForPropState(smach.StateMachine):
    """
    1. SetAlignTarget: Sets alignment to rotating frame.
    2. RotationState: Rotates to find a prop's frame using CreateRotatingFrameState.
    3. AimToProp: Aligns to the prop while looking at it.
    4. CancelAlignControllerState: Cancels the align controller target.
    """

    def __init__(
        self,
        look_at_frame: str,
        alignment_frame: str,
        full_rotation: bool = False,
        timeout: float = 30.0,
        source_frame: str = None,
        rotation_speed: float = 0.4,
        confirm_duration: float = 2.0,
    ):
        """
        Args:
            look_at_frame (str): The frame to rotate towards and look at.
            alignment_frame (str): The frame to set as the align controller target
                                and whose pose is set by SetFrameLookingAtState.
            full_rotation (bool): Whether to perform a full 360-degree rotation
                                  or stop when look_at_frame is found.
            timeout (float): Timeout for the AimToProp state.
            source_frame (str): The base frame of the vehicle. Defaults to get_base_link().
            rotation_speed (float): The angular velocity for rotation (default: 0.3).
        """
        if source_frame is None:
            source_frame = get_base_link()

        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        rotating_frame_name = f"{look_at_frame}_search_rotating"
        rotation_period = 2.0 * math.pi / rotation_speed if rotation_speed > 0 else 15.0

        with self:
            smach.StateMachine.add(
                "ALIGN_AND_ROTATE",
                AlignAndCreateRotatingFrame(
                    source_frame=source_frame,
                    rotating_frame_name=rotating_frame_name,
                    rotation_period=rotation_period,
                    look_at_frame=look_at_frame,
                    full_rotation=full_rotation,
                ),
                transitions={
                    "succeeded": "AIM_TO_PROP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "AIM_TO_PROP",
                AimToProp(
                    source_frame=source_frame,
                    look_at_frame=look_at_frame,
                    alignment_frame=alignment_frame,
                    timeout=timeout,
                    confirm_duration=confirm_duration,
                ),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER_TARGET",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class AimToProp(smach.Concurrence):
    """
    Runs SetFrameLookingAtState and AlignFrame concurrently.
    Terminates when AlignFrame finishes.
    """

    def __init__(
        self,
        source_frame,
        look_at_frame,
        alignment_frame,
        timeout=60.0,
        cancel_on_success=False,
        confirm_duration=2.0,
    ):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            outcome_map={
                "succeeded": {"ALIGN_FRAME": "succeeded"},
                "preempted": {"ALIGN_FRAME": "preempted"},
                "aborted": {"ALIGN_FRAME": "aborted"},
            },
            child_termination_cb=lambda outcome_map: True,  # Terminate if any child terminates
        )

        with self:
            smach.Concurrence.add(
                "LOOK_AT",
                SetFrameLookingAtState(
                    source_frame=source_frame,
                    look_at_frame=look_at_frame,
                    alignment_frame=alignment_frame,
                    duration_time=timeout,
                ),
            )

            smach.Concurrence.add(
                "ALIGN_FRAME",
                AlignFrame(
                    source_frame=source_frame,
                    target_frame=alignment_frame,
                    timeout=timeout,
                    cancel_on_success=cancel_on_success,
                    confirm_duration=confirm_duration,
                ),
            )


class PlanPathToSingleFrameState(smach.State):
    def __init__(self, tf_buffer, target_frame: str, source_frame: str = None):
        if source_frame is None:
            source_frame = get_base_link()
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.tf_buffer = tf_buffer
        self.target_frame = target_frame
        self.source_frame = source_frame

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                get_node().get_logger().warning(
                    f"[PlanPathToSingleFrameState] Preempt requested for path to {self.target_frame}"
                )
                return "preempted"

            path_planners = PathPlanners(get_node(), self.tf_buffer)
            path = path_planners.straight_path_to_frame(
                source_frame=self.source_frame,
                target_frame=self.target_frame,
                num_waypoints=50,
            )

            if path is None:
                get_node().get_logger().error(
                    f"[PlanPathToSingleFrameState] Failed to plan path to {self.target_frame}"
                )
                return "aborted"

            userdata.planned_paths = [path]  #
            get_node().get_logger().info(
                f"[PlanPathToSingleFrameState] Successfully planned path to {self.target_frame}"
            )
            return "succeeded"

        except Exception as e:
            get_node().get_logger().error(
                f"[PlanPathToSingleFrameState] Error planning path to {self.target_frame}: {e}"
            )
            return "aborted"


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
        use_frame_depth=False,
        closest_yaw=False,
        closest_yaw_180=False,
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
        self.use_frame_depth = use_frame_depth
        self.closest_yaw = closest_yaw
        self.closest_yaw_180 = closest_yaw_180
        self.tf_buffer = get_tf_buffer()
        self.rate = Rate(get_node(), 10)

    def get_error(self):
        try:
            transform = lookup_fresh_transform(
                self.tf_buffer,
                get_node(),
                self.source_frame,
                self.target_frame,
                Duration(seconds=get_param(get_node(), "~tf_lookup_timeout", 0.2)),
                Duration(seconds=get_param(get_node(), "~tf_freshness_threshold", 0.4)),
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation

            dist_error = math.sqrt(trans.x**2 + trans.y**2)
            if self.use_frame_depth:
                dist_error = math.sqrt(dist_error**2 + trans.z**2)

            _, _, yaw = transformations.euler_from_quaternion(
                (rot.x, rot.y, rot.z, rot.w)
            )
            yaw_with_offset = abs(
                angles.shortest_angular_distance(0, yaw + self.angle_offset)
            )
            if self.closest_yaw:
                yaw_error = min(
                    yaw_with_offset,
                    abs(
                        angles.shortest_angular_distance(
                            0, yaw + self.angle_offset + math.pi / 2
                        )
                    ),
                    abs(
                        angles.shortest_angular_distance(
                            0, yaw + self.angle_offset + math.pi
                        )
                    ),
                    abs(
                        angles.shortest_angular_distance(
                            0, yaw + self.angle_offset + 3 * math.pi / 2
                        )
                    ),
                )
            elif self.closest_yaw_180:
                yaw_error = min(
                    yaw_with_offset,
                    abs(
                        angles.shortest_angular_distance(
                            0, yaw + self.angle_offset + math.pi
                        )
                    ),
                )
            else:
                yaw_error = yaw_with_offset

            return dist_error, yaw_error
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            get_node().get_logger().warning(
                f"CheckAlignmentState: TF lookup failed: {e}", throttle_duration_sec=3.0
            )
            return None, None

    def is_aligned_distance_only(self, dist_error):
        return dist_error < self.dist_threshold

    def is_aligned_distance_and_yaw(self, dist_error, yaw_error):
        return dist_error < self.dist_threshold and yaw_error < self.yaw_threshold

    def execute(self, userdata):
        start_time = get_node().get_clock().now()
        first_success_time = None

        while (
            get_node().get_clock().now() - start_time
        ).nanoseconds / 1e9 < self.timeout:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            dist_error, yaw_error = self.get_error()

            if dist_error is not None and yaw_error is not None:
                get_node().get_logger().info(
                    f"Alignment check: dist_error={dist_error:.2f}m, yaw_error={yaw_error:.2f}rad",
                    throttle_duration_sec=1.0,
                )
                if self.keep_orientation:
                    aligned = self.is_aligned_distance_only(dist_error)
                else:
                    aligned = self.is_aligned_distance_and_yaw(dist_error, yaw_error)

                if aligned:
                    if self.confirm_duration == 0.0:
                        get_node().get_logger().info(
                            "CheckAlignmentState: Alignment successful."
                        )
                        return "succeeded"
                    if first_success_time is None:
                        first_success_time = get_node().get_clock().now()
                    if (
                        get_node().get_clock().now() - first_success_time
                    ).nanoseconds / 1e9 >= self.confirm_duration:
                        get_node().get_logger().info(
                            f"CheckAlignmentState: Alignment successful for {self.confirm_duration} seconds."
                        )
                        return "succeeded"
                else:
                    first_success_time = None
            else:
                first_success_time = None

            self.rate.sleep()

        get_node().get_logger().warning("CheckAlignmentState: Timeout reached.")
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
        max_linear_velocity=None,
        max_angular_velocity=None,
        use_frame_depth=False,
        closest_yaw=False,
        closest_yaw_180=False,
    ):
        super().__init__(outcomes=["succeeded", "aborted", "preempted"])

        if closest_yaw and closest_yaw_180:
            raise ValueError("closest_yaw and closest_yaw_180 cannot both be enabled")

        with self:
            watch_succeeded_transition = (
                "CANCEL_ALIGNMENT_ON_SUCCESS" if cancel_on_success else "succeeded"
            )

            smach.StateMachine.add(
                "REQUEST_ALIGNMENT",
                SetAlignControllerTargetState(
                    source_frame=source_frame,
                    target_frame=target_frame,
                    angle_offset=angle_offset,
                    keep_orientation=keep_orientation,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    use_depth=use_frame_depth,
                    closest_yaw=closest_yaw,
                    closest_yaw_180=closest_yaw_180,
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
                    use_frame_depth=use_frame_depth,
                    closest_yaw=closest_yaw,
                    closest_yaw_180=closest_yaw_180,
                ),
                transitions={
                    "succeeded": watch_succeeded_transition,
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


class SetPlanState(smach.State):
    """State that calls the /set_plan service"""

    def __init__(
        self,
        target_frame: str,
        angle_offset: float = 0.0,
        dynamic: bool = True,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.target_frame = target_frame
        self.angle_offset = angle_offset
        self.dynamic = dynamic

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                get_node().get_logger().warning("[SetPlanState] Preempt requested")
                return "preempted"

            set_plan = service_client(get_node(), PlanPath, "/set_plan")
            set_plan.wait_for_service()

            req = PlanPath.Request(
                target_frame=self.target_frame,
                angle_offset=self.angle_offset,
                dynamic=self.dynamic,
            )
            response = set_plan(req)
            if not response.success:
                get_node().get_logger().error(
                    "[SetPlanState] /set_plan returned success=False for target_frame=%s"
                    % (self.target_frame,)
                )
                return "aborted"
            return "succeeded"

        except Exception as e:
            get_node().get_logger().error("[SetPlanState] Error: %s" % (str(e),))
            return "aborted"


class SetPlanningNotActive(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, get_node(), "/stop_planning", Trigger, request=Trigger.Request()
        )


class GravityZEnable(smach_ros.ServiceState):
    def __init__(self, enable):
        smach_ros.ServiceState.__init__(
            self,
            get_node(),
            "gravity_compensation_estimator/set_apply_updates",
            SetBool,
            request=SetBool.Request(data=enable),
        )


class DynamicPathState(smach.StateMachine):
    def __init__(
        self,
        plan_target_frame: str,
        align_source_frame: str = None,
        align_target_frame: str = "dynamic_target",
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        angle_offset: float = 0.0,
        keep_orientation: bool = False,
        dynamic: bool = True,
    ):
        if align_source_frame is None:
            align_source_frame = get_base_link()
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])
        with self:
            smach.StateMachine.add(
                "SET_PATH_PLAN",
                SetPlanState(
                    target_frame=plan_target_frame,
                    angle_offset=angle_offset,
                    dynamic=dynamic,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_PATH",
                SetAlignControllerTargetState(
                    source_frame=align_source_frame,
                    target_frame=align_target_frame,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    keep_orientation=keep_orientation,
                ),
                transitions={
                    "succeeded": "EXECUTE_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_PATH",
                ExecutePathState(),
                transitions={
                    "succeeded": "SET_PLANNING_NOT_ACTIVE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_PLANNING_NOT_ACTIVE",
                SetPlanningNotActive(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class LookAroundState(smach.StateMachine):
    def __init__(
        self,
        source_frame: str = None,
        angle_offset: float = 0.5,
        confirm_duration: float = 0.1,
        timeout: float = 10.0,
        max_linear_velocity: float = 0.1,
        max_angular_velocity: float = 0.15,
        current_pose_frame: str = "selam_frame",
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])
        if source_frame is None:
            source_frame = get_base_link()

        with self:
            smach.StateMachine.add(
                "CREATE_START_FRAME",
                CreateFrameAtCurrentPositionState(
                    current_pose_frame=current_pose_frame
                ),
                transitions={
                    "succeeded": "LOOK_LEFT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_LEFT",
                AlignFrame(
                    source_frame=source_frame,
                    target_frame=current_pose_frame,
                    angle_offset=angle_offset,
                    confirm_duration=confirm_duration,
                    timeout=timeout,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "LOOK_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_RIGHT",
                AlignFrame(
                    source_frame=source_frame,
                    target_frame=current_pose_frame,
                    angle_offset=-angle_offset,
                    confirm_duration=confirm_duration,
                    timeout=timeout,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "LOOK_STRAIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_STRAIGHT",
                AlignFrame(
                    source_frame=source_frame,
                    target_frame=current_pose_frame,
                    angle_offset=0.0,
                    confirm_duration=confirm_duration,
                    timeout=timeout,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class CreateFrameAtCurrentPositionState(smach.State):
    def __init__(
        self,
        source_frame: str = None,
        current_pose_frame: str = "selam_frame",
        reference_frame: str = "odom",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        if source_frame is None:
            source_frame = get_base_link()
        self.source_frame = source_frame
        self.current_pose_frame = current_pose_frame
        self.reference_frame = reference_frame
        self.tf_buffer = get_tf_buffer()
        self.set_object_transform_service = service_client(
            get_node(), SetObjectTransform, "set_object_transform"
        )

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"
        try:
            current_transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.source_frame,
                Time(),
                Duration(seconds=4.0),
            )
            new_transform = TransformStamped()
            new_transform.header.stamp = get_node().get_clock().now().to_msg()
            new_transform.header.frame_id = self.reference_frame
            new_transform.child_frame_id = self.current_pose_frame

            new_transform.transform.translation = (
                current_transform.transform.translation
            )
            new_transform.transform.rotation = current_transform.transform.rotation

            req = SetObjectTransform.Request()
            req.transform = new_transform
            res = self.set_object_transform_service(req)

            if not res.success:
                get_node().get_logger().error(
                    f"CreateFrameAtCurrentPositionState: Failed to create frame '{self.current_pose_frame}': {res.message}"
                )
                return "aborted"

            return "succeeded"

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            get_node().get_logger().error(
                f"CreateFrameAtCurrentPositionState: TF lookup failed: {e}"
            )
            return "aborted"

        except Exception as e:
            get_node().get_logger().error(
                f"CreateFrameAtCurrentPositionState: Service call failed: {e}"
            )
            return "aborted"


class CreateRotatingFrameState(smach.State):
    def __init__(
        self,
        target_frame: str,
        source_frame: str,
        reference_frame: str = "odom",
        rotation_period: float = 15.0,
        rotation_count: int = 1,
        rate_hz: int = 20,
        look_at_frame: str = None,
        full_rotation: bool = True,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        if rotation_count < 1:
            raise ValueError("rotation_count must be at least 1")
        self.target_frame = target_frame
        self.source_frame = source_frame
        self.reference_frame = reference_frame
        self.rotation_period = rotation_period
        self.rotation_count = rotation_count
        self.look_at_frame = look_at_frame
        self.full_rotation = full_rotation
        self.tf_buffer = get_tf_buffer()
        self.rate = Rate(get_node(), rate_hz)
        self.set_object_transform_service = service_client(
            get_node(), SetObjectTransform, "set_object_transform"
        )

    def is_transform_available(self):
        if not self.look_at_frame:
            return False
        try:
            lookup_fresh_transform(
                self.tf_buffer,
                get_node(),
                self.source_frame,
                self.look_at_frame,
                Duration(seconds=get_param(get_node(), "~tf_lookup_timeout", 0.2)),
                Duration(seconds=get_param(get_node(), "~tf_freshness_threshold", 0.4)),
            )
            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            return False

    def execute(self, userdata):
        get_node().get_logger().info(
            "Creating rotating frame '%s' for %d rotation(s) (%.2fs each)."
            % (self.target_frame, self.rotation_count, self.rotation_period)
        )
        try:
            initial_transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.source_frame,
                Time(),
                Duration(seconds=4.0),
            )
            initial_translation = initial_transform.transform.translation
            initial_rotation = initial_transform.transform.rotation
            initial_yaw = euler_from_quaternion(
                [
                    initial_rotation.x,
                    initial_rotation.y,
                    initial_rotation.z,
                    initial_rotation.w,
                ]
            )[2]
            get_node().get_logger().info(
                "Initial pose locked at: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f"
                % (
                    initial_translation.x,
                    initial_translation.y,
                    initial_translation.z,
                    initial_yaw,
                )
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            get_node().get_logger().error(
                "CreateRotatingFrameState: Failed to get initial robot pose: %s"
                % (str(e),)
            )
            return "aborted"

        start_time = get_node().get_clock().now()
        end_time = start_time + Duration(
            seconds=self.rotation_period * self.rotation_count
        )
        angular_velocity = 2.0 * math.pi / self.rotation_period

        if not self.full_rotation and self.is_transform_available():
            get_node().get_logger().info(
                "CreateRotatingFrameState: transform found before starting, returning"
            )
            return "succeeded"

        while rclpy.ok() and get_node().get_clock().now() < end_time:
            if self.preempt_requested():
                get_node().get_logger().warning(
                    "CreateRotatingFrameState: Preempted while creating frame '%s'."
                    % (self.target_frame,)
                )
                return "preempted"

            if not self.full_rotation and self.is_transform_available():
                get_node().get_logger().info(
                    "CreateRotatingFrameState: transform found during rotation, stopping."
                )
                return "succeeded"

            try:
                elapsed_time = (
                    get_node().get_clock().now() - start_time
                ).nanoseconds / 1e9
                current_yaw = initial_yaw + angular_velocity * elapsed_time
                qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, current_yaw)
                t = TransformStamped()
                t.header.stamp = get_node().get_clock().now().to_msg()
                t.header.frame_id = self.reference_frame
                t.child_frame_id = self.target_frame
                t.transform.translation.x = initial_translation.x
                t.transform.translation.y = initial_translation.y
                t.transform.translation.z = initial_translation.z
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw
                req = SetObjectTransform.Request()
                req.transform = t
                res = self.set_object_transform_service(req)
                if not res.success:
                    get_node().get_logger().warning(
                        "SetObjectTransform failed: %s" % (res.message,)
                    )
            except Exception as e:
                get_node().get_logger().warning("Service call failed: %s" % (str(e),))
                return "aborted"
            self.rate.sleep()

        if not rclpy.ok():
            return "aborted"
        get_node().get_logger().info(
            "Finished rotating frame '%s'." % (self.target_frame,)
        )
        return "succeeded"


class AlignAndCreateRotatingFrame(smach.StateMachine):
    def __init__(
        self,
        source_frame: str,
        rotating_frame_name: str,
        rotation_period: float = 15.0,
        rotation_count: int = 1,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        look_at_frame: str = None,
        full_rotation: bool = False,
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])
        with self:
            smach.StateMachine.add(
                "SET_ALIGN_TARGET",
                SetAlignControllerTargetState(
                    source_frame=source_frame,
                    target_frame=rotating_frame_name,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "CREATE_ROTATING_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CREATE_ROTATING_FRAME",
                CreateRotatingFrameState(
                    target_frame=rotating_frame_name,
                    source_frame=source_frame,
                    reference_frame="odom",
                    rotation_period=rotation_period,
                    rotation_count=rotation_count,
                    rate_hz=20,
                    look_at_frame=look_at_frame,
                    full_rotation=full_rotation,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class CheckForTransformState(smach.State):
    def __init__(
        self,
        source_frame: str,
        target_frame,
        timeout: float = 60.0,
        check_rate_hz: int = 10,
        allow_mutli_check_goal: bool = False,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.timeout = timeout
        self.tf_buffer = get_tf_buffer()
        self.rate = Rate(get_node(), check_rate_hz)
        self.allow_mutli_check_goal = allow_mutli_check_goal

    def is_transform_available(self):
        if self.allow_mutli_check_goal and isinstance(self.target_frame, list):
            frames = self.target_frame
        else:
            frames = [self.target_frame]

        for frame in frames:
            if not frame:
                continue
            try:
                lookup_fresh_transform(
                    self.tf_buffer,
                    get_node(),
                    self.source_frame,
                    frame,
                    Duration(seconds=get_param(get_node(), "~tf_lookup_timeout", 0.2)),
                    Duration(
                        seconds=get_param(get_node(), "~tf_freshness_threshold", 0.2)
                    ),
                )
                return True
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                pass
        return False

    def execute(self, userdata):
        start_time = get_node().get_clock().now()

        while rclpy.ok():
            if self.preempt_requested():
                get_node().get_logger().info("CheckForTransformState: Preempted")
                self.service_preempt()
                return "preempted"

            elapsed = (get_node().get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > self.timeout:
                get_node().get_logger().warning(
                    f"CheckForTransformState: Timeout ({self.timeout}s) reached while waiting for transform from {self.source_frame} to {self.target_frame}"
                )
                return "aborted"

            if self.is_transform_available():
                get_node().get_logger().info(
                    f"CheckForTransformState: Transform found from {self.source_frame} to {self.target_frame}"
                )
                return "succeeded"

            get_node().get_logger().debug(
                f"CheckForTransformState: Checking for transform from {self.source_frame} to {self.target_frame}...",
                throttle_duration_sec=1.0,
            )
            self.rate.sleep()

        return "aborted"


class DynamicPathWithTransformCheck(smach.Concurrence):

    def __init__(
        self,
        plan_target_frame: str,
        transform_source_frame: str,
        transform_target_frame,
        align_source_frame: str = "taluy/base_link",
        align_target_frame: str = "dynamic_target",
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        angle_offset: float = 0.0,
        keep_orientation: bool = False,
        transform_timeout: float = 60.0,
        allow_mutli_check_goal: bool = False,
    ):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            outcome_map={
                "succeeded": {"CHECK_FOR_TRANSFORM": "succeeded"},
                "preempted": {"DYNAMIC_PATH": "preempted"},
                "aborted": {"DYNAMIC_PATH": "aborted"},
            },
            child_termination_cb=lambda outcome_map: True,
        )

        with self:
            smach.Concurrence.add(
                "DYNAMIC_PATH",
                DynamicPathState(
                    plan_target_frame=plan_target_frame,
                    align_source_frame=align_source_frame,
                    align_target_frame=align_target_frame,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    angle_offset=angle_offset,
                    keep_orientation=keep_orientation,
                ),
            )

            smach.Concurrence.add(
                "CHECK_FOR_TRANSFORM",
                CheckForTransformState(
                    source_frame=transform_source_frame,
                    target_frame=transform_target_frame,
                    timeout=transform_timeout,
                    allow_mutli_check_goal=allow_mutli_check_goal,
                ),
            )


class AlignFrameWithTransformCheck(smach.Concurrence):

    def __init__(
        self,
        source_frame,
        target_frame,
        transform_source_frame: str,
        transform_target_frame,
        angle_offset=0.0,
        dist_threshold=0.1,
        yaw_threshold=0.1,
        timeout=30.0,
        cancel_on_success=False,
        confirm_duration=0.0,
        keep_orientation=False,
        max_linear_velocity=None,
        max_angular_velocity=None,
        use_frame_depth=False,
        closest_yaw=False,
        closest_yaw_180=False,
        transform_timeout: float = 60.0,
        allow_mutli_check_goal: bool = False,
    ):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            outcome_map={
                "succeeded": {"CHECK_FOR_TRANSFORM": "succeeded"},
                "preempted": {"ALIGN_FRAME": "preempted"},
                "aborted": {"ALIGN_FRAME": "aborted"},
            },
            child_termination_cb=lambda outcome_map: True,
        )

        with self:
            smach.Concurrence.add(
                "ALIGN_FRAME",
                AlignFrame(
                    source_frame=source_frame,
                    target_frame=target_frame,
                    angle_offset=angle_offset,
                    dist_threshold=dist_threshold,
                    yaw_threshold=yaw_threshold,
                    timeout=timeout,
                    cancel_on_success=cancel_on_success,
                    confirm_duration=confirm_duration,
                    keep_orientation=keep_orientation,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    use_frame_depth=use_frame_depth,
                    closest_yaw=closest_yaw,
                    closest_yaw_180=closest_yaw_180,
                ),
            )

            smach.Concurrence.add(
                "CHECK_FOR_TRANSFORM",
                CheckForTransformState(
                    source_frame=transform_source_frame,
                    target_frame=transform_target_frame,
                    timeout=transform_timeout,
                    allow_mutli_check_goal=allow_mutli_check_goal,
                ),
            )
