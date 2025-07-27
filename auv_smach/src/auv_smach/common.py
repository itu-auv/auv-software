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
    PlanPath,
    PlanPathRequest,
    SetDepth,
    SetDepthRequest,
    SetObjectTransform,
    SetObjectTransformRequest,
    SetObjectTransformResponse,
    SetDetectionFocus,
    SetDetectionFocusRequest,
)

from auv_msgs.srv import SetDepth, SetDepthRequest
from auv_msgs.srv import VisualServoing, VisualServoingRequest

from auv_navigation.follow_path_action import follow_path_client
from auv_msgs.srv import PlanPath, PlanPathRequest
from auv_navigation.path_planning.path_planners import PathPlanners

from auv_msgs.srv import PlanPath, PlanPathRequest
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


class VisualServoingCentering(smach_ros.ServiceState):
    def __init__(self, target_prop):
        request = VisualServoingRequest()
        request.target_prop = target_prop
        super(VisualServoingCentering, self).__init__(
            "visual_servoing/start",
            VisualServoing,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class VisualServoingNavigation(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingNavigation, self).__init__(
            "visual_servoing/navigate",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )


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

    def __init__(self, depth: float, sleep_duration: float = 5.0, frame_id: str = ""):
        set_depth_request = SetDepthRequest()
        set_depth_request.target_depth = depth
        set_depth_request.frame_id = frame_id
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


class SetHeadingControlState(smach_ros.ServiceState):
    def __init__(self, enable: bool):
        request = SetBoolRequest(data=enable)
        super(SetHeadingControlState, self).__init__(
            "set_heading_control",
            SetBool,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
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
    ):
        align_request = AlignFrameControllerRequest()
        align_request.source_frame = source_frame
        align_request.target_frame = target_frame
        align_request.angle_offset = angle_offset
        align_request.keep_orientation = keep_orientation
        if max_linear_velocity is not None:
            align_request.max_linear_velocity = max_linear_velocity
        if max_angular_velocity is not None:
            align_request.max_angular_velocity = max_angular_velocity

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


class RotationState(smach.State):
    def __init__(
        self,
        source_frame,
        look_at_frame,
        rotation_speed=0.2,
        full_rotation=False,
        full_rotation_timeout=25.0,
        rate_hz=10,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.odom_topic = "odometry"
        self.cmd_vel_topic = "cmd_vel"
        self.rotation_speed = rotation_speed
        self.odom_data = False
        self.yaw = None
        self.yaw_prev = None
        self.total_yaw = 0.0
        self.rate = rospy.Rate(rate_hz)
        self.active = True

        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.full_rotation = full_rotation
        self.full_rotation_timeout = full_rotation_timeout

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        self.enable_pub = rospy.Publisher(
            "enable",
            Bool,
            queue_size=1,
        )

        self.killswitch_sub = rospy.Subscriber(
            "propulsion_board/status",
            Bool,
            self.killswitch_callback,
        )

    def killswitch_callback(self, msg):
        if not msg.data:
            self.active = False
            rospy.logwarn("RotationState: Killswitch activated, stopping rotation")

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
            return self.tf_buffer.can_transform(
                self.source_frame,
                self.look_at_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logdebug(f"RotationState: Transform check failed: {e}")
            return False

    def execute(self, userdata):
        while not rospy.is_shutdown() and not self.odom_data:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            self.rate.sleep()

        self.yaw_prev = self.yaw
        self.total_yaw = 0.0
        twist = Twist()
        twist.angular.z = self.rotation_speed
        self.active = True
        rotation_start_time = rospy.Time.now()

        transform_found = self.is_transform_available()
        if transform_found and not self.full_rotation:
            rospy.loginfo(
                "RotationState: transform already available, no need to rotate"
            )
            return "succeeded"

        while not rospy.is_shutdown() and self.total_yaw < 2 * math.pi and self.active:
            if self.preempt_requested():
                twist.angular.z = 0.0
                self.pub.publish(twist)
                self.service_preempt()
                return "preempted"

            if (
                self.full_rotation
                and (rospy.Time.now() - rotation_start_time).to_sec()
                > self.full_rotation_timeout
            ):
                rospy.logwarn(
                    f"RotationState: Timeout reached after {self.full_rotation_timeout} seconds during full rotation."
                )
                twist.angular.z = 0.0
                self.pub.publish(twist)
                break

            self.enable_pub.publish(Bool(data=True))

            if not self.full_rotation and self.is_transform_available():
                twist.angular.z = 0.0
                self.pub.publish(twist)
                rospy.loginfo(
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

        if not self.active:
            rospy.loginfo("RotationState: rotation aborted by killswitch.")
            return "aborted"

        rospy.loginfo(
            f"RotationState: completed full rotation. Total yaw: {self.total_yaw}"
        )

        if self.is_transform_available():
            return "succeeded"
        else:
            rospy.logwarn(
                "RotationState: completed full rotation but no transform found between %s and %s",
                self.source_frame,
                self.look_at_frame,
            )
            return "aborted"


class SetFrameLookingAtState(smach.State):
    def __init__(self, source_frame, look_at_frame, alignment_frame, duration_time=3.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.alignment_frame = alignment_frame
        self.duration_time = duration_time
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.rate = rospy.Rate(10)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

    def execute(self, userdata):
        start_time = rospy.Time.now()
        end_time = start_time + rospy.Duration(self.duration_time)

        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            try:

                base_to_look_at_transform = self.tf_buffer.lookup_transform(
                    self.source_frame,
                    self.look_at_frame,
                    rospy.Time(0),
                    rospy.Duration(1.0),
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
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.source_frame
                t.child_frame_id = self.alignment_frame
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]

                req = SetObjectTransformRequest()
                req.transform = t
                res = self.set_object_transform_service(req)

                if not res.success:
                    rospy.logwarn(f"SetObjectTransform failed: {res.message}")

                time_remaining = (end_time - rospy.Time.now()).to_sec()
                rospy.loginfo_throttle(
                    1.0,
                    f"Looking at {self.look_at_frame}. Time remaining: {time_remaining:.2f}s",
                )

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn(f"TF lookup exception: {e}")

            except rospy.ServiceException as e:
                rospy.logwarn(f"Service call failed: {e}")
                return "aborted"

            self.rate.sleep()

        rospy.loginfo(
            f"Successfully looked at {self.look_at_frame} for {self.duration_time} seconds"
        )
        return "succeeded"


class ClearObjectMapState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "clear_object_transforms",
            Trigger,
            request=TriggerRequest(),
        )


class SetDetectionState(smach_ros.ServiceState):
    """
    Calls the service to enable or disable camera detections.
    """

    def __init__(self, camera_name: str, enable: bool):
        if camera_name not in ["front", "bottom"]:
            raise ValueError("camera_name must be 'front' or 'bottom'")

        service_name = f"enable_{camera_name}_camera_detections"
        request = SetBoolRequest(data=enable)

        super(SetDetectionState, self).__init__(
            service_name,
            SetBool,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class SetDetectionFocusState(smach_ros.ServiceState):
    """
    Calls the service to set the focus for the front camera detections.
    """

    def __init__(self, focus_object: str):
        service_name = "set_front_camera_focus"
        request = SetDetectionFocusRequest(focus_object=focus_object)

        super(SetDetectionFocusState, self).__init__(
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
            rospy.logdebug("[ExecutePathState] Initializing the FollowPathActionClient")
            self._client = follow_path_client.FollowPathActionClient()

        # Check for preemption before proceeding
        if self.preempt_requested():
            rospy.logwarn("[ExecutePathState] Preempt requested")
            return "preempted"
        try:
            # We send an empty goal, as the action server now listens to a topic
            success = self._client.execute_paths([])
            if success:
                rospy.logdebug("[ExecutePathState] Planned paths executed successfully")
                return "succeeded"
            else:
                rospy.logwarn("[ExecutePathState] Execution of planned paths failed")
                return "aborted"

        except Exception as e:
            rospy.logerr("[ExecutePathState] Exception occurred: %s", str(e))
            return "aborted"


class SearchForPropState(smach.StateMachine):
    """
    1. RotationState: Rotates to find a prop's frame.
    2. SetAlignControllerTargetState: Sets the align controller target.
    3. SetFrameLookingAtState: Sets a target frame's pose based on looking at another frame.
    4. CancelAlignControllerState: Cancels the align controller target.
    """

    def __init__(
        self,
        look_at_frame: str,
        alignment_frame: str,
        full_rotation: bool,
        set_frame_duration: float,
        source_frame: str = "taluy/base_link",
        rotation_speed: float = 0.3,
        max_angular_velocity: float = 0.25,
    ):
        """
        Args:
            look_at_frame (str): The frame to rotate towards and look at.
            alignment_frame (str): The frame to set as the align controller target
                                and whose pose is set by SetFrameLookingAtState.
            full_rotation (bool): Whether to perform a full 360-degree rotation
                                  or stop when look_at_frame is found.
            set_frame_duration (float): Duration for the SetFrameLookingAtState.
            source_frame (str): The base frame of the vehicle (default: "taluy/base_link").
            rotation_speed (float): The angular velocity for rotation (default: 0.3).
            max_angular_velocity (float): Max angular velocity for align controller (optional).
        """
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        with self:
            smach.StateMachine.add(
                "ROTATE_TO_FIND_PROP",
                RotationState(
                    source_frame=source_frame,
                    look_at_frame=look_at_frame,
                    rotation_speed=rotation_speed,
                    full_rotation=full_rotation,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame=source_frame,
                    target_frame=alignment_frame,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "BROADCAST_ALIGNMENT_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "BROADCAST_ALIGNMENT_FRAME",
                SetFrameLookingAtState(
                    source_frame=source_frame,
                    look_at_frame=look_at_frame,
                    alignment_frame=alignment_frame,
                    duration_time=set_frame_duration,
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


class PlanPathToSingleFrameState(smach.State):
    def __init__(
        self, tf_buffer, target_frame: str, source_frame: str = "taluy/base_link"
    ):
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
                rospy.logwarn(
                    f"[PlanPathToSingleFrameState] Preempt requested for path to {self.target_frame}"
                )
                return "preempted"

            path_planners = PathPlanners(self.tf_buffer)
            path = path_planners.straight_path_to_frame(
                source_frame=self.source_frame,
                target_frame=self.target_frame,
                num_waypoints=50,
            )

            if path is None:
                rospy.logerr(
                    f"[PlanPathToSingleFrameState] Failed to plan path to {self.target_frame}"
                )
                return "aborted"

            userdata.planned_paths = [path]  #
            rospy.loginfo(
                f"[PlanPathToSingleFrameState] Successfully planned path to {self.target_frame}"
            )
            return "succeeded"

        except Exception as e:
            rospy.logerr(
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
        max_linear_velocity=None,
        max_angular_velocity=None,
        heading_control=True,
    ):
        super().__init__(outcomes=["succeeded", "aborted", "preempted"])

        with self:
            watch_succeeded_transition = (
                "CANCEL_ALIGNMENT_ON_SUCCESS" if cancel_on_success else "succeeded"
            )
            cancel_on_success_succeeded_transition = "succeeded"
            cancel_on_fail_succeeded_transition = "aborted"
            cancel_on_preempt_succeeded_transition = "preempted"

            if not heading_control:  # If heading control will not be used
                watch_succeeded_transition = (
                    "CANCEL_ALIGNMENT_ON_SUCCESS"
                    if cancel_on_success
                    else "ENABLE_HEADING_CONTROL_ON_SUCCESS"
                )
                cancel_on_success_succeeded_transition = (
                    "ENABLE_HEADING_CONTROL_ON_SUCCESS"
                )
                cancel_on_fail_succeeded_transition = "ENABLE_HEADING_CONTROL_ON_FAIL"
                cancel_on_preempt_succeeded_transition = (
                    "ENABLE_HEADING_CONTROL_ON_PREEMPT"
                )

                smach.StateMachine.add(
                    "DISABLE_HEADING_CONTROL",
                    SetHeadingControlState(enable=False),
                    transitions={
                        "succeeded": "REQUEST_ALIGNMENT",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
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
                    "succeeded": watch_succeeded_transition,
                    "aborted": "CANCEL_ALIGNMENT_ON_FAIL",
                    "preempted": "CANCEL_ALIGNMENT_ON_PREEMPT",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGNMENT_ON_SUCCESS",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": cancel_on_success_succeeded_transition,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGNMENT_ON_FAIL",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": cancel_on_fail_succeeded_transition,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGNMENT_ON_PREEMPT",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": cancel_on_preempt_succeeded_transition,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            if not heading_control:
                smach.StateMachine.add(
                    "ENABLE_HEADING_CONTROL_ON_SUCCESS",
                    SetHeadingControlState(enable=True),
                    transitions={
                        "succeeded": "succeeded",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    "ENABLE_HEADING_CONTROL_ON_FAIL",
                    SetHeadingControlState(enable=True),
                    transitions={
                        "succeeded": "aborted",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    "ENABLE_HEADING_CONTROL_ON_PREEMPT",
                    SetHeadingControlState(enable=True),
                    transitions={
                        "succeeded": "preempted",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )


class SetPlanState(smach.State):
    """State that calls the /set_plan service"""

    def __init__(self, target_frame: str, angle_offset: float = 0.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.target_frame = target_frame
        self.angle_offset = angle_offset

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[SetPlanState] Preempt requested")
                return "preempted"

            rospy.wait_for_service("/set_plan")
            set_plan = rospy.ServiceProxy("/set_plan", PlanPath)

            req = PlanPathRequest(
                target_frame=self.target_frame,
                angle_offset=self.angle_offset,
            )
            set_plan(req)
            return "succeeded"

        except Exception as e:
            rospy.logerr("[SetPlanState] Error: %s", str(e))
            return "aborted"


class SetPlanningNotActive(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "/stop_planning", Trigger, request=TriggerRequest()
        )


class DynamicPathState(smach.StateMachine):
    def __init__(
        self,
        plan_target_frame: str,
        align_source_frame: str = "taluy/base_link",
        align_target_frame: str = "dynamic_target",
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        angle_offset: float = 0.0,
        keep_orientation: bool = False,
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])
        with self:
            smach.StateMachine.add(
                "SET_PATH_PLAN",
                SetPlanState(target_frame=plan_target_frame, angle_offset=angle_offset),
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
