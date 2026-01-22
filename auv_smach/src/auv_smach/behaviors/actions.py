import py_trees
import rospy
import threading
import math
import tf2_ros
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped
import numpy as np
from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetDetectionFocus,
    SetDetectionFocusRequest,
    SetObjectTransform,
    SetObjectTransformRequest,
    AlignFrameController,
    AlignFrameControllerRequest,
    PlanPath,
    PlanPathRequest,
)
from auv_navigation.follow_path_action import follow_path_client


class SetDepthBehavior(py_trees.behaviour.Behaviour):
    """
    Sets the vehicle depth and publishes enable signal for a specified duration.

    Calls /taluy/set_depth with the requested depth.
    Continuously publishes True to /taluy/enable topic whilst running.

    Returns:
        SUCCESS: Operation completed
        FAILURE: Service error
        RUNNING: Still waiting
    """

    def __init__(
        self,
        name: str,
        depth: float,
        sleep_duration: float = 5.0,
        frame_id: str = "",
    ):
        # Call parent class __init__
        super().__init__(name)

        # Store parameters
        self.depth = depth
        self.sleep_duration = sleep_duration
        self.frame_id = frame_id

        # State tracking variables
        self._service_called = False
        self._start_time = None
        self._stop_publishing = threading.Event()
        self._pub_thread = None

    def setup(self, timeout=15, **kwargs):
        """
        Setup ROS connections.
        Called once when the tree is first initialized.
        """
        try:
            # Create service proxy
            self.set_depth_srv = rospy.ServiceProxy("set_depth", SetDepth)
            # Create publisher
            self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        """
        Called when the behavior starts running.
        Reset state variables.
        """
        self._service_called = False
        self._start_time = None
        self._stop_publishing.clear()

    def _publish_enable_loop(self):
        """Publish enable signal in background thread"""
        publish_rate = rospy.get_param("~enable_rate", 20)
        rate = rospy.Rate(publish_rate)
        while not rospy.is_shutdown() and not self._stop_publishing.is_set():
            self.enable_pub.publish(Bool(True))
            rate.sleep()

    def update(self):
        """
        Called on every tick. This method must be NON-BLOCKING!
        """
        # First tick: call the service
        if not self._service_called:
            try:
                # Create service request
                req = SetDepthRequest()
                req.target_depth = self.depth
                req.frame_id = self.frame_id

                # Call the service
                self.set_depth_srv(req)
                self._service_called = True
                self._start_time = rospy.Time.now()

                # Start publishing enable signal (background thread)
                self._stop_publishing.clear()
                self._pub_thread = threading.Thread(target=self._publish_enable_loop)
                self._pub_thread.start()

                rospy.loginfo(f"[{self.name}] Depth set to {self.depth}m")

            except rospy.ServiceException as e:
                rospy.logerr(f"[{self.name}] Service error: {e}")
                return py_trees.common.Status.FAILURE

        # Check if wait duration has elapsed
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed >= self.sleep_duration:
            # Time's up, stop publishing
            self._stop_publishing.set()
            if self._pub_thread:
                # Ensure we don't block too long on join
                # If sleep_duration was 0, thread might have just started
                self._pub_thread.join(timeout=1.0)

            rospy.loginfo(f"[{self.name}] Complete (waited {self.sleep_duration}s)")
            return py_trees.common.Status.SUCCESS

        # Still waiting
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Called when behavior terminates (SUCCESS, FAILURE, or preempted).
        Cleanup resources.
        """
        self._stop_publishing.set()
        if self._pub_thread and self._pub_thread.is_alive():
            self._pub_thread.join(timeout=1.0)


class CancelAlignControllerBehavior(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "CancelAlignController"):
        super().__init__(name)

    def setup(self, timeout=15, **kwargs):
        """Setup ROS service connection."""
        try:
            self.cancel_srv = rospy.ServiceProxy("align_frame/cancel", Trigger)
            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def update(self):
        """Call the cancel service."""
        try:
            response = self.cancel_srv(TriggerRequest())
            if response.success:
                rospy.loginfo(f"[{self.name}] Align controller cancelled")
                return py_trees.common.Status.SUCCESS
            else:
                rospy.logwarn(f"[{self.name}] Cancel failed: {response.message}")
                return py_trees.common.Status.FAILURE
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class SetDetectionFocusBehavior(py_trees.behaviour.Behaviour):
    """
    Sets the focus object for the front camera detection system.
    """

    def __init__(self, name: str, focus_object: str):
        super().__init__(name)
        self.focus_object = focus_object

    def setup(self, timeout=15, **kwargs):
        """Setup ROS service connection."""
        try:
            self.focus_srv = rospy.ServiceProxy(
                "set_front_camera_focus", SetDetectionFocus
            )
            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def update(self):
        """Call the set focus service."""
        try:
            request = SetDetectionFocusRequest(focus_object=self.focus_object)
            self.focus_srv(request)
            rospy.loginfo(f"[{self.name}] Detection focus set to: {self.focus_object}")
            return py_trees.common.Status.SUCCESS
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class SetBoolServiceBehavior(py_trees.behaviour.Behaviour):
    """
    Calls a SetBool service to enable/disable a feature.
    """

    def __init__(self, name: str, service_name: str, value: bool):
        super().__init__(name)
        self.service_name = service_name
        self.value = value

    def setup(self, timeout=15, **kwargs):
        """Setup ROS service connection."""
        try:
            self.service = rospy.ServiceProxy(self.service_name, SetBool)
            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def update(self):
        """Call the SetBool service."""
        try:
            response = self.service(SetBoolRequest(data=self.value))
            if response.success:
                rospy.loginfo(f"[{self.name}] {self.service_name} set to {self.value}")
                return py_trees.common.Status.SUCCESS
            else:
                rospy.logwarn(f"[{self.name}] Service failed: {response.message}")
                return py_trees.common.Status.FAILURE
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class RotateBehavior(py_trees.behaviour.Behaviour):
    """
    Rotates the vehicle until the target TF frame is found.
    Publishes cmd_vel for rotation and enable signal.
    """

    def __init__(
        self,
        name: str,
        source_frame: str,
        look_at_frame: str,
        rotation_speed: float = 0.2,
        rotation_radian: float = 2 * math.pi,
        timeout: float = 25.0,
        full_rotation: bool = False,
    ):
        super().__init__(name)
        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.rotation_speed = rotation_speed
        self.rotation_radian = rotation_radian
        self.timeout = timeout
        self.full_rotation = full_rotation

        # State tracking
        self._started = False
        self._start_time = None
        self._yaw = None
        self._yaw_prev = None
        self._total_yaw = 0.0
        self._odom_received = False
        self._killswitch_active = False

    def setup(self, timeout=15, **kwargs):
        """Setup ROS connections."""
        try:
            # TF buffer for checking transform availability
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            # Publishers
            self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
            self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

            # Subscriber for odometry (to track rotation)
            self.odom_sub = rospy.Subscriber("odometry", Odometry, self._odom_callback)

            # Subscriber for killswitch (propulsion_board/status) - Matches SMACH implementation
            # If msg.data is False, it means motors are OFF (Killswitch Active)
            self.killswitch_sub = rospy.Subscriber(
                "propulsion_board/status",
                Bool,
                self._killswitch_callback,
            )

            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def _killswitch_callback(self, msg):
        """Callback for killswitch status. False means KILLSWITCH ACTIVE (Motors OFF)."""
        if not msg.data:
            self._killswitch_active = True
            rospy.logwarn(f"[{self.name}] Killswitch activated! Stopping rotation.")

    def _odom_callback(self, msg):
        """Odometry callback to track yaw angle."""
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._yaw = yaw
        self._odom_received = True

    def _is_transform_available(self):
        """Check if target TF frame is available."""
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
        ):
            return False

    def initialise(self):
        """Called when behavior starts running. Reset state."""
        self._started = False
        self._start_time = None
        self._yaw_prev = None
        self._total_yaw = 0.0
        self._killswitch_active = False

    def update(self):
        """Called on every tick. Rotate until TF found or timeout."""

        # Wait for odometry data
        if not self._odom_received:
            return py_trees.common.Status.RUNNING

        # Check Killswitch status
        if self._killswitch_active:
            self._stop_rotation()
            rospy.logerr(f"[{self.name}] Aborted: Killswitch is active")
            return py_trees.common.Status.FAILURE

        # First tick: initialize
        if not self._started:
            # Check if transform already available (Only skip if NOT doing full rotation)
            if self._is_transform_available() and not self.full_rotation:
                rospy.loginfo(
                    f"[{self.name}] Transform already available, no rotation needed"
                )
                return py_trees.common.Status.SUCCESS

            self._started = True
            self._start_time = rospy.Time.now()
            self._yaw_prev = self._yaw
            rospy.loginfo(
                f"[{self.name}] Starting rotation to find {self.look_at_frame} (Full Rotation: {self.full_rotation})"
            )

        # Check timeout
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed > self.timeout:
            self._stop_rotation()
            rospy.logwarn(f"[{self.name}] Timeout reached, transform not found")
            return py_trees.common.Status.FAILURE

        # Check if transform found (Skip early return if doing full rotation)
        if self._is_transform_available() and not self.full_rotation:
            self._stop_rotation()
            rospy.loginfo(f"[{self.name}] Transform found: {self.look_at_frame}")
            return py_trees.common.Status.SUCCESS

        # Check if full rotation completed
        if self._yaw is not None and self._yaw_prev is not None:
            dyaw = math.atan2(
                math.sin(self._yaw - self._yaw_prev),
                math.cos(self._yaw - self._yaw_prev),
            )
            self._total_yaw += abs(dyaw)
            self._yaw_prev = self._yaw

            if self._total_yaw >= self.rotation_radian:
                self._stop_rotation()

                # After rotation, check if we found the transform
                if self._is_transform_available():
                    rospy.loginfo(
                        f"[{self.name}] Full rotation completed, transform found."
                    )
                    return py_trees.common.Status.SUCCESS
                else:
                    rospy.logwarn(
                        f"[{self.name}] Full rotation completed, transform NOT found."
                    )
                    return py_trees.common.Status.FAILURE

        # Continue rotating
        twist = Twist()
        twist.angular.z = self.rotation_speed
        self.cmd_vel_pub.publish(twist)
        self.enable_pub.publish(Bool(data=True))

        return py_trees.common.Status.RUNNING

    def _stop_rotation(self):
        """Stop the rotation by publishing zero velocity."""
        twist = Twist()
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def terminate(self, new_status):
        """Called when behavior terminates. Stop rotation."""
        self._stop_rotation()


class SetFrameLookingAtBehavior(py_trees.behaviour.Behaviour):
    """
    Sets the alignment frame to look at a target frame.
    Calculates the angle to face the target and calls set_object_transform service.
    SMACH equivalent: SetFrameLookingAtState (common.py:522-605)
    """

    def __init__(
        self,
        name: str,
        source_frame: str,
        look_at_frame: str,
        alignment_frame: str,
        duration: float = 3.0,
    ):
        super().__init__(name)
        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.alignment_frame = alignment_frame
        self.duration = duration

        # State tracking
        self._started = False
        self._start_time = None

    def setup(self, timeout=15, **kwargs):
        """Setup ROS connections."""
        try:
            # TF buffer for looking up transforms
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            # Service for setting the alignment target
            self.set_transform_srv = rospy.ServiceProxy(
                "set_object_transform", SetObjectTransform
            )

            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        """Called when behavior starts running. Reset state."""
        self._started = False
        self._start_time = None

    def update(self):
        """Called on every tick. Calculate facing angle and set alignment target."""

        # First tick: start timer
        if not self._started:
            self._started = True
            self._start_time = rospy.Time.now()
            rospy.loginfo(f"[{self.name}] Starting to look at {self.look_at_frame}")

        # Check if duration elapsed
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed >= self.duration:
            rospy.loginfo(f"[{self.name}] Completed looking at {self.look_at_frame}")
            return py_trees.common.Status.SUCCESS

        # Try to set the facing direction
        try:
            # Get transform from source to target
            transform = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.look_at_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            # Calculate facing angle
            direction = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                ]
            )
            facing_angle = np.arctan2(direction[1], direction[0])
            quaternion = quaternion_from_euler(0, 0, facing_angle)

            # Create alignment transform
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

            # Call service
            req = SetObjectTransformRequest()
            req.transform = t
            res = self.set_transform_srv(req)

            if not res.success:
                rospy.logwarn(f"[{self.name}] Service failed: {res.message}")

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"[{self.name}] TF error: {e}")

        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING


class AlignFrameBehavior(py_trees.behaviour.Behaviour):
    """
    Aligns the vehicle to a target frame by calling the align_frame/start service.
    Monitors alignment progress via TF and returns SUCCESS when aligned or FAILURE on timeout.

    SMACH equivalent: AlignFrame (common.py:934-1050)
    """

    def __init__(
        self,
        name: str,
        source_frame: str,
        target_frame: str,
        angle_offset: float = 0.0,
        dist_threshold: float = 0.1,
        yaw_threshold: float = 0.1,
        keep_orientation: bool = False,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        timeout: float = 30.0,
        confirm_duration: float = 0.0,
        cancel_on_success: bool = True,
        heading_control: bool = True,
        enable_heading_control_afterwards: bool = True,
        wait_for_alignment: bool = True,
    ):
        super().__init__(name)
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.angle_offset = angle_offset
        self.dist_threshold = dist_threshold
        self.yaw_threshold = yaw_threshold
        self.keep_orientation = keep_orientation
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.timeout = timeout
        self.confirm_duration = confirm_duration
        self.cancel_on_success = cancel_on_success
        self.heading_control = heading_control
        self.enable_heading_control_afterwards = enable_heading_control_afterwards
        self.wait_for_alignment = wait_for_alignment

        # State tracking
        self._started = False
        self._start_time = None
        self._alignment_requested = False
        self._first_success_time = None
        self._heading_disabled = False

    def setup(self, timeout=15, **kwargs):
        """Setup ROS connections."""
        try:
            # Service to start alignment
            self.align_srv = rospy.ServiceProxy(
                "align_frame/start", AlignFrameController
            )

            # Service to cancel alignment (matches controller's "cancel_control" service)
            # SMACH uses "align_frame/cancel", so we should too for parity.
            self.cancel_srv = rospy.ServiceProxy("align_frame/cancel", Trigger)

            # Service to enable/disable heading control
            self.heading_srv = rospy.ServiceProxy("set_heading_control", SetBool)

            # TF buffer for checking alignment
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        """Called when behavior starts running. Reset state."""
        self._started = False
        self._start_time = None
        self._alignment_requested = False
        self._first_success_time = None
        self._heading_disabled = False

    def _get_alignment_error(self):
        """Get distance and yaw error from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation

            # Distance error (XY plane)
            dist_error = math.sqrt(trans.x**2 + trans.y**2)

            # Yaw error
            _, _, yaw = euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
            # Normalize angle with offset
            yaw_with_offset = yaw + self.angle_offset
            # Normalize to [-pi, pi]
            while yaw_with_offset > math.pi:
                yaw_with_offset -= 2 * math.pi
            while yaw_with_offset < -math.pi:
                yaw_with_offset += 2 * math.pi
            yaw_error = abs(yaw_with_offset)

            return dist_error, yaw_error
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(2.0, f"[{self.name}] TF error: {e}")
            return None, None

    def _is_aligned(self, dist_error, yaw_error):
        """Check if aligned based on thresholds."""
        if self.keep_orientation:
            return dist_error < self.dist_threshold
        else:
            return dist_error < self.dist_threshold and yaw_error < self.yaw_threshold

    def _set_heading_control(self, enable: bool):
        """Enable or disable heading control."""
        try:
            self.heading_srv(SetBoolRequest(data=enable))
            rospy.loginfo(f"[{self.name}] Heading control set to {enable}")
        except rospy.ServiceException as e:
            rospy.logwarn(f"[{self.name}] Failed to set heading control: {e}")

    def update(self):
        """Called on every tick. Start alignment and monitor progress."""

        # First tick: initialization logic
        if not self._started:
            self._started = True
            self._start_time = rospy.Time.now()

            # Disable heading control if requested
            if not self.heading_control:
                self._set_heading_control(False)
                self._heading_disabled = True

            try:
                req = AlignFrameControllerRequest()
                req.source_frame = self.source_frame
                req.target_frame = self.target_frame
                req.angle_offset = self.angle_offset
                req.keep_orientation = self.keep_orientation
                if self.max_linear_velocity is not None:
                    req.max_linear_velocity = self.max_linear_velocity
                if self.max_angular_velocity is not None:
                    req.max_angular_velocity = self.max_angular_velocity

                res = self.align_srv(req)
                if res.success:
                    self._alignment_requested = True
                    self._alignment_requested = True
                    rospy.loginfo(
                        f"[{self.name}] Alignment started: {self.target_frame}"
                    )

                    if not self.wait_for_alignment:
                        return py_trees.common.Status.SUCCESS
                else:
                    rospy.logerr(f"[{self.name}] Alignment failed: {res.message}")
                    return py_trees.common.Status.FAILURE

            except rospy.ServiceException as e:
                rospy.logerr(f"[{self.name}] Service error: {e}")
                return py_trees.common.Status.FAILURE

        # Check timeout
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed > self.timeout:
            rospy.logwarn(f"[{self.name}] Timeout reached")
            self._cancel_alignment()
            return py_trees.common.Status.FAILURE

        # Check alignment via TF
        dist_error, yaw_error = self._get_alignment_error()

        if dist_error is not None and yaw_error is not None:
            rospy.loginfo_throttle(
                1.0, f"[{self.name}] dist={dist_error:.2f}m, yaw={yaw_error:.2f}rad"
            )

            if self._is_aligned(dist_error, yaw_error):
                # Check confirm duration
                if self.confirm_duration == 0.0:
                    rospy.loginfo(
                        f"[{self.name}] Alignment completed: {self.target_frame}"
                    )
                    if self.cancel_on_success:
                        self._cancel_alignment()
                    return py_trees.common.Status.SUCCESS

                # Need to stay aligned for confirm_duration
                if self._first_success_time is None:
                    self._first_success_time = rospy.Time.now()

                confirm_elapsed = (rospy.Time.now() - self._first_success_time).to_sec()
                if confirm_elapsed >= self.confirm_duration:
                    rospy.loginfo(
                        f"[{self.name}] Alignment confirmed: {self.target_frame}"
                    )
                    if self.cancel_on_success:
                        self._cancel_alignment()
                    return py_trees.common.Status.SUCCESS
            else:
                # Not aligned, reset confirm timer
                self._first_success_time = None

        return py_trees.common.Status.RUNNING

    def _cancel_alignment(self):
        """Cancel the alignment controller."""
        try:
            self.cancel_srv(TriggerRequest())
        except rospy.ServiceException as e:
            rospy.logwarn(f"[{self.name}] Cancel error: {e}")

    def terminate(self, new_status):
        """Called when behavior terminates. Cancel alignment if needed and restore heading control."""
        if self._alignment_requested and new_status != py_trees.common.Status.SUCCESS:
            self._cancel_alignment()

        # Restore heading control if it was disabled and we should enable it afterwards
        if self._heading_disabled and self.enable_heading_control_afterwards:
            self._set_heading_control(True)


class RotateBehavior(py_trees.behaviour.Behaviour):
    """
    Rotates the vehicle until a TF frame is found or for a specific duration.
    Can perform a 'blind' rotation (Twist) if needed.
    SMACH equivalent: RotationState (common.py:408)
    """

    def __init__(
        self,
        name: str,
        source_frame: str,
        look_at_frame: str,
        rotation_speed: float = 0.2,
        timeout: float = 25.0,
        rotation_radian: float = None,  # Total angle to rotate (if None, rotate until TF found)
        velocity_topic: str = "cmd_vel",
    ):
        super().__init__(name)
        self.source_frame = source_frame
        self.look_at_frame = look_at_frame
        self.rotation_speed = rotation_speed
        self.timeout = timeout
        self.rotation_radian = rotation_radian
        self.velocity_topic = velocity_topic

        # Runtime
        self._start_time = None
        self._tf_buffer = None
        self._tf_listener = None
        self.pub_vel = None

    def setup(self, timeout=15, **kwargs):
        try:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
            self.pub_vel = rospy.Publisher(self.velocity_topic, Twist, queue_size=1)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        self._start_time = rospy.Time.now()
        rospy.loginfo(f"[{self.name}] Starting rotation...")

    def update(self):
        # 0. Check Timeout
        elapsed = (rospy.Time.now() - self._start_time).to_sec()
        if elapsed > self.timeout:
            rospy.logwarn(f"[{self.name}] Timeout ({elapsed:.1f}s)")
            self._stop()
            return py_trees.common.Status.FAILURE

        # 1. If we just want to find a TF frame (classic usage)
        if self.rotation_radian is None:
            if self._is_transform_available():
                rospy.loginfo(f"[{self.name}] Target frame found via TF")
                self._stop()
                return py_trees.common.Status.SUCCESS

            # Spin
            self._spin()
            return py_trees.common.Status.RUNNING

        # 2. If we want to rotate for a specific angle (e.g. 360 spin)
        # Simply rotate for calculated time (open loop approximation as per SMACH)
        # SMACH calculates time = radian / speed
        duration_needed = abs(self.rotation_radian / self.rotation_speed)

        if elapsed >= duration_needed:
            rospy.loginfo(f"[{self.name}] Rotation complete (Time-based)")
            self._stop()
            return py_trees.common.Status.SUCCESS

        self._spin()
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if new_status != py_trees.common.Status.RUNNING:
            self._stop()

    def _spin(self):
        cmd = Twist()
        cmd.angular.z = self.rotation_speed
        self.pub_vel.publish(cmd)

    def _stop(self):
        cmd = Twist()
        cmd.angular.z = 0.0
        if self.pub_vel:
            self.pub_vel.publish(cmd)

    def _is_transform_available(self):
        try:
            self._tf_buffer.lookup_transform(
                self.source_frame,
                self.look_at_frame,
                rospy.Time(0),
                rospy.Duration(0.0),  # Non-blocking check
            )
            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return False


class CreateFrameBehavior(py_trees.behaviour.Behaviour):
    """
    Calls the /create_frame service to create a new TF frame.
    SMACH equivalent: CreateFrameState (common.py:1000)
    """

    def __init__(
        self,
        name: str,
        parent_frame: str,
        child_frame: str,
        x: float,
        y: float,
        yaw: float,
    ):
        super().__init__(name)
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.x = x
        self.y = y
        self.yaw = yaw
        self._done = False

    def setup(self, timeout=15, **kwargs):
        try:
            self.create_frame_srv = rospy.ServiceProxy("/create_frame", CreateFrame)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        self._done = False

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        try:
            req = CreateFrameRequest(
                parent_frame=self.parent_frame,
                child_frame=self.child_frame,
                x=self.x,
                y=self.y,
                yaw=self.yaw,
            )
            res = self.create_frame_srv(req)
            if res.success:
                rospy.loginfo(
                    f"[{self.name}] Frame '{self.child_frame}' created relative to '{self.parent_frame}'"
                )
                self._done = True
                return py_trees.common.Status.SUCCESS
            else:
                rospy.logerr(f"[{self.name}] Failed to create frame: {res.message}")
                return py_trees.common.Status.FAILURE

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"[{self.name}] TF error: {e}")
            return py_trees.common.Status.FAILURE

        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class PlanPathBehavior(py_trees.behaviour.Behaviour):
    """
    Calls the /set_plan service to generate a path.
    SMACH equivalent: SetPlanState (common.py:1084)
    """

    def __init__(self, name: str, target_frame: str, angle_offset: float = 0.0):
        super().__init__(name)
        self.target_frame = target_frame
        self.angle_offset = angle_offset
        self._done = False

    def setup(self, timeout=15, **kwargs):
        try:
            self.plan_srv = rospy.ServiceProxy("/set_plan", PlanPath)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        self._done = False

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        try:
            req = PlanPathRequest(
                target_frame=self.target_frame, angle_offset=self.angle_offset
            )
            res = self.plan_srv(req)

            if res.success:
                rospy.loginfo(f"[{self.name}] Path planned to {self.target_frame}")
                self._done = True
                return py_trees.common.Status.SUCCESS
            else:
                rospy.logerr(f"[{self.name}] Plan path failed (success=False)")
                return py_trees.common.Status.FAILURE

        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class ExecutePathBehavior(py_trees.behaviour.Behaviour):
    """
    Executes the planned path using FollowPathActionClient in a non-blocking thread.
    SMACH equivalent: ExecutePathState (common.py:655)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._client = None
        self._thread = None
        self._execution_result = None

    def setup(self, timeout=15, **kwargs):
        # Initializing client here ensures ROS is ready
        try:
            self._client = follow_path_client.FollowPathActionClient()
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        self._execution_result = None
        # Ensure previous thread is cleaned up if it exists
        if self._thread and self._thread.is_alive():
            rospy.logwarn(
                f"[{self.name}] Waiting for previous execution thread to finish..."
            )
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                rospy.logerr(
                    f"[{self.name}] CRITICAL: Thread did not finish ignoring cancel!"
                )
        self._thread = None

    def update(self):
        # If thread completed, return the result
        if self._execution_result is not None:
            return self._execution_result

        # Start thread if not started
        if self._thread is None:
            rospy.loginfo(f"[{self.name}] Starting path execution thread...")
            self._thread = threading.Thread(target=self._run_execution)
            self._thread.start()
            return py_trees.common.Status.RUNNING

        # While thread is alive, keep running
        if self._thread.is_alive():
            return py_trees.common.Status.RUNNING

        # Thread finished but result is somehow None (should not happen usually)
        return self._execution_result or py_trees.common.Status.FAILURE

    def _run_execution(self):
        """Thread worker function to run the blocking client call."""
        try:
            if self._client:
                # execute_paths is blocking.
                # We assume the user has reverted follow_path_client.py to original.
                # So we just call it.
                success = self._client.execute_paths([])
                if success:
                    rospy.loginfo(f"[{self.name}] Path execution successful")
                    self._execution_result = py_trees.common.Status.SUCCESS
                else:
                    rospy.logwarn(f"[{self.name}] Path execution failed/aborted")
                    self._execution_result = py_trees.common.Status.FAILURE
            else:
                rospy.logerr(f"[{self.name}] Client not initialized")
                self._execution_result = py_trees.common.Status.FAILURE
        except Exception as e:
            rospy.logerr(f"[{self.name}] Execution error: {e}")
            self._execution_result = py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Called when behavior terminates. Cancel execution if not successful."""
        if new_status != py_trees.common.Status.SUCCESS and self._client:
            rospy.logwarn(f"[{self.name}] Terminating: Cancelling path execution")
            try:
                # Access the inner SimpleActionClient directly
                self._client.client.cancel_all_goals()
            except AttributeError:
                rospy.logwarn(
                    f"[{self.name}] Could not cancel goals: inner client not found"
                )


class TriggerServiceBehavior(py_trees.behaviour.Behaviour):
    """
    Calls a Trigger service (e.g., /stop_planning).
    """

    def __init__(self, name: str, service_name: str):
        super().__init__(name)
        self.service_name = service_name
        self._done = False

    def setup(self, timeout=15, **kwargs):
        try:
            self.service = rospy.ServiceProxy(self.service_name, Trigger)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        self._done = False

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        try:
            self.service(TriggerRequest())
            rospy.loginfo(f"[{self.name}] Triggered {self.service_name}")
            self._done = True
            return py_trees.common.Status.SUCCESS
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class ResetOdometryBehavior(py_trees.behaviour.Behaviour):
    """
    Calls the /reset_odometry service to reset the EKF filter.
    SMACH equivalent: ResetOdometryState (initialize.py)
    """

    def __init__(self, name: str = "ResetOdometry"):
        super().__init__(name)

    def setup(self, timeout=15, **kwargs):
        try:
            self.service = rospy.ServiceProxy("reset_odometry", Trigger)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def update(self):
        try:
            self.service(TriggerRequest())
            rospy.loginfo(f"[{self.name}] Odometry reset triggered")
            return py_trees.common.Status.SUCCESS
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class DelayBehavior(py_trees.behaviour.Behaviour):
    """
    Waits for a specified duration.
    SMACH equivalent: DelayState (initialize.py)
    """

    def __init__(self, name: str, duration: float):
        super().__init__(name)
        self.duration = duration
        self._start_time = None

    def initialise(self):
        self._start_time = rospy.Time.now()

    def update(self):
        if (rospy.Time.now() - self._start_time).to_sec() >= self.duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class ClearObjectMapBehavior(py_trees.behaviour.Behaviour):
    """
    Calls the /clear_object_transforms service to clear the map.
    SMACH equivalent: ClearObjectMapState (common.py)
    """

    def __init__(self, name: str = "ClearObjectMap"):
        super().__init__(name)

    def setup(self, timeout=15, **kwargs):
        try:
            self.service = rospy.ServiceProxy("clear_object_transforms", Trigger)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def update(self):
        try:
            self.service(TriggerRequest())
            rospy.loginfo(f"[{self.name}] Object map cleared")
            return py_trees.common.Status.SUCCESS
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] Service error: {e}")
            return py_trees.common.Status.FAILURE


class CreateFrameAtCurrentPositionBehavior(py_trees.behaviour.Behaviour):
    """
    Creates a static TF frame at the current position of the source frame.
    Uses /set_object_transform service.
    """

    def __init__(
        self,
        name: str,
        target_frame_name: str,
        source_frame: str,
        reference_frame: str = "odom",
    ):
        super().__init__(name)
        self.target_frame_name = target_frame_name
        self.source_frame = source_frame
        self.reference_frame = reference_frame
        self._done = False

    def setup(self, timeout=15, **kwargs):
        try:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            self.srv = rospy.ServiceProxy("set_object_transform", SetObjectTransform)
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def initialise(self):
        self._done = False

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        try:
            # Lookup transform
            t = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.source_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            # Create new transform request
            req = SetObjectTransformRequest()
            req.transform = TransformStamped()
            req.transform.header.stamp = rospy.Time.now()
            req.transform.header.frame_id = self.reference_frame
            req.transform.child_frame_id = self.target_frame_name
            req.transform.transform = t.transform

            self.srv(req)
            self._done = True
            rospy.loginfo(
                f"[{self.name}] Created frame '{self.target_frame_name}' at current pos"
            )
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            rospy.logwarn(f"[{self.name}] TF/Service Error: {e}")
            return py_trees.common.Status.FAILURE
