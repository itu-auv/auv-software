import math

import rospy
import smach
import tf.transformations as transformations
from geometry_msgs.msg import TransformStamped, WrenchStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import Trigger

from auv_msgs.srv import (
    AlignFrameController,
    AlignFrameControllerRequest,
    SetObjectTransform,
    SetObjectTransformRequest,
)
from auv_smach.common import (
    SearchForPropState,
    SetDepthState,
    SetDetectionFocusState,
    SetDetectionState,
)
from auv_smach.initialize import DelayState
from auv_smach.tf_utils import get_base_link


class FollowMiniSlalomState(smach.State):
    def __init__(
        self,
        depth: float,
        white_side: str = "right",
        forward_wrench: float = 15.0,
        lateral_kp: float = 0.0,
        lateral_kd: float = 0.0,
        max_lateral_wrench: float = 30.0,
        max_angular_velocity: float = 0.20,
        duration: float = 0.0,
        pipe_angle_stale_timeout: float = 3.0,
        rate_hz: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        if white_side not in ["left", "right"]:
            raise ValueError("white_side must be 'left' or 'right'")

        self.depth = depth
        self.white_side = white_side
        self.forward_wrench = forward_wrench
        self.lateral_kp = lateral_kp
        self.lateral_kd = lateral_kd
        self.max_lateral_wrench = abs(max_lateral_wrench)
        self.max_angular_velocity = max_angular_velocity
        self.duration = duration
        self.rate_hz = rate_hz
        self.base_link = get_base_link()
        self.latest_odom = None
        self.latest_pipe_angles = None
        self.last_lateral_error = None
        self.last_lateral_error_time = None
        self.follow_frame = "slalom_mini_follow"
        self.alignment_started = False
        self.pipe_angle_stale_timeout = rospy.Duration(pipe_angle_stale_timeout)
        self.last_pipe_angle_values = None
        self.last_pipe_angle_change_time = None

        self.cmd_wrench_pub = rospy.Publisher("cmd_wrench", WrenchStamped, queue_size=1)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.set_object_transform = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.align_start = rospy.ServiceProxy("align_frame/start", AlignFrameController)
        self.align_cancel = rospy.ServiceProxy("align_frame/cancel", Trigger)
        self.odom_sub = rospy.Subscriber("odometry", Odometry, self.odom_callback)
        self.pipe_angle_sub = rospy.Subscriber(
            "slalom/pipe_angles", Float32MultiArray, self.pipe_angle_callback
        )

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def pipe_angle_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 6:
            rospy.logwarn_throttle(
                2.0,
                "[FollowMiniSlalomState] Expected 6 slalom values, got %d",
                len(msg.data),
            )
            return
        pipe_angle_values = tuple(msg.data[:3])
        if self.last_pipe_angle_values is None or not self.pipe_angle_values_equal(
            self.last_pipe_angle_values, pipe_angle_values
        ):
            self.last_pipe_angle_values = pipe_angle_values
            self.last_pipe_angle_change_time = rospy.Time.now()
        self.latest_pipe_angles = msg

    def execute(self, userdata) -> str:
        rate = rospy.Rate(self.rate_hz)
        started_at = rospy.Time.now()

        while not rospy.is_shutdown():
            self.enable_pub.publish(Bool(data=True))

            if self.preempt_requested():
                self.service_preempt()
                self.cancel_alignment()
                self.publish_zero_wrench()
                return "preempted"

            if (
                self.duration > 0
                and (rospy.Time.now() - started_at).to_sec() > self.duration
            ):
                self.cancel_alignment()
                self.publish_zero_wrench()
                return "succeeded"

            if self.pipe_angles_stale():
                rospy.loginfo(
                    "[FollowMiniSlalomState] First 3 slalom/pipe_angles values "
                    "unchanged for %.1f seconds, succeeding",
                    self.pipe_angle_stale_timeout.to_sec(),
                )
                self.cancel_alignment()
                self.publish_zero_wrench()
                return "succeeded"

            if self.latest_odom is None or self.latest_pipe_angles is None:
                rospy.logwarn_throttle(
                    2.0,
                    "[FollowMiniSlalomState] Waiting for odometry and slalom/pipe_angles",
                )
                rate.sleep()
                continue

            target_relative_yaw = self.target_relative_yaw()
            lateral_error = self.lateral_height_error()
            if target_relative_yaw is None or lateral_error is None:
                rospy.logwarn_throttle(
                    2.0,
                    "[FollowMiniSlalomState] Waiting for valid red and %s white slalom data",
                    self.white_side,
                )
                rate.sleep()
                continue

            if not self.publish_follow_frame(target_relative_yaw):
                rate.sleep()
                continue

            if not self.ensure_alignment_started():
                rate.sleep()
                continue

            self.cmd_wrench_pub.publish(self.build_cmd_wrench(lateral_error))
            rate.sleep()

        return "aborted"

    def pipe_angles_stale(self):
        if self.last_pipe_angle_change_time is None:
            return False
        return (
            rospy.Time.now() - self.last_pipe_angle_change_time
            > self.pipe_angle_stale_timeout
        )

    def target_relative_yaw(self):
        red_angle = self.latest_pipe_angles.data[0]
        white_index = 1 if self.white_side == "left" else 2
        white_angle = self.latest_pipe_angles.data[white_index]

        if math.isnan(red_angle) or math.isnan(white_angle):
            return None

        return self.average_angles([red_angle, white_angle])

    def lateral_height_error(self):
        red_height = self.latest_pipe_angles.data[3]
        white_index = 4 if self.white_side == "left" else 5
        white_height = self.latest_pipe_angles.data[white_index]

        if math.isnan(red_height) or math.isnan(white_height):
            return None

        return red_height - white_height

    def publish_follow_frame(self, target_relative_yaw: float):
        odom = self.latest_odom
        current_orientation = odom.pose.pose.orientation
        _, _, current_yaw = transformations.euler_from_quaternion(
            [
                current_orientation.x,
                current_orientation.y,
                current_orientation.z,
                current_orientation.w,
            ]
        )
        target_yaw = self.normalize_angle(current_yaw + target_relative_yaw)
        quat = transformations.quaternion_from_euler(0.0, 0.0, target_yaw)

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = odom.header.frame_id or "odom"
        transform.child_frame_id = self.follow_frame
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = self.depth
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        req = SetObjectTransformRequest()
        req.transform = transform

        try:
            res = self.set_object_transform(req)
            if not res.success:
                rospy.logwarn_throttle(
                    2.0,
                    "[FollowMiniSlalomState] set_object_transform failed: %s",
                    res.message,
                )
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(
                2.0, "[FollowMiniSlalomState] set_object_transform call failed: %s", e
            )
            return False

    def ensure_alignment_started(self):
        if self.alignment_started:
            return True

        req = AlignFrameControllerRequest()
        req.source_frame = self.base_link
        req.target_frame = self.follow_frame
        req.angle_offset = 0.0
        req.keep_orientation = False
        req.use_depth = True
        req.closest_yaw = False
        req.max_angular_velocity = self.max_angular_velocity

        try:
            res = self.align_start(req)
            if not res.success:
                rospy.logwarn_throttle(
                    2.0,
                    "[FollowMiniSlalomState] align_frame/start failed: %s",
                    res.message,
                )
                return False
            self.alignment_started = True
            return True
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(
                2.0, "[FollowMiniSlalomState] align_frame/start call failed: %s", e
            )
            return False

    def cancel_alignment(self):
        if not self.alignment_started:
            return

        try:
            self.align_cancel()
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(
                2.0, "[FollowMiniSlalomState] align_frame/cancel call failed: %s", e
            )
        self.alignment_started = False

    def build_cmd_wrench(self, lateral_error: float):
        now = rospy.Time.now()
        derivative = 0.0

        if (
            self.last_lateral_error is not None
            and self.last_lateral_error_time is not None
        ):
            dt = (now - self.last_lateral_error_time).to_sec()
            if dt > 1e-3:
                derivative = (lateral_error - self.last_lateral_error) / dt

        self.last_lateral_error = lateral_error
        self.last_lateral_error_time = now

        lateral_wrench = self.lateral_kp * lateral_error + self.lateral_kd * derivative
        lateral_wrench = max(
            -self.max_lateral_wrench,
            min(self.max_lateral_wrench, lateral_wrench),
        )

        cmd_wrench = WrenchStamped()
        cmd_wrench.header.stamp = now
        cmd_wrench.header.frame_id = self.base_link
        cmd_wrench.wrench.force.x = self.forward_wrench
        cmd_wrench.wrench.force.y = lateral_wrench
        return cmd_wrench

    def publish_zero_wrench(self):
        stop = WrenchStamped()
        stop.header.stamp = rospy.Time.now()
        stop.header.frame_id = self.base_link
        self.cmd_wrench_pub.publish(stop)

    @staticmethod
    def average_angles(angles):
        return math.atan2(
            sum(math.sin(angle) for angle in angles),
            sum(math.cos(angle) for angle in angles),
        )

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def pipe_angle_values_equal(left, right):
        return all(
            (math.isnan(left_value) and math.isnan(right_value))
            or left_value == right_value
            for left_value, right_value in zip(left, right)
        )


class NavigateThroughSlalomMiniState(smach.State):
    def __init__(
        self,
        slalom_depth: float,
        white_side: str = "right",
        forward_wrench: float = 0.0,
        lateral_kp: float = 0.0,
        lateral_kd: float = 0.0,
        max_lateral_wrench: float = 30.0,
        max_angular_velocity: float = 0.15,
        follow_duration: float = 0.0,
        pipe_angle_stale_timeout: float = 3.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.base_link = get_base_link()
        self.slalom_depth = slalom_depth
        self.white_side = white_side
        self.forward_wrench = forward_wrench
        self.lateral_kp = lateral_kp
        self.lateral_kd = lateral_kd
        self.max_lateral_wrench = max_lateral_wrench
        self.max_angular_velocity = max_angular_velocity
        self.follow_duration = follow_duration
        self.pipe_angle_stale_timeout = pipe_angle_stale_timeout

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_SLALOM_DEPTH",
                SetDepthState(depth=self.slalom_depth),
                transitions={
                    "succeeded": "ENABLE_SLALOM_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_SLALOM_DETECTION",
                SetDetectionState(camera_name="slalom", enable=True),
                transitions={
                    "succeeded": "SET_SLALOM_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_SLALOM_FOCUS",
                SetDetectionFocusState(focus_object="slalom"),
                transitions={
                    "succeeded": "SEARCH_RED_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_RED_PIPE",
                SearchForPropState(
                    look_at_frame="slalom_red_pipe_link",
                    alignment_frame="slalom_mini_search",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "APPROACH_PLACEHOLDER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "APPROACH_PLACEHOLDER",
                DelayState(delay_time=0.1),
                transitions={
                    "succeeded": "FOLLOW_SLALOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOLLOW_SLALOM",
                FollowMiniSlalomState(
                    depth=self.slalom_depth,
                    white_side=self.white_side,
                    forward_wrench=self.forward_wrench,
                    lateral_kp=self.lateral_kp,
                    lateral_kd=self.lateral_kd,
                    max_lateral_wrench=self.max_lateral_wrench,
                    max_angular_velocity=self.max_angular_velocity,
                    duration=self.follow_duration,
                    pipe_angle_stale_timeout=self.pipe_angle_stale_timeout,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        return self.state_machine.execute()
