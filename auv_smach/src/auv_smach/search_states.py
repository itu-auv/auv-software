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

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from tf.transformations import euler_from_quaternion

from auv_smach.alignment_states import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
)


class RotationState(smach.State):
    def __init__(
        self,
        source_frame,
        look_at_frame,
        rotation_speed=0.3,
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
                    source_frame=source_frame, target_frame=alignment_frame
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
