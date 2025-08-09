from .initialize import *
import smach
import rospy
import math
import tf2_ros
from geometry_msgs.msg import Twist, Wrench
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion
from auv_smach.common import (
    SetDepthState,
    VisualServoingCentering,
    VisualServoingNavigation,
)
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest
from auv_smach.initialize import DelayState


class TurnAroundState(smach.State):
    def __init__(
        self,
        rotation_speed=0.2,
        timeout=15.0,
        rate_hz=10,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.odom_topic = "odometry"
        self.cmd_wrench_topic = "cmd_wrench"
        self.rotation_speed = rotation_speed
        self.timeout = timeout
        self.odom_data = False
        self.yaw = None
        self.yaw_prev = None
        self.total_yaw = 0.0
        self.rate = rospy.Rate(rate_hz)
        self.active = True

        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        from geometry_msgs.msg import Wrench

        self.pub = rospy.Publisher(self.cmd_wrench_topic, Wrench, queue_size=1)

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

    def execute(self, userdata):
        # Wait for odometry data
        while not rospy.is_shutdown() and not self.odom_data:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            self.rate.sleep()

        # Initialize rotation
        self.yaw_prev = self.yaw
        self.total_yaw = 0.0
        twist = Twist()
        twist.angular.z = self.rotation_speed
        # Wrench'e dönüştür
        scale_linear = 20.0
        scale_angular = 10.0
        wrench = Wrench()
        wrench.force.x = twist.linear.x * scale_linear
        wrench.force.y = twist.linear.y * scale_linear
        wrench.force.z = twist.linear.z * scale_linear
        wrench.torque.x = twist.angular.x * scale_angular
        wrench.torque.y = twist.angular.y * scale_angular
        wrench.torque.z = twist.angular.z * scale_angular
        self.active = True
        rotation_start_time = rospy.Time.now()

        rospy.loginfo("RotationState: Starting 180 degree rotation")

        # Rotate until 180 degrees (pi radians) is completed
        while not rospy.is_shutdown() and self.total_yaw < math.pi and self.active:
            if self.preempt_requested():
                twist.angular.z = 0.0
                wrench.torque.z = 0.0
                self.pub.publish(wrench)
                self.service_preempt()
                return "preempted"

            # Check timeout
            if (rospy.Time.now() - rotation_start_time).to_sec() > self.timeout:
                rospy.logwarn(
                    f"RotationState: Timeout reached after {self.timeout} seconds during rotation."
                )
                twist.angular.z = 0.0
                wrench.torque.z = 0.0
                self.pub.publish(wrench)
                return "aborted"

            # Enable propulsion and publish rotation command
            self.enable_pub.publish(Bool(data=True))
            self.pub.publish(wrench)

            # Calculate total rotation
            if self.yaw is not None and self.yaw_prev is not None:
                dyaw = TurnAroundState.normalize_angle(self.yaw - self.yaw_prev)
                self.total_yaw += abs(dyaw)
                self.yaw_prev = self.yaw

            self.rate.sleep()

        # Stop rotation
        twist.angular.z = 0.0
        wrench.torque.z = 0.0
        self.pub.publish(wrench)

        if not self.active:
            rospy.loginfo("RotationState: rotation aborted by killswitch.")
            return "aborted"

        rospy.loginfo(
            f"RotationState: completed 180 degree rotation. Total yaw: {self.total_yaw:.2f} radians"
        )
        return "succeeded"


class NavigateThroughGateStateVS(smach.State):
    def __init__(self, gate_depth: float, target_prop: str, wait_duration: float = 6.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        # Open the container for adding states
        with self.state_machine:

            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(
                    depth=gate_depth,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 5.0),
                ),
                transitions={
                    "succeeded": "VISUAL_SERVOING_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_CENTERING",
                VisualServoingCentering(target_prop=target_prop),
                transitions={
                    "succeeded": "WAIT_FOR_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_CENTERING",
                DelayState(delay_time=wait_duration),
                transitions={
                    "succeeded": "VISUAL_SERVOING_NAVIGATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_NAVIGATION",
                VisualServoingNavigation(),
                transitions={
                    "succeeded": "WAIT_FOR_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_EXIT",
                DelayState(delay_time=60.0),
                transitions={
                    "succeeded": "CANCEL_NAVIGATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_NAVIGATION",
                VisualServoingCancelNavigation(),
                transitions={
                    "succeeded": "CANCEL_VISUAL_SERVOING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_VISUAL_SERVOING",
                VisualServoingCancel(),
                transitions={
                    "succeeded": "TURN_AROUND",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TURN_AROUND",
                TurnAroundState(),
                transitions={
                    "succeeded": "FARUK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FARUK",
                VisualServoingCentering(target_prop=target_prop),
                transitions={
                    "succeeded": "WAIT_FOR_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughGateStateVS] Starting state machine execution.")

        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:  # ctrl + c
            return "preempted"
        # Return the outcome of the state machine
        return outcome


class VisualServoingCancelNavigation(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingCancelNavigation, self).__init__(
            "visual_servoing/cancel_navigation",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )


class VisualServoingCancel(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingCancel, self).__init__(
            "visual_servoing/cancel",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )
