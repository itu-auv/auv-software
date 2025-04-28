from .initialize import *
import smach
import rospy
import tf2_ros
import math
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePlannedPathsState,
)
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped


class TwoRollState(smach.State):
    def __init__(self,
                 imu_topic='taluy/sensors/imu/data',
                 wrench_topic='taluy/wrench',
                 roll_rate=1.0,   
                 Kp=5.0):         
        super(TwoRollState, self).__init__(outcomes=['succeeded', 'preempted', 'aborted'])
        self.roll_rate = roll_rate
        self.Kp = Kp
        self.wrench_pub = rospy.Publisher(wrench_topic, WrenchStamped, queue_size=1)
        self.imu_data = None
        rospy.Subscriber(imu_topic, Imu, self.imu_cb)
        self.rate = rospy.Rate(20)

    def imu_cb(self, msg):
        self.imu_data = msg

    def get_roll(self):

        q = self.imu_data.orientation
        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
        return math.atan2(sinr, cosr)

    def wrap_angle(self, ang):

        return (ang + math.pi) % (2 * math.pi) - math.pi

    def execute(self, userdata):

        while not rospy.is_shutdown() and self.imu_data is None:
            rospy.sleep(0.01)

        prev_roll = self.get_roll()
        accumulated = 0.0
        target = 4 * math.pi  

        rospy.loginfo("Starting TwoRollState: target = %.1f rad", target)

        while not rospy.is_shutdown():
            if self.preempt_requested():
                zero = WrenchStamped()
                zero.header.stamp = rospy.Time.now()
                zero.header.frame_id = 'base_link'
                self.wrench_pub.publish(zero)
                return 'preempted'

            current = self.get_roll()
            delta = abs(self.wrap_angle(current - prev_roll))
            accumulated += delta
            prev_roll = current

            if accumulated >= target:
                break

            omega_x = self.imu_data.angular_velocity.x
            torque_cmd = self.Kp * (self.roll_rate - omega_x)

            wrench = WrenchStamped()
            wrench.header.stamp = rospy.Time.now()
            wrench.header.frame_id = 'base_link'
            wrench.wrench.torque.x = torque_cmd
            self.wrench_pub.publish(wrench)

            self.rate.sleep()

        stop = WrenchStamped()
        stop.header.stamp = rospy.Time.now()
        stop.header.frame_id = 'base_link'
        self.wrench_pub.publish(stop)

        rospy.loginfo("Two full X-rolls completed (%.1f rad).", accumulated)
        return 'succeeded'



class PlanGatePathsState(smach.State):
    """State that plans the paths for the gate task"""

    def __init__(self, tf_buffer):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanGatePathsState] Preempt requested")
                return "preempted"

            path_planners = PathPlanners(
                self.tf_buffer
            )  # instance of PathPlanners with tf_buffer
            paths = path_planners.path_for_gate()
            if paths is None:
                return "aborted"

            userdata.planned_paths = paths
            return "succeeded"

        except Exception as e:
            rospy.logerr("[PlanGatePathsState] Error: %s", str(e))
            return "aborted"


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:

            # smach.StateMachine.add(
            #     "ENABLE_GATE_TRAJECTORY_PUBLISHER",
            #     TransformServiceEnableState(req=True),
            #     transitions={
            #         "succeeded": "SET_GATE_DEPTH",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            # smach.StateMachine.add(
            #     "SET_GATE_DEPTH",
            #     SetDepthState(
            #         depth=gate_depth,
            #         sleep_duration=rospy.get_param("~set_depth_sleep_duration", 5.0),
            #     ),
            #     transitions={
            #         "succeeded": "DISABLE_GATE_TRAJECTORY_PUBLISHER",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            # smach.StateMachine.add(
            #     "DISABLE_GATE_TRAJECTORY_PUBLISHER",
            #     TransformServiceEnableState(req=False),
            #     transitions={
            #         "succeeded": "PLAN_GATE_PATHS",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            # smach.StateMachine.add(
            #     "PLAN_GATE_PATHS",
            #     PlanGatePathsState(self.tf_buffer),
            #     transitions={
            #         "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            # smach.StateMachine.add(
            #     "SET_ALIGN_CONTROLLER_TARGET",
            #     SetAlignControllerTargetState(
            #         source_frame="taluy/base_link", target_frame="dynamic_target"
            #     ),
            #     transitions={
            #         "succeeded": "EXECUTE_GATE_PATHS",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            # smach.StateMachine.add(
            #     "EXECUTE_GATE_PATHS",
            #     ExecutePlannedPathsState(),
            #     transitions={
            #         "succeeded": "CANCEL_ALIGN_CONTROLLER",
            #         "preempted": "CANCEL_ALIGN_CONTROLLER",  # if aborted or preempted, cancel the alignment request
            #         "aborted": "CANCEL_ALIGN_CONTROLLER",  # to disable the controllers.
            #     },
            # )
            # smach.StateMachine.add(
            #     "CANCEL_ALIGN_CONTROLLER",
            #     CancelAlignControllerState(),
            #     transitions={
            #         "succeeded": "DISABLE_DVL_ODOM",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            smach.StateMachine.add(
                "DISABLE_DVL_ODOM",
                smach_ros.ServiceState(
                    "/dvl_to_odom_node/enable", SetBool,
                    request=SetBoolRequest(data=False)
                ),
                transitions={
                    "succeeded": "TWO_ROLL", "preempted": "preempted", "aborted": "aborted"
                },
            )
            smach.StateMachine.add(
                "TWO_ROLL",
                TwoRollState(),
                transitions={
                    "succeeded": "ENABLE_DVL_ODOM", "preempted": "preempted", "aborted": "aborted"
                },
            )
            smach.StateMachine.add(
                "ENABLE_DVL_ODOM",
                smach_ros.ServiceState(
                    "/dvl_to_odom_node/enable", SetBool,
                    request=SetBoolRequest(data=True)
                ),
                transitions={
                    "succeeded": "succeeded", "preempted": "preempted", "aborted": "aborted"
                },
            )


    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughGateState] Starting state machine execution.")

        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:  # ctrl + c
            return "preempted"
        # Return the outcome of the state machine
        return outcome
