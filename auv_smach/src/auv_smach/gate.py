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
    def __init__(
        self,
        imu_topic="/taluy/sensors/imu/data",
        wrench_topic="/taluy/wrench",
        roll_rate=1.0,
        Kp=5.0,
        min_wrench=0.5,
        max_wrench=60.0,
    ):
        super(TwoRollState, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.roll_rate = roll_rate
        self.Kp = Kp
        self.min_wrench = min_wrench
        self.max_wrench = max_wrench
        self.wrench_pub = rospy.Publisher(wrench_topic, WrenchStamped, queue_size=1)
        self.imu_data = None
        rospy.Subscriber(imu_topic, Imu, self.imu_cb)
        self.rate = rospy.Rate(20)
        self.start_time = None
        self.timeout = rospy.Duration(30)

    def imu_cb(self, msg):
        self.imu_data = msg

    def get_roll(self):
        if self.imu_data is None:
            rospy.logerr("IMU data is None!")
            return 0.0
        q = self.imu_data.orientation
        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
        return math.atan2(sinr, cosr)

    def wrap_angle(self, ang):
        return (ang + math.pi) % (2 * math.pi) - math.pi

    def clamp_wrench(self, value):

        if abs(value) < self.min_wrench:
            return math.copysign(self.min_wrench, value) if value != 0 else 0.0
        elif abs(value) > self.max_wrench:
            return math.copysign(self.max_wrench, value)
        return value

    def execute(self, userdata):
        wait_start = rospy.Time.now()
        while not rospy.is_shutdown() and self.imu_data is None:
            if (rospy.Time.now() - wait_start).to_sec() > 5.0:
                rospy.logerr("No IMU data received after 5 seconds. Aborting.")
                return "aborted"
            rospy.sleep(0.1)

        self.start_time = rospy.Time.now()
        prev_roll = self.get_roll()
        accumulated = 0.0
        target = 4 * math.pi

        rospy.loginfo(
            "Starting TwoRollState: target = %.1f rad (2 full rotations)", target
        )

        error_sum = 0.0
        last_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.publish_zero_wrench()
                rospy.loginfo(
                    "TwoRollState preempted after accumulating %.1f rad", accumulated
                )
                return "preempted"

            if (rospy.Time.now() - self.start_time) > self.timeout:
                rospy.logerr(
                    "TwoRollState timed out after %.1f seconds", self.timeout.to_sec()
                )
                self.publish_zero_wrench()
                return "aborted"

            try:
                current = self.get_roll()
                delta = self.wrap_angle(current - prev_roll)

                if abs(delta) > math.pi / 2:
                    rospy.logwarn(
                        "Large angle delta detected: %.2f rad, ignoring", delta
                    )
                else:
                    accumulated += abs(delta)

                prev_roll = current

                if accumulated >= target:
                    break

                now = rospy.Time.now()
                dt = (now - last_time).to_sec()
                last_time = now

                if dt > 0:
                    omega_x = self.imu_data.angular_velocity.x
                    error = self.roll_rate - omega_x

                    error_sum += error * dt
                    Ki = 0.1

                    torque_cmd = self.Kp * error + Ki * error_sum

                    torque_cmd = self.clamp_wrench(torque_cmd)

                    if abs(error_sum) > 10.0:
                        error_sum = math.copysign(10.0, error_sum)

                    wrench = WrenchStamped()
                    wrench.header.stamp = now
                    wrench.header.frame_id = "taluy/base_link"
                    wrench.wrench.torque.x = torque_cmd
                    self.wrench_pub.publish(wrench)

                    if rospy.get_time() % 1.0 < 0.05:
                        rospy.logdebug(
                            "Roll: %.2f, Accumulated: %.2f, Error: %.2f, Torque: %.2f",
                            current,
                            accumulated,
                            error,
                            torque_cmd,
                        )

            except Exception as e:
                rospy.logerr("Error in TwoRollState: %s", str(e))
                self.publish_zero_wrench()
                return "aborted"

            self.rate.sleep()

        self.publish_zero_wrench()
        duration = (rospy.Time.now() - self.start_time).to_sec()
        rospy.loginfo(
            "Two full X-rolls completed (%.1f rad) in %.1f seconds.",
            accumulated,
            duration,
        )
        return "succeeded"

    def publish_zero_wrench(self):
        zero = WrenchStamped()
        zero.header.stamp = rospy.Time.now()
        zero.header.frame_id = "taluy/base_link"
        self.wrench_pub.publish(zero)


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
                    "/dvl_to_odom_node/enable",
                    SetBool,
                    request=SetBoolRequest(data=False),
                ),
                transitions={
                    "succeeded": "TWO_ROLL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_ROLL",
                TwoRollState(),
                transitions={
                    "succeeded": "ENABLE_DVL_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_DVL_ODOM",
                smach_ros.ServiceState(
                    "/dvl_to_odom_node/enable",
                    SetBool,
                    request=SetBoolRequest(data=True),
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
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
