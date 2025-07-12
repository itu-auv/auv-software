import smach
import rospy
from std_srvs.srv import SetBool, SetBoolRequest
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from .initialize import *
import smach_ros
import math
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    ClearObjectMapState,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool


class DvlOdometryServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "dvl_to_odom_node/enable",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PitchCorrection(smach.State):
    def __init__(self, fixed_torque=2.0, rate_hz=20, timeout_s=10.0):
        super(PitchCorrection, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.odometry_topic = "odometry"
        self.killswitch_topic = "propulsion_board/status"
        self.wrench_topic = "wrench"
        self.frame_id = "taluy/base_link"

        self.fixed_torque = -abs(fixed_torque)
        self.timeout = rospy.Duration(timeout_s)
        self.rate = rospy.Rate(rate_hz)

        self.current_pitch = 0.0
        self.odom_ready = False
        self.active = True
        self.start_time = None

        self.sub_odom = rospy.Subscriber(self.odometry_topic, Odometry, self.odom_cb)
        self.sub_kill = rospy.Subscriber(
            self.killswitch_topic, Bool, self.killswitch_cb
        )
        self.pub_wrench = rospy.Publisher(
            self.wrench_topic, WrenchStamped, queue_size=1
        )

    def odom_cb(self, msg: Odometry):
        orientation = msg.pose.pose.orientation
        q = [orientation.x, orientation.y, orientation.z, orientation.w]

        try:
            _, pitch, _ = euler_from_quaternion(q)
            self.current_pitch = pitch
            self.odom_ready = True
        except:
            rospy.logwarn("PITCH_CORRECTION: Quaternion conversion failed!")

    def killswitch_cb(self, msg: Bool):
        if not msg.data:
            self.active = False
            rospy.logwarn("PITCH_CORRECTION: Propulsion board disabled → aborting")

    def execute(self, userdata):
        rospy.loginfo("PITCH_CORRECTION: Waiting for odometry data...")
        start_wait = rospy.Time.now()

        while not rospy.is_shutdown() and not self.odom_ready:
            if (rospy.Time.now() - start_wait) > rospy.Duration(5.0):
                rospy.logerr("PITCH_CORRECTION: No odometry data → abort")
                return "aborted"
            if self.preempt_requested():
                return "preempted"
            self.rate.sleep()

        if self.current_pitch <= 0:
            rospy.loginfo(
                "PITCH_CORRECTION: Pitch already corrected (%.2f°)",
                math.degrees(self.current_pitch),
            )
            return "succeeded"

        rospy.loginfo(
            "PITCH_CORRECTION: Correcting pitch from %.2f° with fixed torque %.2f Nm",
            math.degrees(self.current_pitch),
            self.fixed_torque,
        )

        self.start_time = rospy.Time.now()

        try:
            while not rospy.is_shutdown() and self.active:
                if self.preempt_requested():
                    return self._stop_and("preempted")
                if (rospy.Time.now() - self.start_time) > self.timeout:
                    rospy.logerr(
                        "PITCH_CORRECTION: Timeout after %.1f s", self.timeout.to_sec()
                    )
                    return self._stop_and("aborted")

                if self.current_pitch <= 0:
                    rospy.loginfo(
                        "PITCH_CORRECTION: Pitch corrected to %.2f°",
                        math.degrees(self.current_pitch),
                    )
                    return self._stop_and("succeeded")

                cmd = WrenchStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.header.frame_id = self.frame_id
                cmd.wrench.torque.y = self.fixed_torque
                self.pub_wrench.publish(cmd)

                rospy.loginfo_throttle(
                    1.0,
                    "PITCH_CORRECTION: Current pitch: %.2f°",
                    math.degrees(self.current_pitch),
                )

                self.rate.sleep()

        except rospy.ROSInterruptException:
            return self._abort_on_shutdown()

        return "aborted"

    def _stop_and(self, outcome):
        stop_cmd = WrenchStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.header.frame_id = self.frame_id
        self.pub_wrench.publish(stop_cmd)
        return outcome

    def _abort_on_shutdown(self):
        rospy.logwarn("PITCH_CORRECTION: ROS shutdown → aborting")
        return self._stop_and("aborted")


class RollTwoTimes(smach.State):
    def __init__(self, roll_torque, rate_hz=20, timeout_s=15.0):
        super(RollTwoTimes, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.odometry_topic = "odometry"
        self.killswitch_topic = "propulsion_board/status"
        self.wrench_topic = "wrench"
        self.frame_id = "taluy/base_link"

        self.roll_torque = roll_torque
        self.timeout = rospy.Duration(timeout_s)
        self.rate = rospy.Rate(rate_hz)

        self.odom_ready = False
        self.active = True
        self.total_roll = 0.0
        self.last_time = None

        self.sub_odom = rospy.Subscriber(self.odometry_topic, Odometry, self.odom_cb)
        self.sub_kill = rospy.Subscriber(
            self.killswitch_topic, Bool, self.killswitch_cb
        )
        self.pub_wrench = rospy.Publisher(
            self.wrench_topic, WrenchStamped, queue_size=1
        )

    def odom_cb(self, msg: Odometry):
        now = rospy.Time.now()

        if not self.odom_ready:
            self.last_time = now
            self.odom_ready = True
            return

        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).to_sec()
        self.last_time = now

        omega_x = msg.twist.twist.angular.x
        delta_angle = omega_x * dt
        self.total_roll += abs(delta_angle)

    def killswitch_cb(self, msg: Bool):
        if not msg.data:
            self.active = False
            rospy.logwarn("ROLL_TWO_TIMES: Propulsion board disabled → aborting")

    def execute(self, userdata):
        rospy.loginfo("ROLL_TWO_TIMES: Waiting for odometry data...")
        start_wait = rospy.Time.now()

        while not rospy.is_shutdown() and not self.odom_ready:
            if (rospy.Time.now() - start_wait) > rospy.Duration(5.0):
                rospy.logerr("ROLL_TWO_TIMES: No odometry data → abort")
                return "aborted"
            if self.preempt_requested():
                return "preempted"
            self.rate.sleep()

        rospy.loginfo(
            "ROLL_TWO_TIMES: Starting roll with torque %.2f Nm", self.roll_torque
        )
        self.start_time = rospy.Time.now()

        try:
            while not rospy.is_shutdown() and self.active:
                if self.preempt_requested():
                    return self._stop_and("preempted")
                if (rospy.Time.now() - self.start_time) > self.timeout:
                    rospy.logerr(
                        "ROLL_TWO_TIMES: Timeout after %.1f s", self.timeout.to_sec()
                    )
                    return self._stop_and("aborted")

                if self.total_roll >= 2 * math.pi:
                    rospy.loginfo("ROLL_TWO_TIMES: Completed two full rolls")
                    return self._stop_and("succeeded")

                cmd = WrenchStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.header.frame_id = self.frame_id
                cmd.wrench.torque.x = self.roll_torque
                self.pub_wrench.publish(cmd)

                rospy.loginfo_throttle(
                    1.0,
                    "ROLL_TWO_TIMES: Total roll: %.2f°",
                    math.degrees(self.total_roll),
                )

                self.rate.sleep()

        except rospy.ROSInterruptException:
            return self._abort_on_shutdown()

        return "aborted"

    def _stop_and(self, outcome):
        stop_cmd = WrenchStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.header.frame_id = self.frame_id
        self.pub_wrench.publish(stop_cmd)
        return outcome

    def _abort_on_shutdown(self):
        rospy.logwarn("ROLL_TWO_TIMES: ROS shutdown → aborting")
        return self._stop_and("aborted")


class TwoRollState(smach.StateMachine):
    def __init__(self, roll_torque=50.0):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        self.roll_torque = roll_torque

        with self:
            smach.StateMachine.add(
                "DISABLE_DVL_ODOM",
                DvlOdometryServiceEnableState(req=False),
                transitions={
                    "succeeded": "WAIT_FOR_PITCH_CORRECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_PITCH_CORRECTION",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "PITCH_CORRECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PITCH_CORRECTION",
                PitchCorrection(fixed_torque=3.0, rate_hz=20, timeout_s=10.0),
                transitions={
                    "succeeded": "ROLL_TWO_TIMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROLL_TWO_TIMES",
                RollTwoTimes(roll_torque=self.roll_torque, rate_hz=20, timeout_s=15.0),
                transitions={
                    "succeeded": "WAIT_FOR_STABILIZATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_STABILIZATION",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ENABLE_DVL_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_DVL_ODOM",
                DvlOdometryServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_DVL_ODOM_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_DVL_ODOM_ENABLE",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "ALIGN_TO_LOOK_AT_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_LOOK_AT_GATE",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="gate_search"
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNMENT",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER_BEFORE_ODOM_ENABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER_BEFORE_ODOM_ENABLE",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "RESET_ODOMETRY_POSE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESET_ODOMETRY_POSE",
                ResetOdometryState(),
                transitions={
                    "succeeded": "DELAY_AFTER_RESET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DELAY_AFTER_RESET",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "CLEAR_OBJECT_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLEAR_OBJECT_MAP",
                ClearObjectMapState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
