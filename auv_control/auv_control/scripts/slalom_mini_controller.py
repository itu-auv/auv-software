#!/usr/bin/env python3

import math
from typing import Optional

import rospy
import tf.transformations
from auv_msgs.msg import SlalomTarget
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class SlalomMiniController:
    def __init__(self):
        rospy.init_node("slalom_mini_controller")

        self._load_parameters()

        self.enabled = self.auto_enable
        self.latest_target: Optional[SlalomTarget] = None
        self.latest_odom: Optional[Odometry] = None

        self.cmd_pose_pub = rospy.Publisher("cmd_pose", PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.yaw_error_pub = rospy.Publisher("slalom/yaw_error", Float64, queue_size=1)
        self.target_yaw_pub = rospy.Publisher(
            "slalom/target_yaw", Float64, queue_size=1
        )

        rospy.Subscriber(
            "slalom/target", SlalomTarget, self.target_callback, queue_size=1
        )
        rospy.Subscriber("odometry", Odometry, self.odom_callback, queue_size=1)

        rospy.Service("slalom_controller/set_enabled", SetBool, self.set_enabled)
        rospy.Service("slalom_controller/stop", Trigger, self.stop)

        rospy.loginfo(
            "Slalom mini controller started. depth=%.2f vx=%.2f lateral_strategy=%s",
            self.depth,
            self.forward_velocity,
            self.lateral_strategy,
        )

    def target_callback(self, msg: SlalomTarget):
        self.latest_target = msg
        self.yaw_error_pub.publish(Float64(msg.yaw_error))

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def _load_parameters(self):
        self.depth = rospy.get_param("~depth", -1.1)
        self.forward_velocity = rospy.get_param("~forward_velocity", 0.25)
        self.yaw_gain = rospy.get_param("~yaw_gain", 1.0)
        self.max_yaw_step = rospy.get_param("~max_yaw_step_rad", 0.0)
        self.target_timeout_s = rospy.get_param("~target_timeout_s", 0.5)
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)
        self.auto_enable = rospy.get_param("~auto_enable", True)
        self.lateral_strategy = rospy.get_param("~lateral_strategy", "none")
        self.lateral_gain = rospy.get_param("~lateral_gain", 0.0)
        self.max_lateral_velocity = rospy.get_param("~max_lateral_velocity", 0.15)

    def set_enabled(self, req: SetBool) -> SetBoolResponse:
        if req.data:
            self._load_parameters()
            rospy.loginfo(
                "Slalom mini controller config: depth=%.2f vx=%.2f yaw_gain=%.2f "
                "max_yaw_step=%.2f lateral_strategy=%s lateral_gain=%.2f max_vy=%.2f",
                self.depth,
                self.forward_velocity,
                self.yaw_gain,
                self.max_yaw_step,
                self.lateral_strategy,
                self.lateral_gain,
                self.max_lateral_velocity,
            )
        self.enabled = req.data
        if not self.enabled:
            self._publish_stop()
        state = "enabled" if self.enabled else "disabled"
        rospy.loginfo("Slalom mini controller %s.", state)
        return SetBoolResponse(success=True, message=f"Controller {state}.")

    def stop(self, req: Trigger) -> TriggerResponse:
        self.enabled = False
        self._publish_stop()
        return TriggerResponse(success=True, message="Controller stopped.")

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.enabled:
                self.control_step()
            rate.sleep()

    def control_step(self):
        if self.latest_odom is None:
            rospy.logwarn_throttle(1.0, "Waiting for odometry.")
            self._publish_stop()
            return

        if self.latest_target is None or self._target_is_stale():
            rospy.logwarn_throttle(1.0, "Waiting for fresh slalom target.")
            self._publish_stop()
            return

        current_yaw = self._current_yaw()
        yaw_step = self.yaw_gain * self.latest_target.yaw_error
        if self.max_yaw_step > 0.0:
            yaw_step = clamp(yaw_step, self.max_yaw_step)
        target_yaw = normalize_angle(current_yaw + yaw_step)
        self.target_yaw_pub.publish(Float64(target_yaw))

        self.enable_pub.publish(Bool(data=True))
        self.cmd_pose_pub.publish(self._build_cmd_pose(target_yaw))
        self.cmd_vel_pub.publish(self._build_cmd_vel())
        rospy.loginfo_throttle(
            0.5,
            "Slalom mini cmd_pose: x=%.2f y=%.2f z=%.2f current_yaw=%.2f "
            "yaw_error=%.2f target_yaw=%.2f",
            self.latest_odom.pose.pose.position.x,
            self.latest_odom.pose.pose.position.y,
            self.depth,
            current_yaw,
            self.latest_target.yaw_error,
            target_yaw,
        )

    def _target_is_stale(self) -> bool:
        stamp = self.latest_target.header.stamp
        if stamp == rospy.Time(0):
            return False
        return (rospy.Time.now() - stamp).to_sec() > self.target_timeout_s

    def _current_yaw(self) -> float:
        orientation = self.latest_odom.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def _build_cmd_pose(self, target_yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = self.latest_odom.header.stamp
        if pose.header.stamp == rospy.Time(0):
            pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.latest_odom.header.frame_id or "odom"
        pose.pose.position.x = self.latest_odom.pose.pose.position.x
        pose.pose.position.y = self.latest_odom.pose.pose.position.y
        pose.pose.position.z = self.depth

        q = tf.transformations.quaternion_from_euler(0.0, 0.0, target_yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def _build_cmd_vel(self) -> Twist:
        twist = Twist()
        twist.linear.x = self.forward_velocity
        if self.lateral_strategy == "yaw_error":
            twist.linear.y = clamp(
                self.lateral_gain * self.latest_target.yaw_error,
                self.max_lateral_velocity,
            )
        elif self.lateral_strategy != "none":
            rospy.logwarn_throttle(
                2.0,
                "Unknown lateral_strategy '%s'. Using zero lateral velocity.",
                self.lateral_strategy,
            )
        return twist

    def _publish_stop(self):
        self.cmd_vel_pub.publish(Twist())
        self.enable_pub.publish(Bool(data=False))


if __name__ == "__main__":
    try:
        SlalomMiniController().spin()
    except rospy.ROSInterruptException:
        pass
