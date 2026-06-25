#!/usr/bin/env python3

import threading

import actionlib
import rospy
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Twist, WrenchStamped
from std_msgs.msg import Bool, String

from auv_control.cfg import MiniSlalomConfig
from auv_msgs.msg import (
    MiniSlalomAction,
    MiniSlalomFeedback,
    MiniSlalomResult,
    SlalomTarget,
)
from mini_slalom_core import ControllerConfig, SlalomController, Target


class MiniSlalomControllerNode:
    def __init__(self):
        rospy.init_node("mini_slalom_controller")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.35))
        self.overall_timeout = float(rospy.get_param("~overall_timeout", 90.0))
        self.body_frame = rospy.get_param("~body_frame", "taluy_mini/base_link")
        self.config = ControllerConfig(
            exit_duration=float(rospy.get_param("~exit_duration", 1.5)),
            yaw_sign=float(rospy.get_param("~yaw_sign", -1.0)),
        )
        self.controller = SlalomController(self.config)
        self.target_lock = threading.Lock()
        self.latest_target = None

        self.wrench_pub = rospy.Publisher(
            "slalom/visual_wrench", WrenchStamped, queue_size=1
        )
        self.active_pub = rospy.Publisher(
            "slalom/active", Bool, queue_size=1, latch=True
        )
        self.direction_pub = rospy.Publisher(
            "slalom/direction", String, queue_size=1, latch=True
        )
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(
            "slalom/target", SlalomTarget, self.target_callback, queue_size=1
        )

        self.reconfigure_server = Server(MiniSlalomConfig, self.reconfigure_callback)
        self.server = actionlib.SimpleActionServer(
            "slalom/run",
            MiniSlalomAction,
            execute_cb=self.execute,
            auto_start=False,
        )
        self.active_pub.publish(False)
        self.server.start()
        rospy.loginfo("Mini slalom controller action server started")

    def reconfigure_callback(self, config, _level):
        self.config.yaw_kp = config.yaw_kp
        self.config.yaw_kd = config.yaw_kd
        self.config.max_yaw_torque = config.max_yaw_torque
        self.config.forward_force = config.forward_force
        self.config.exit_force = config.exit_force
        self.config.align_error_threshold = config.align_error_threshold
        self.config.near_height_ratio = config.near_height_ratio
        self.config.near_width_ratio = config.near_width_ratio
        self.config.pass_loss_duration = config.pass_loss_duration
        self.config.search_yaw_torque = config.search_yaw_torque
        return config

    def target_callback(self, msg):
        with self.target_lock:
            self.latest_target = msg

    def get_target(self, now):
        with self.target_lock:
            msg = self.latest_target
        if msg is None:
            return Target()
        stamp = msg.header.stamp
        if stamp == rospy.Time():
            stamp = now
        age = (now - stamp).to_sec()
        if age < 0.0 or age > self.target_timeout:
            return Target()
        return Target(
            valid=msg.valid,
            center_error=msg.center_error,
            gate_width_ratio=msg.gate_width_ratio,
            gate_height_ratio=msg.gate_height_ratio,
            red_center_x=msg.red_center_x,
            white_center_x=msg.white_center_x,
            confidence=msg.confidence,
        )

    def publish_command(self, force_x, torque_z):
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.body_frame
        msg.wrench.force.x = force_x
        msg.wrench.torque.z = torque_z
        self.wrench_pub.publish(msg)

    def stop(self):
        for _ in range(3):
            self.publish_command(0.0, 0.0)
            self.cmd_vel_pub.publish(Twist())
            self.active_pub.publish(False)
            rospy.sleep(0.03)

    def execute(self, goal):
        direction = goal.direction.lower()
        if direction not in ("left", "right"):
            self.server.set_aborted(
                MiniSlalomResult(
                    success=False,
                    message="direction must be 'left' or 'right'",
                    gates_passed=0,
                )
            )
            return

        gate_count = int(goal.target_gate_count) or 3
        self.controller.reset(direction, gate_count)
        with self.target_lock:
            self.latest_target = None
        self.direction_pub.publish(String(data=direction))
        start_time = rospy.Time.now()
        rate = rospy.Rate(self.rate_hz)

        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    self.server.set_preempted(
                        MiniSlalomResult(
                            success=False,
                            message="mini slalom preempted",
                            gates_passed=self.controller.gates_passed,
                        )
                    )
                    return

                now = rospy.Time.now()
                if (now - start_time).to_sec() > self.overall_timeout:
                    self.server.set_aborted(
                        MiniSlalomResult(
                            success=False,
                            message="mini slalom overall timeout",
                            gates_passed=self.controller.gates_passed,
                        )
                    )
                    return

                target = self.get_target(now)
                command = self.controller.update(now.to_sec(), target)
                self.active_pub.publish(True)
                self.enable_pub.publish(True)
                self.cmd_vel_pub.publish(Twist())
                self.publish_command(command.force_x, command.torque_z)
                self.server.publish_feedback(
                    MiniSlalomFeedback(
                        state=command.state,
                        gates_passed=command.gates_passed,
                        horizontal_error=target.center_error if target.valid else 0.0,
                    )
                )

                if command.finished:
                    self.server.set_succeeded(
                        MiniSlalomResult(
                            success=True,
                            message="target gate count completed",
                            gates_passed=command.gates_passed,
                        )
                    )
                    return
                rate.sleep()
        finally:
            self.stop()


if __name__ == "__main__":
    try:
        MiniSlalomControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
