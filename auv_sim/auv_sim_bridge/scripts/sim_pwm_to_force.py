#!/usr/bin/env python3
import bisect
import time

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Float64, UInt16MultiArray


class SimPwmToForce(Node):
    def __init__(self):
        super().__init__("sim_pwm_to_force")
        self.declare_parameter("namespace", "taluy")
        self.declare_parameter("pwm_curve", [0.0])
        self.declare_parameter("force_curve", [0.0])
        self.declare_parameter("command_timeout", 1.0)
        namespace = self.get_parameter("namespace").value.strip("/")
        self.command_timeout = self.get_parameter("command_timeout").value
        self.pwm = self.get_parameter("pwm_curve").value
        self.force = self.get_parameter("force_curve").value
        if len(self.pwm) != len(self.force) or len(self.pwm) < 2:
            raise RuntimeError("pwm_curve and force_curve must have equal lengths >= 2")
        self.thrust_publishers = [
            self.create_publisher(
                Float64,
                f"/model/{namespace}/joint/{namespace}/thruster_{i}_propeller_joint/cmd_thrust",
                1,
            )
            for i in range(8)
        ]
        self.subscription = self.create_subscription(
            UInt16MultiArray, f"/{namespace}/board/drive_pulse", self.on_pwm, 1
        )
        self.last_command_time = time.monotonic()
        self.timed_out = False
        self.watchdog = self.create_timer(
            0.05,
            self.check_command_timeout,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def on_pwm(self, message):
        if len(message.data) < len(self.thrust_publishers):
            self.get_logger().warning("Received PWM array with fewer than 8 channels")
            return
        self.last_command_time = time.monotonic()
        self.timed_out = False
        for pwm, publisher in zip(message.data, self.thrust_publishers):
            index = bisect.bisect_left(self.pwm, pwm)
            if index == 0:
                force = self.force[0]
            elif index >= len(self.pwm):
                force = self.force[-1]
            elif self.pwm[index] == pwm:
                force = self.force[index]
            else:
                fraction = (pwm - self.pwm[index - 1]) / (
                    self.pwm[index] - self.pwm[index - 1]
                )
                force = self.force[index - 1] + fraction * (
                    self.force[index] - self.force[index - 1]
                )
            output = Float64()
            output.data = float(force)
            publisher.publish(output)

    def check_command_timeout(self):
        if (
            self.timed_out
            or time.monotonic() - self.last_command_time <= self.command_timeout
        ):
            return
        self.publish_neutral()
        self.timed_out = True
        self.get_logger().warning(
            "PWM input timed out; publishing neutral thrust to all thrusters"
        )

    def publish_neutral(self):
        neutral_index = bisect.bisect_left(self.pwm, 1500)
        if neutral_index >= len(self.pwm) or self.pwm[neutral_index] != 1500:
            self.get_logger().error("PWM curve does not contain the neutral 1500 point")
            return
        neutral_force = float(self.force[neutral_index])
        for publisher in self.thrust_publishers:
            output = Float64()
            output.data = neutral_force
            publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = SimPwmToForce()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
