#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import BatteryState


class BatteryMonitorNode(Node):
    def __init__(self):
        # Initialize the node
        super().__init__("battery_monitor_node")

        # Parameters
        self.voltage_threshold = self.declare_parameter(
            "minimum_voltage_threshold", 13.5
        ).value
        self.voltage_warn_threshold = self.declare_parameter(
            "voltage_warn_threshold", 15.0
        ).value
        self.timeout_duration = self.declare_parameter(
            "power_message_timeout", 5
        ).value  # Timeout duration in seconds
        self.min_print_interval = self.declare_parameter(
            "min_log_interval", 1.0
        ).value  # Minimum print interval in seconds
        self.max_print_interval = self.declare_parameter(
            "max_log_interval", 10.0
        ).value  # Maximum print interval in seconds

        # Variables
        self.last_msg_time = self.get_clock().now()
        self.voltage = None
        self.is_timeouted = True
        self.is_undervoltage = False
        self.print_interval = 1.0  # Start with a slow print interval (e.g., 1 seconds)

        # Subscribers
        self.power_sub = self.create_subscription(
            BatteryState, "power", self.power_callback, 10
        )

        # Timers
        self.voltage_print_timer = self.create_timer(
            self.print_interval, self.print_voltage_callback
        )
        # self.timeout_check_timer = rospy.Timer(rospy.Duration(0.1), self.check_timeout_and_low_voltage_callback)

        self.get_logger().info(
            "Power Monitor Node started with voltage threshold: {:.2f}".format(
                self.voltage_threshold
            )
        )

    def power_callback(self, msg):
        self.voltage = msg.voltage
        self.is_undervoltage = self.voltage < self.voltage_threshold
        self.last_msg_time = self.get_clock().now()

    def reset_timer_with_new_interval(self, new_interval):
        self.print_interval = new_interval
        self.destroy_timer(self.voltage_print_timer)
        self.voltage_print_timer = self.create_timer(
            self.print_interval, self.print_voltage_callback
        )

    def print_voltage(self):

        if self.voltage == None:
            self.reset_timer_with_new_interval(1.0)
            self.get_logger().warning("No voltage received.")
            return

        self.is_timeouted = self.get_clock().now() - self.last_msg_time > Duration(
            seconds=self.timeout_duration
        )
        if self.is_timeouted:
            self.reset_timer_with_new_interval(1.0)
            self.get_logger().error(
                "Power message timeout. Last voltage: {:.2f} V".format(self.voltage)
            )
            return

        if self.voltage < self.voltage_threshold:
            self.reset_timer_with_new_interval(1.0)
            self.get_logger().error(
                "Battery voltage is below {:.2f}V threshold: {:.2f} V".format(
                    self.voltage_threshold, self.voltage
                )
            )
            return
        elif self.voltage < self.voltage_warn_threshold:
            self.reset_timer_with_new_interval(1.0)
            self.get_logger().info(
                "Battery voltage:"
                + "\033[93m"
                + " {:.2f} V".format(self.voltage)
                + "\033[0m"
            )
            return
        elif self.voltage >= self.voltage_warn_threshold:
            new_interval = self.interpolate_print_interval(self.voltage)
            self.reset_timer_with_new_interval(new_interval)
            self.get_logger().info(
                "Battery voltage:"
                + "\033[92m"
                + " {:.2f} V".format(self.voltage)
                + "\033[0m"
            )
            return

        # if still alive, set the timer for the next print
        self.print_interval = 1.0
        self.destroy_timer(self.voltage_print_timer)
        self.voltage_print_timer = self.create_timer(
            self.print_interval, self.print_voltage_callback
        )

    def print_voltage_callback(self):
        self.print_voltage()

    def check_timeout_and_low_voltage_callback(self, event):
        new_timeout = self.get_clock().now() - self.last_msg_time > Duration(
            seconds=self.timeout_duration
        )
        self.is_undervoltage = self.voltage < self.voltage_threshold

        if new_timeout != self.is_timeouted and self.voltage != None:
            if not self.is_undervoltage:
                self.print_voltage()
            self.is_timeouted = new_timeout

        if self.is_timeouted:
            if self.voltage != None:
                self.get_logger().error(
                    "Power message timeout. Last voltage: {:.2f} V".format(self.voltage)
                )
            else:
                self.get_logger().error("Power message timeout. No voltage received.")
            return

        if self.voltage != None and self.is_undervoltage:
            self.get_logger().error(
                "Battery voltage is below {:.2f}V threshold: {:.2f} V".format(
                    self.voltage_threshold, self.voltage
                )
            )

    def interpolate_print_interval(self, voltage):
        # Interpolate between 5 seconds and 60 seconds based on voltage
        min_voltage = self.voltage_threshold
        max_voltage = self.voltage_warn_threshold
        min_interval = self.min_print_interval
        max_interval = self.max_print_interval

        if voltage <= min_voltage:
            return min_interval
        elif voltage >= max_voltage:
            return max_interval
        else:
            # Linear interpolation
            return (
                (voltage - min_voltage)
                / (max_voltage - min_voltage)
                * (max_interval - min_interval)
            )

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = BatteryMonitorNode()
    try:
        node.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
