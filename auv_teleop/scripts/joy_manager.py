#!/usr/bin/env python3

import rclpy
from auv_common_lib.ros import (
    get_param,
    service_client,
    ROSException,
    Rate,
    on_shutdown,
)
from auv_common_lib import ros
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import threading
from std_srvs.srv import Trigger
from auv_common_lib.reconfigure import Client as ReconfigureClient


class JoystickEvent:
    def __init__(self, change_threshold, callback):
        self.previous_value = 0.0
        self.change_threshold = change_threshold
        self.callback = callback

    def update(self, value):
        if (value - self.previous_value) > self.change_threshold:
            self.callback()
        self.previous_value = value


class JoystickNode(Node):
    def __init__(self):
        super().__init__("joystick_node")

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.enable_pub = self.create_publisher(Bool, "enable", 10)

        self.joy_data = None
        self.lock = threading.Lock()

        self.publish_rate = 50  # 50 Hz
        self.rate = Rate(self, self.publish_rate)

        self.buttons = get_param(self, "~buttons")
        self.axes = get_param(self, "~axes")
        self.rov_mode = get_param(self, "~rov_mode", 0)

        if self.rov_mode == 1:
            self.get_logger().info(
                "ROV mode enabled - using buttons for pitch/roll control"
            )
            self.reset_orientation_button_event = JoystickEvent(
                0.1, self.reset_orientation
            )
            self.reset_orientation_service = service_client(
                self, Trigger, "reset_command_orientation"
            )
        else:
            self.torpedo1_button_event = JoystickEvent(0.1, self.launch_torpedo1)
            self.torpedo2_button_event = JoystickEvent(0.1, self.launch_torpedo2)
            self.dropper_button_event = JoystickEvent(0.1, self.drop_dropper)

            self.dropper_service = service_client(self, Trigger, "ball_dropper/drop")

            self.torpedo1_service = service_client(self, Trigger, "torpedo_1/launch")

            self.torpedo2_service = service_client(self, Trigger, "torpedo_2/launch")

        self.joy_sub = self.create_subscription(Joy, "joy", self.joy_callback, 10)
        # smach_monitor_node stops teleop through this (ROS 2 has no `rosnode kill`)
        self.shutdown_service = self.create_service(
            Trigger, "~/shutdown", self.handle_shutdown
        )
        self.get_logger().info("Joystick node initialized")

        # use_vel mode: zero position PID gains for X and Y
        self.use_vel = get_param(self, "~use_vel", False)
        self.original_pid_gains = None
        self.reconfigure_client = None

        if self.use_vel:
            self._setup_use_vel_mode()
            on_shutdown(self._restore_pid_gains)

    def handle_shutdown(self, request, response):
        self.get_logger().info("Shutdown requested, stopping joystick node")
        self.shutdown_requested = True
        response.success = True
        return response

    def _setup_use_vel_mode(self):
        """Setup use_vel mode by connecting to controller reconfigure and zeroing X/Y position gains."""
        try:
            controller_server = get_param(
                self, "~controller_reconfigure_server", "auv_control_node"
            )
            target_server = controller_server
            self.reconfigure_client = ReconfigureClient(self, target_server, timeout=5)
            self.get_logger().info(
                f"Connected to dynamic reconfigure server: {target_server}"
            )

            # Setup sync_cmd_pose service to reset cmd_pose on shutdown
            self.sync_cmd_pose_service = service_client(self, Trigger, "sync_cmd_pose")

            # Read and store original PID gains
            # (the ROS 2 controller exposes kp/ki/kd as 12-element arrays)
            current_cfg = self.reconfigure_client.get_configuration()
            if current_cfg:
                self.original_pid_gains = {
                    "kp": list(current_cfg.get("kp", [0.0] * 12)),
                    "ki": list(current_cfg.get("ki", [0.0] * 12)),
                    "kd": list(current_cfg.get("kd", [0.0] * 12)),
                }
                self.get_logger().info(
                    f"Stored original PID gains for X/Y: "
                    f"kp=[{self.original_pid_gains['kp'][0]}, {self.original_pid_gains['kp'][1]}], "
                    f"ki=[{self.original_pid_gains['ki'][0]}, {self.original_pid_gains['ki'][1]}], "
                    f"kd=[{self.original_pid_gains['kd'][0]}, {self.original_pid_gains['kd'][1]}]"
                )

                # Zero the X and Y position PID gains (kp, ki, kd)
                self.reconfigure_client.update_configuration(
                    {
                        name: [0.0, 0.0] + gains[2:]
                        for name, gains in self.original_pid_gains.items()
                    }
                )
                self.get_logger().info(
                    "use_vel mode: X and Y position PID gains (kp, ki, kd) set to 0"
                )
            else:
                self.get_logger().warning("Failed to read controller configuration")
                self.use_vel = False
        except Exception as e:
            self.get_logger().warning(f"Failed to setup use_vel mode: {e}")
            self.use_vel = False

    def _restore_pid_gains(self):
        """Restore original PID gains on shutdown and reset cmd_pose to current position."""
        if not self.reconfigure_client or not self.original_pid_gains:
            return

        # First, sync cmd_pose to current robot position
        # This prevents the robot from jumping to old cmd_pose when PID gains are restored
        try:
            self.sync_cmd_pose_service.wait_for_service(timeout=1.0)
            self.sync_cmd_pose_service(Trigger.Request())
            self.get_logger().info("Synced cmd_pose to current position")
        except (Exception, ROSException) as e:
            self.get_logger().warning(f"Failed to sync cmd_pose: {e}")

        # Then restore original PID gains
        try:
            self.reconfigure_client.update_configuration(self.original_pid_gains)
            self.get_logger().info(
                f"Restored original PID gains: kp_0={self.original_pid_gains['kp'][0]}, "
                f"kp_1={self.original_pid_gains['kp'][1]}"
            )
        except Exception as e:
            self.get_logger().warning(f"Failed to restore PID gains: {e}")

    def call_service_if_available(self, service, success_message, failure_message):
        try:
            service.wait_for_service(timeout=1)
            response = service(Trigger.Request())
            if response.success:
                self.get_logger().info(success_message)
            else:
                self.get_logger().warning(failure_message)
        except ROSException:
            self.get_logger().warning(
                f"Service {service.resolved_name} is not available"
            )

    def launch_torpedo1(self):
        self.call_service_if_available(
            self.torpedo1_service, "Torpedo 1 launched", "Failed to launch torpedo 1"
        )

    def launch_torpedo2(self):
        self.call_service_if_available(
            self.torpedo2_service, "Torpedo 2 launched", "Failed to launch torpedo 2"
        )

    def drop_dropper(self):
        self.call_service_if_available(
            self.dropper_service, "Ball dropped", "Failed to drop the ball"
        )

    def reset_orientation(self):
        self.call_service_if_available(
            self.reset_orientation_service,
            "Roll and pitch reset to zero",
            "Failed to reset orientation",
        )

    def joy_callback(self, msg):
        with self.lock:
            self.joy_data = msg

            if self.rov_mode == 1:
                # ROV modunda orientation reset butonu dinle
                self.reset_orientation_button_event.update(
                    self.joy_data.buttons[self.buttons["reset_orientation"]]
                )
            else:
                self.torpedo1_button_event.update(
                    self.joy_data.buttons[self.buttons["launch_torpedo1"]]
                )
                self.torpedo2_button_event.update(
                    self.joy_data.buttons[self.buttons["launch_torpedo2"]]
                )
                self.dropper_button_event.update(
                    self.joy_data.buttons[self.buttons["drop_ball"]]
                )

    def get_axis_value(self, indices):
        if isinstance(indices, list):
            return sum(self.joy_data.axes[i] for i in indices)
        return self.joy_data.axes[indices]

    def get_z_axis_value(self):
        # Check if using Xbox controller configuration (with +/- z buttons)
        if "z_axis_pos" in self.buttons and "z_axis_neg" in self.buttons:
            pos_value = (
                self.joy_data.buttons[self.buttons["z_axis_pos"]["index"]]
                * self.buttons["z_axis_pos"]["gain"]
            )
            neg_value = (
                self.joy_data.buttons[self.buttons["z_axis_neg"]["index"]]
                * self.buttons["z_axis_neg"]["gain"]
            )
            return pos_value - neg_value
        # Check if using Joy controller configuration (with z_control button)
        elif (
            "z_control" in self.buttons
            and self.joy_data.buttons[self.buttons["z_control"]]
        ):
            return (
                self.get_axis_value(self.axes["z_axis"]["index"])
                * self.axes["z_axis"]["gain"]
            )
        return 0.0

    def run(self):
        while rclpy.ok() and not getattr(self, "shutdown_requested", False):
            twist = Twist()

            with self.lock:
                if self.joy_data:
                    if self.rov_mode == 1:
                        # ROV mode: use buttons for pitch and roll control
                        pitch_gain = self.axes.get("rov_pitch_gain", 0.5)
                        roll_gain = self.axes.get("rov_roll_gain", 0.5)

                        # Pitch control: button 0 (positive), button 3 (negative)
                        pitch_pos = self.joy_data.buttons[self.buttons["rov_pitch_pos"]]
                        pitch_neg = self.joy_data.buttons[self.buttons["rov_pitch_neg"]]
                        twist.angular.y = (pitch_pos - pitch_neg) * pitch_gain

                        # Roll control: button 1 (positive), button 2 (negative)
                        roll_pos = self.joy_data.buttons[self.buttons["rov_roll_pos"]]
                        roll_neg = self.joy_data.buttons[self.buttons["rov_roll_neg"]]
                        twist.angular.x = (roll_pos - roll_neg) * roll_gain

                        # Keep normal z-axis control
                        twist.linear.z = self.get_z_axis_value()

                        # Keep normal x, y, yaw controls
                        twist.linear.x = (
                            self.get_axis_value(self.axes["x_axis"]["index"])
                            * self.axes["x_axis"]["gain"]
                        )
                        twist.linear.y = (
                            self.get_axis_value(self.axes["y_axis"]["index"])
                            * self.axes["y_axis"]["gain"]
                        )
                        twist.angular.z = (
                            self.get_axis_value(self.axes["yaw_axis"]["index"])
                            * self.axes["yaw_axis"]["gain"]
                        )
                    else:
                        # Normal AUV mode
                        # Get z-axis value based on controller type
                        twist.linear.z = self.get_z_axis_value()

                        # Set x-axis value
                        if (
                            "z_control" in self.buttons
                            and self.joy_data.buttons[self.buttons["z_control"]]
                        ):
                            twist.linear.x = 0.0
                        else:
                            twist.linear.x = (
                                self.get_axis_value(self.axes["x_axis"]["index"])
                                * self.axes["x_axis"]["gain"]
                            )

                        twist.linear.y = (
                            self.get_axis_value(self.axes["y_axis"]["index"])
                            * self.axes["y_axis"]["gain"]
                        )
                        twist.angular.z = (
                            self.get_axis_value(self.axes["yaw_axis"]["index"])
                            * self.axes["yaw_axis"]["gain"]
                        )
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

            self.enable_pub.publish(Bool(data=True))
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()


def main():
    ros.init()
    node = JoystickNode()
    ros.spin(node, node.run)


if __name__ == "__main__":
    main()
