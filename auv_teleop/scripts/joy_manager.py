#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import threading
from std_srvs.srv import Trigger, TriggerRequest
import dynamic_reconfigure.client


class JoystickEvent:
    def __init__(self, change_threshold, callback):
        self.previous_value = 0.0
        self.change_threshold = change_threshold
        self.callback = callback

    def update(self, value):
        if (value - self.previous_value) > self.change_threshold:
            self.callback()
        self.previous_value = value


class JoystickNode:
    def __init__(self):
        rospy.init_node("joystick_node", anonymous=True)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=10)

        self.joy_data = None
        self.lock = threading.Lock()

        self.publish_rate = 50  # 50 Hz
        self.rate = rospy.Rate(self.publish_rate)

        self.buttons = rospy.get_param("~buttons")
        self.axes = rospy.get_param("~axes")
        self.rov_mode = rospy.get_param("~rov_mode", 0)

        if self.rov_mode == 1:
            rospy.loginfo("ROV mode enabled - using buttons for pitch/roll control")
            self.reset_orientation_button_event = JoystickEvent(
                0.1, self.reset_orientation
            )
            self.reset_orientation_service = rospy.ServiceProxy(
                "reset_command_orientation", Trigger
            )
        else:
            self.torpedo1_button_event = JoystickEvent(0.1, self.launch_torpedo1)
            self.torpedo2_button_event = JoystickEvent(0.1, self.launch_torpedo2)
            self.dropper_button_event = JoystickEvent(0.1, self.drop_dropper)

            self.dropper_service = rospy.ServiceProxy("ball_dropper/drop", Trigger)

            self.torpedo1_service = rospy.ServiceProxy("torpedo_1/launch", Trigger)

            self.torpedo2_service = rospy.ServiceProxy("torpedo_2/launch", Trigger)

        self.joy_sub = rospy.Subscriber("joy", Joy, self.joy_callback)
        rospy.loginfo("Joystick node initialized")

        # use_vel mode: zero position PID gains for X and Y
        self.use_vel = rospy.get_param("~use_vel", False)
        self.original_pid_gains = None
        self.reconfigure_client = None

        if self.use_vel:
            self._setup_use_vel_mode()
            rospy.on_shutdown(self._restore_pid_gains)

    def _setup_use_vel_mode(self):
        """Setup use_vel mode by connecting to controller reconfigure and zeroing X/Y position gains."""
        try:
            controller_server = rospy.get_param(
                "~controller_reconfigure_server", "auv_control_node"
            )
            target_server = rospy.resolve_name(controller_server)
            self.reconfigure_client = dynamic_reconfigure.client.Client(
                target_server, timeout=5
            )
            rospy.loginfo(f"Connected to dynamic reconfigure server: {target_server}")

            # Setup sync_cmd_pose service to reset cmd_pose on shutdown
            self.sync_cmd_pose_service = rospy.ServiceProxy("sync_cmd_pose", Trigger)

            # Read and store original PID gains
            current_cfg = self.reconfigure_client.get_configuration()
            if current_cfg:
                self.original_pid_gains = {
                    "kp_0": current_cfg.get("kp_0", 0.0),
                    "kp_1": current_cfg.get("kp_1", 0.0),
                    "ki_0": current_cfg.get("ki_0", 0.0),
                    "ki_1": current_cfg.get("ki_1", 0.0),
                    "kd_0": current_cfg.get("kd_0", 0.0),
                    "kd_1": current_cfg.get("kd_1", 0.0),
                }
                rospy.loginfo(
                    f"Stored original PID gains for X/Y: "
                    f"kp=[{self.original_pid_gains['kp_0']}, {self.original_pid_gains['kp_1']}], "
                    f"ki=[{self.original_pid_gains['ki_0']}, {self.original_pid_gains['ki_1']}], "
                    f"kd=[{self.original_pid_gains['kd_0']}, {self.original_pid_gains['kd_1']}]"
                )

                # Zero the X and Y position PID gains (kp, ki, kd)
                self.reconfigure_client.update_configuration(
                    {
                        "kp_0": 0.0,
                        "kp_1": 0.0,
                        "ki_0": 0.0,
                        "ki_1": 0.0,
                        "kd_0": 0.0,
                        "kd_1": 0.0,
                    }
                )
                rospy.loginfo(
                    "use_vel mode: X and Y position PID gains (kp, ki, kd) set to 0"
                )
            else:
                rospy.logwarn("Failed to read controller configuration")
                self.use_vel = False
        except Exception as e:
            rospy.logwarn(f"Failed to setup use_vel mode: {e}")
            self.use_vel = False

    def _restore_pid_gains(self):
        """Restore original PID gains on shutdown and reset cmd_pose to current position."""
        if not self.reconfigure_client or not self.original_pid_gains:
            return

        # First, sync cmd_pose to current robot position
        # This prevents the robot from jumping to old cmd_pose when PID gains are restored
        try:
            self.sync_cmd_pose_service.wait_for_service(timeout=1.0)
            self.sync_cmd_pose_service(TriggerRequest())
            rospy.loginfo("Synced cmd_pose to current position")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"Failed to sync cmd_pose: {e}")

        # Then restore original PID gains
        try:
            self.reconfigure_client.update_configuration(self.original_pid_gains)
            rospy.loginfo(
                f"Restored original PID gains: kp_0={self.original_pid_gains['kp_0']}, "
                f"kp_1={self.original_pid_gains['kp_1']}"
            )
        except Exception as e:
            rospy.logwarn(f"Failed to restore PID gains: {e}")

    def call_service_if_available(self, service, success_message, failure_message):
        try:
            service.wait_for_service(timeout=1)
            response = service(TriggerRequest())
            if response.success:
                rospy.loginfo(success_message)
            else:
                rospy.logwarn(failure_message)
        except rospy.exceptions.ROSException:
            rospy.logwarn(f"Service {service.resolved_name} is not available")

    def launch_torpedo1(self):
        threading.Thread(
            target=lambda: self.call_service_if_available(
                self.torpedo1_service, "Torpedo 1 launched", "Failed to launch torpedo 1"
            ),
            daemon=True
        ).start()

    def launch_torpedo2(self):
        threading.Thread(
            target=lambda: self.call_service_if_available(
                self.torpedo2_service, "Torpedo 2 launched", "Failed to launch torpedo 2"
            ),
            daemon=True
        ).start()

    def drop_dropper(self):
        threading.Thread(
            target=lambda: self.call_service_if_available(
                self.dropper_service, "Ball dropped", "Failed to drop the ball"
            ),
            daemon=True
        ).start()

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
        while not rospy.is_shutdown():
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

            self.enable_pub.publish(Bool(True))
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        joystick_node = JoystickNode()
        joystick_node.run()
    except rospy.ROSInterruptException:
        pass
