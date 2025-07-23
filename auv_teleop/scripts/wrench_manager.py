#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Wrench
from std_msgs.msg import Bool
import threading


class JoystickNode:
    def __init__(self):
        rospy.init_node("joystick_node", anonymous=True)

        self.cmd_vel_pub = rospy.Publisher("wrench_cmd", Wrench, queue_size=10)
        # self.enable_pub = rospy.Publisher("enable", Bool, queue_size=10)

        self.joy_data = None
        self.lock = threading.Lock()

        self.publish_rate = 50  # 50 Hz
        self.rate = rospy.Rate(self.publish_rate)

        self.buttons = rospy.get_param("~buttons")
        self.axes = rospy.get_param("~axes")

        self.joy_sub = rospy.Subscriber("joy", Joy, self.joy_callback)
        rospy.loginfo("Joystick node initialized")

    def joy_callback(self, msg):
        with self.lock:
            self.joy_data = msg

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
            wrench = Wrench()

            with self.lock:
                if self.joy_data:
                    # Get z-axis value based on controller type
                    wrench.force.z = self.get_z_axis_value()

                    # Set x-axis value
                    if (
                        "z_control" in self.buttons
                        and self.joy_data.buttons[self.buttons["z_control"]]
                    ):
                        wrench.force.x = 0.0
                    else:
                        wrench.force.x = (
                            self.get_axis_value(self.axes["x_axis"]["index"])
                            * self.axes["x_axis"]["gain"]
                        )

                    wrench.force.y = (
                        self.get_axis_value(self.axes["y_axis"]["index"])
                        * self.axes["y_axis"]["gain"]
                    )
                    wrench.torque.z = (
                        self.get_axis_value(self.axes["yaw_axis"]["index"])
                        * self.axes["yaw_axis"]["gain"]
                    )
                else:
                    wrench.force.x = 0.0
                    wrench.torque.z = 0.0

            # self.enable_pub.publish(Bool(True))
            self.cmd_vel_pub.publish(wrench)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        joystick_node = JoystickNode()
        joystick_node.run()
    except rospy.ROSInterruptException:
        pass
