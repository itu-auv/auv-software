#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import UInt16MultiArray


class MiniRovDrive:
    def __init__(self):
        rospy.init_node("minirov_drive", anonymous=True)

        self.max_pwm = rospy.get_param("~max_pwm", 1800)
        self.min_pwm = rospy.get_param("~min_pwm", 1200)
        self.default_pwm = rospy.get_param("~default_pwm", 1500)
        self.publish_topic = rospy.get_param("~publish_topic", "/minirov/drive_pulse")

        self.motors = rospy.get_param("~motors")
        self.toggle_states = {}
        self.last_pressed = {}
        for motor in self.motors:
            self.toggle_states[motor["id"]] = False

        self.pub = rospy.Publisher(self.publish_topic, UInt16MultiArray, queue_size=10)
        self.sub = rospy.Subscriber("joy", Joy, self.joy_callback, queue_size=1)
        rospy.loginfo(f"MiniRovDrive started, publishing to {self.publish_topic}")

    def get_axis_value(self, axis_spec, joy_data):
        if isinstance(axis_spec, list):
            return sum(joy_data.axes[i] for i in axis_spec)
        return joy_data.axes[axis_spec]

    def compute_pwm(self, motor, value):
        if motor["type"] == "x":
            ratio = max(-1.0, min(1.0, value))
            return int(
                round(self.default_pwm + (self.max_pwm - self.default_pwm) * ratio)
            )
        if motor["type"] == "z":
            ratio = max(-1.0, min(1.0, value))
            if self.toggle_states.get(motor["id"]):
                low = self.min_pwm
            else:
                low = self.max_pwm
            return int(
                round(self.default_pwm + (self.default_pwm - low) * (ratio - 1) / 2.0)
            )
        return self.default_pwm

    def joy_callback(self, msg):
        pwm_values = {}

        for motor in self.motors:
            mid = motor["id"]
            if "toggle_button" in motor:
                pressed = msg.buttons[motor["toggle_button"]] == 1
                prev = self.last_pressed.get(mid, False)
                if pressed and not prev:
                    self.toggle_states[mid] = not self.toggle_states[mid]
                    direction = "FORWARD" if not self.toggle_states[mid] else "REVERSE"
                    rospy.loginfo(f"[{motor['name']}] direction toggled -> {direction}")
                self.last_pressed[mid] = pressed
            pwm_values[mid] = self.compute_pwm(
                motor, self.get_axis_value(motor["axis"], msg)
            )

        ordered = [pwm_values[mid] for mid in sorted(pwm_values)]
        self.pub.publish(UInt16MultiArray(data=ordered))

        if getattr(self, "last_pwm", None) != ordered:
            self.last_pwm = ordered
            rospy.loginfo(f"drive_pulse: {ordered}")


if __name__ == "__main__":
    try:
        node = MiniRovDrive()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
