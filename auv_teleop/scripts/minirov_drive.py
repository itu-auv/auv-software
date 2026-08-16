#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, UInt16MultiArray


class MiniRovDrive:
    def __init__(self):
        rospy.init_node("minirov_drive")

        self.min_pwm = rospy.get_param("~min_pwm", 1200)
        self.neutral_pwm = rospy.get_param("~neutral_pwm", 1500)
        self.max_pwm = rospy.get_param("~max_pwm", 1800)
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)
        self.ramp_step = rospy.get_param("~ramp_step", 15)
        self.publish_topic = rospy.get_param("~publish_topic")
        self.motor_count = rospy.get_param("~motor_count", 4)
        self.controls = rospy.get_param("~controls")
        self.lumen_topic = rospy.get_param(
            "~lumen_topic", "/minirov/lumen_brightness"
        )
        self.lumen_on_button = rospy.get_param("~lumen_on_button", 3)
        self.lumen_off_button = rospy.get_param("~lumen_off_button", 0)

        neutral = [self.neutral_pwm] * self.motor_count
        self.current_pwm = list(neutral)
        self.target_pwm = list(neutral)
        self.ramping = [False] * self.motor_count
        self.ramp_motors = {
            config["motor"]
            for config in self.controls.values()
            if config.get("ramp", False)
        }
        self.lumen_on_pressed = False
        self.lumen_off_pressed = False
        self.trigger_ready = {
            name: False for name, config in self.controls.items() if "trigger" in config
        }

        self.publisher = rospy.Publisher(
            self.publish_topic, UInt16MultiArray, queue_size=10
        )
        self.lumen_publisher = rospy.Publisher(
            self.lumen_topic, Bool, queue_size=10
        )
        rospy.Subscriber("joy", Joy, self.joy_callback, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish)

        rospy.loginfo(
            f"MiniRovDrive publishing {self.publish_topic} at "
            f"{self.publish_rate:g} Hz"
        )

    def publish(self, _event):
        for motor in range(self.motor_count):
            if self.ramping[motor]:
                self.current_pwm[motor] = self.move_towards(
                    self.current_pwm[motor], self.target_pwm[motor], self.ramp_step
                )
            else:
                self.current_pwm[motor] = self.target_pwm[motor]

        self.publisher.publish(UInt16MultiArray(data=self.current_pwm))

    @staticmethod
    def move_towards(current, target, step):
        if current < target:
            return min(current + step, target)
        return max(current - step, target)

    @staticmethod
    def axis_value(indices, joy):
        if isinstance(indices, list):
            return sum(joy.axes[index] for index in indices)
        return joy.axes[indices]

    def trigger_value(self, name, index, joy):
        raw = max(-1.0, min(1.0, joy.axes[index]))
        if not self.trigger_ready[name]:
            if abs(raw) < 0.5:
                return 0.0
            self.trigger_ready[name] = True
        return (1.0 - raw) / 2.0

    def control_value(self, name, config, joy):
        if "axis" in config:
            value = self.axis_value(config["axis"], joy)
        elif "trigger" in config:
            value = self.trigger_value(name, config["trigger"], joy)
        else:
            value = float(joy.buttons[config["button"]])

        value = max(-1.0, min(1.0, value))
        return -value if config.get("inverse", False) else value

    def value_to_pwm(self, value):
        if value >= 0:
            return round(self.neutral_pwm + (self.max_pwm - self.neutral_pwm) * value)
        return round(self.neutral_pwm + (self.neutral_pwm - self.min_pwm) * value)

    def joy_callback(self, joy):
        targets = [self.neutral_pwm] * self.motor_count
        active_controls = [None] * self.motor_count

        for name, config in self.controls.items():
            value = self.control_value(name, config, joy)
            if value == 0:
                continue

            motor = config["motor"]
            targets[motor] = self.value_to_pwm(value)
            active_controls[motor] = config

        for motor, config in enumerate(active_controls):
            self.target_pwm[motor] = targets[motor]
            self.ramping[motor] = (
                config.get("ramp", False)
                if config is not None
                else motor in self.ramp_motors
                and self.current_pwm[motor] < self.neutral_pwm
            )

        lumen_on = bool(joy.buttons[self.lumen_on_button])
        lumen_off = bool(joy.buttons[self.lumen_off_button])

        if lumen_on and not self.lumen_on_pressed:
            self.lumen_publisher.publish(Bool(data=True))
        elif lumen_off and not self.lumen_off_pressed:
            self.lumen_publisher.publish(Bool(data=False))

        self.lumen_on_pressed = lumen_on
        self.lumen_off_pressed = lumen_off


if __name__ == "__main__":
    try:
        MiniRovDrive()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
