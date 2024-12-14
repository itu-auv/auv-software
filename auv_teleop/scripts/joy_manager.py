#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from std_msgs.msg import Bool
import threading
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest


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

        self.torpedo1_button_event = JoystickEvent(0.1, self.launch_torpedo1)
        self.torpedo2_button_event = JoystickEvent(0.1, self.launch_torpedo2)
        self.droper_button_event = JoystickEvent(0.1, self.drop_droper)

        self.dropper_service = rospy.ServiceProxy(
            "actuators/ball_dropper/drop", Trigger
        )

        self.torpedo1_service = rospy.ServiceProxy(
            "actuators/torpedo_1/launch", Trigger
        )

        self.torpedo2_service = rospy.ServiceProxy(
            "actuators/torpedo_2/launch", Trigger
        )

        self.joy_sub = rospy.Subscriber("joy", Joy, self.joy_callback)
        rospy.loginfo("Joystick node initialized")

    def call_service_if_available(self, service):
        # check  if service is available
        try:
            service.wait_for_service(timeout=1)
        except rospy.exceptions.ROSException:
            rospy.logwarn("Dropper service is not available")
            return

        # call dropper service
        response = service(TriggerRequest())
        if response.success:
            rospy.loginfo("Dropper dropped")
        else:
            rospy.loginfo("Dropper failed to drop")

    def launch_torpedo1(self):
        self.call_service_if_available(self.torpedo1_service)
        rospy.loginfo("Launching torpedo 1")

    def launch_torpedo2(self):
        self.call_service_if_available(self.torpedo2_service)
        rospy.loginfo("Launching torpedo 2")

    def drop_droper(self):
        rospy.loginfo("Dropping dropper")
        self.call_service_if_available(self.dropper_service)

    def joy_callback(self, msg):
        with self.lock:
            self.joy_data = msg

            self.torpedo1_button_event.update(self.joy_data.buttons[4])
            self.torpedo2_button_event.update(self.joy_data.buttons[2])
            self.droper_button_event.update(self.joy_data.buttons[0])

    def run(self):
        while not rospy.is_shutdown():
            twist = Twist()

            with self.lock:
                if self.joy_data:

                    # if button 1 pressed, control z axis
                    if self.joy_data.buttons[1]:
                        twist.linear.x = 0.0
                        twist.linear.z = self.joy_data.axes[1] * 0.4
                    else:
                        twist.linear.x = self.joy_data.axes[1] * 0.4
                        twist.linear.z = 0.0

                    # control y and angular z axis
                    twist.linear.y = (
                        self.joy_data.axes[0] * 0.4
                    )
                    twist.angular.z = self.joy_data.axes[2] * 0.5

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
