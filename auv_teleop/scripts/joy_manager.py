#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import threading
from std_srvs.srv import SetBool, SetBoolRequest

class JoystickNode:
    def __init__(self):
        rospy.init_node('joystick_node', anonymous=True)
        
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.enable_pub = rospy.Publisher('enable', Bool, queue_size=10)

        self.joy_data = None
        self.lock = threading.Lock()
        self.torpedo_fired = False

        self.publish_rate = 50  # 50 Hz
        self.rate = rospy.Rate(self.publish_rate)

        rospy.wait_for_service('taluy/board/set_torpedo')
        self.set_torpedo_service = rospy.ServiceProxy('taluy/board/set_torpedo', SetBool)

        rospy.loginfo("Joystick node initialized")

    def joy_callback(self, msg):
        with self.lock:
            self.joy_data = msg

    def call_torpedo_service(self, data):
        try:
            req = SetBoolRequest(data=data)
            resp = self.set_torpedo_service(req)
            rospy.loginfo(f"Service response: {resp.success}, {resp.message}")
            if not resp.success or data:
                rospy.Timer(rospy.Duration(1), lambda event: self.call_torpedo_service(False), oneshot=True)
            self.torpedo_fired = False if data == False else self.torpedo_fired
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            self.torpedo_fired = False
            rospy.Timer(rospy.Duration(1), lambda event: self.call_torpedo_service(False), oneshot=True)

    def run(self):
        while not rospy.is_shutdown():
            twist = Twist()
            
            with self.lock:
                if self.joy_data:

                    # if button 1 pressed, control z axis
                    if self.joy_data.buttons[1]:
                        twist.linear.x = 0.0
                        twist.linear.z = self.joy_data.axes[1]
                    else:
                        twist.linear.x = self.joy_data.axes[1]
                        twist.linear.z = 0.0

                    # control y and angular z axis 
                    twist.linear.y = self.joy_data.axes[0]
                    twist.angular.z = self.joy_data.axes[2]

                    # Call the service when button 0 is pressed
                    if self.joy_data.buttons[0]:
                        if not self.torpedo_fired:
                            rospy.loginfo("Calling set_torpedo service with data True")
                            self.call_torpedo_service(True)
                            self.torpedo_fired = True

                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

            self.enable_pub.publish(Bool(True))
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        joystick_node = JoystickNode()
        joystick_node.run()
    except rospy.ROSInterruptException:
        pass
