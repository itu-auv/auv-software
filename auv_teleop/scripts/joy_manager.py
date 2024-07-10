#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from std_msgs.msg import Bool
import threading

class JoystickNode:
    def __init__(self):
        rospy.init_node('joystick_node', anonymous=True)
        
        self.joy_sub = rospy.Subscriber('/taluy/joy', Joy, self.joy_callback)
        self.cmd_vel_pub = rospy.Publisher('/taluy/cmd_vel', Twist, queue_size=10)
        self.enable_pub = rospy.Publisher('/taluy/enable', Bool, queue_size=10)
        
        self.joy_data = None
        self.lock = threading.Lock()
        
        self.publish_rate = 50  # 50 Hz
        self.rate = rospy.Rate(self.publish_rate)

        rospy.loginfo("Joystick node initialized")

    def joy_callback(self, msg):
        with self.lock:
            self.joy_data = msg

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
