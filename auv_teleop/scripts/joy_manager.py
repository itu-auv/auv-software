#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from std_msgs.msg import Bool
import threading
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

class JoystickNode:
    def __init__(self):
        rospy.init_node('joystick_node', anonymous=True)
        
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.enable_pub = rospy.Publisher('enable', Bool, queue_size=10)

        self.joy_data = None
        self.lock = threading.Lock()

        self.publish_rate = 50  # 50 Hz
        self.rate = rospy.Rate(self.publish_rate)
        
        self.service_client = rospy.ServiceProxy('/taluy/board/set_torpedo', SetBool)
        self.service_client.wait_for_service()
        
        rospy.loginfo("Joystick node initialized")

    def joy_callback(self, msg):
        with self.lock:
            self.joy_data = msg
            
            # Check if the specific button is pressed
            if msg.buttons[0]:  # using button 0 for this function
                self.set_torpedo()

    def set_torpedo(self):
        try:
            # Call the service with data=1
            request = SetBoolRequest(data=True)
            self.service_client(request)
            
            # Schedule the reset to data=0 after a short delay
            rospy.Timer(rospy.Duration(0.1), self.reset_torpedo, oneshot=True)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def reset_torpedo(self, event):
        try:
            # Call the service with data=0
            request = SetBoolRequest(data=False)
            self.service_client(request)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

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
