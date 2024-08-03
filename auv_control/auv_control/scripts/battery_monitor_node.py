#!/usr/bin/env python

import rospy
from auv_msgs.msg import Power

class PowerMonitorNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('power_monitor', anonymous=True)

        # Parameters
        self.voltage_threshold = rospy.get_param('~voltage_threshold', 13.0)
        self.voltage_warn_threshold = rospy.get_param('~voltage_warn_threshold', 15.0)
        self.timeout_duration = rospy.get_param('~timeout_duration', 5)  # Timeout duration in seconds
        self.min_print_interval = rospy.get_param('~min_print_interval', 1.0)  # Minimum print interval in seconds
        self.max_print_interval = rospy.get_param('~max_print_interval', 10.0)  # Maximum print interval in seconds

        # Variables
        self.last_msg_time = rospy.Time.now()
        self.voltage = None
        self.timeouted = False
        self.undervoltage = False
        self.print_interval = 1.0  # Start with a slow print interval (e.g., 1 seconds)

        # Subscribers
        self.power_sub = rospy.Subscriber('power', Power, self.power_callback)

        # Timers
        self.voltage_print_timer = rospy.Timer(rospy.Duration(self.print_interval), self.print_voltage_callback, oneshot=True)
        self.timeout_check_timer = rospy.Timer(rospy.Duration(0.1), self.check_timeout_and_low_voltage_callback)

        rospy.loginfo("Power Monitor Node started with voltage threshold: {:.2f}".format(self.voltage_threshold))

    def power_callback(self, msg):
        self.voltage = msg.voltage
        self.undervoltage = self.voltage < self.voltage_threshold
        self.last_msg_time = rospy.Time.now()

    def print_voltage(self):
        green_text = "\033[92m"  # ANSI escape code for green text
        orange_text = "\033[93m" # ANSI escape code for orange text
        reset_text = "\033[0m"   # ANSI escape code to reset text color
        if self.voltage < self.voltage_warn_threshold:
            rospy.loginfo("Battery voltage:" +  orange_text + " {:.2f} V".format(self.voltage) + reset_text)
        else:
            rospy.loginfo("Battery voltage:" + green_text + " {:.2f} V".format(self.voltage) + reset_text)
            
        self.print_interval = self.interpolate_print_interval(self.voltage)
        self.voltage_print_timer.shutdown()
        self.voltage_print_timer = rospy.Timer(rospy.Duration(self.print_interval), self.print_voltage_callback, oneshot=True)

    def print_voltage_callback(self, event):
        if not self.timeouted and self.voltage != None:
            self.print_voltage()

    def check_timeout_and_low_voltage_callback(self, event):
        new_timeout = rospy.Time.now() - self.last_msg_time > rospy.Duration(self.timeout_duration)
        if new_timeout != self.timeouted and self.voltage != None and not self.undervoltage:
            self.print_voltage()  
        
        self.timeouted = new_timeout
            
        if self.timeouted:
            if self.voltage != None:
                rospy.logerr("Power message timeout. Last voltage: {:.2f} V".format(self.voltage))
            else:
                rospy.logerr("Power message timeout. No voltage received.")
            return
        
        if self.voltage != None:
            self.undervoltage = self.voltage < self.voltage_threshold
            if self.undervoltage:
                rospy.logerr("Battery voltage is below {:.2f}V threshold: {:.2f} V".format(self.voltage_threshold, self.voltage))

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
            return (voltage - min_voltage) / (max_voltage - min_voltage) * (max_interval - min_interval)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PowerMonitorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
