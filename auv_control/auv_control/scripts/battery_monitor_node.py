#!/usr/bin/env python3

import rospy
from auv_msgs.msg import Power

class BatteryMonitorNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('battery_monitor_node', anonymous=True)

        # Parameters
        self.voltage_threshold = rospy.get_param('~minimum_voltage_threshold', 13.0)
        self.voltage_warn_threshold = rospy.get_param('~voltage_warn_threshold', 15.0)
        self.timeout_duration = rospy.get_param('~power_message_timeout', 5)  # Timeout duration in seconds
        self.min_print_interval = rospy.get_param('~min_log_interval', 1.0)  # Minimum print interval in seconds
        self.max_print_interval = rospy.get_param('~max_log_interval', 10.0)  # Maximum print interval in seconds

        # Variables
        self.last_msg_time = rospy.Time.now()
        self.voltage = None
        self.is_timeouted = True
        self.is_undervoltage = False
        self.print_interval = 1.0  # Start with a slow print interval (e.g., 1 seconds)

        # Subscribers
        self.power_sub = rospy.Subscriber('power', Power, self.power_callback)

        # Timers
        self.voltage_print_timer = rospy.Timer(rospy.Duration(self.print_interval), self.print_voltage_callback, oneshot=True)
        # self.timeout_check_timer = rospy.Timer(rospy.Duration(0.1), self.check_timeout_and_low_voltage_callback)

        rospy.loginfo("Power Monitor Node started with voltage threshold: {:.2f}".format(self.voltage_threshold))

    def power_callback(self, msg):
        self.voltage = msg.voltage
        self.is_undervoltage = self.voltage < self.voltage_threshold
        self.last_msg_time = rospy.Time.now()

    def reset_timer_with_new_interval(self, new_interval):
        self.print_interval = new_interval
        self.voltage_print_timer.shutdown()
        self.voltage_print_timer = rospy.Timer(rospy.Duration(self.print_interval), self.print_voltage_callback, oneshot=True)

    def print_voltage(self):

        if self.voltage == None:
            self.reset_timer_with_new_interval(1.0)
            rospy.logwarn("No voltage received.")
            return

        self.is_timeouted = rospy.Time.now() - self.last_msg_time > rospy.Duration(self.timeout_duration)
        if self.is_timeouted:
            self.reset_timer_with_new_interval(1.0)
            rospy.logerr("Power message timeout. Last voltage: {:.2f} V".format(self.voltage))
            return

        if self.voltage < self.voltage_threshold:
            self.reset_timer_with_new_interval(1.0)
            rospy.logerr("Battery voltage is below {:.2f}V threshold: {:.2f} V".format(self.voltage_threshold, self.voltage))
            return
        elif self.voltage < self.voltage_warn_threshold:
            self.reset_timer_with_new_interval(1.0)
            rospy.loginfo("Battery voltage:" +  "\033[93m" + " {:.2f} V".format(self.voltage) + "\033[0m")
            return
        elif self.voltage >= self.voltage_warn_threshold:
            new_interval = self.interpolate_print_interval(self.voltage)
            self.reset_timer_with_new_interval(new_interval)
            rospy.loginfo("Battery voltage:" + "\033[92m" + " {:.2f} V".format(self.voltage) + "\033[0m")
            return

        # if still alive, set the timer for the next print
        self.print_interval = 1.0
        self.voltage_print_timer.shutdown()
        self.voltage_print_timer = rospy.Timer(rospy.Duration(self.print_interval), self.print_voltage_callback, oneshot=True)

    def print_voltage_callback(self, event):
        self.print_voltage()

    def check_timeout_and_low_voltage_callback(self, event):
        new_timeout = rospy.Time.now() - self.last_msg_time > rospy.Duration(self.timeout_duration)
        self.is_undervoltage = self.voltage < self.voltage_threshold

        if new_timeout != self.is_timeouted and self.voltage != None:
            if not self.is_undervoltage:
                self.print_voltage()
            self.is_timeouted = new_timeout

        if self.is_timeouted:
            if self.voltage != None:
                rospy.logerr("Power message timeout. Last voltage: {:.2f} V".format(self.voltage))
            else:
                rospy.logerr("Power message timeout. No voltage received.")
            return

        if self.voltage != None and self.is_undervoltage:
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
        node = BatteryMonitorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
