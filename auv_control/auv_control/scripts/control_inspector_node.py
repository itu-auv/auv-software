#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import Float32, Bool
from auv_msgs.msg._Power import Power


class ControlInspectorNode:
    def __init__(self):
        # Subscribers
        self.odometry_sub = rospy.Subscriber(
            "odometry", Odometry, self.odometry_callback, tcp_nodelay=True
        )
        self.altitude_sub = rospy.Subscriber(
            "altitude", Float32, self.altitude_callback, tcp_nodelay=True
        )
        self.power_sub = rospy.Subscriber(
            "power", Power, self.battery_callback, tcp_nodelay=True
        )
        self.dvl_sub = rospy.Subscriber(
            "dvl_enabled", Bool, self.dvl_callback, tcp_nodelay=True
        )

        # Publishers
        self.control_enable_pub = rospy.Publisher(
            "control_enable", Bool, queue_size=1, latch=True, tcp_nodelay=True
        )
        self.control_inspector_enabled_pub = rospy.Publisher(
            "control_inspector_enabled",
            Bool,
            queue_size=1,
            latch=True,
            tcp_nodelay=True,
        )

        # Service
        self.enable_service = rospy.Service(
            "set_control_enable", SetBool, self.handle_control_enable_request
        )

        # Initialize variables
        self.last_odometry_time = None
        self.last_altitude_time = None
        self.altitude = None
        self.battery_voltage = None
        self.last_dvl_time = None
        self.last_cmd = None
        self.control_inspector_enabled = False

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 20)
        self.command_timeout = rospy.get_param("~command_timeout", 0.1)
        self.min_battery_voltage = rospy.get_param("~min_battery_voltage", 13.0)
        self.odometry_timeout = rospy.get_param("~odometry_timeout", 0.5)
        self.altitude_timeout = rospy.get_param("~altitude_timeout", 0.5)
        self.dvl_timeout = rospy.get_param("~dvl_timeout", 2.0)
        self.min_altitude = rospy.get_param("~min_altitude", 30.0)

    def odometry_callback(self, msg):
        self.last_odometry_time = rospy.get_time()

    def altitude_callback(self, msg):
        self.last_altitude_time = rospy.get_time()
        self.altitude = msg.data

    def battery_callback(self, msg):
        self.battery_voltage = msg.data.voltage

    def dvl_callback(self, msg):
        if msg.data:
            self.last_dvl_time = rospy.get_time()

    def cmd_callback(self, msg):
        self.last_cmd = msg

    def handle_control_enable_request(self, req):
        if req.data:
            if self.safety_checks_passed():
                self.control_inspector_enabled = True
                return SetBoolResponse(True, "Control inspector enabled")
            else:
                return SetBoolResponse(
                    False, "Control inspector denied: Safety checks failed"
                )
        else:
            self.control_inspector_enabled = False
            return SetBoolResponse(True, "Control inspector disabled")

    def safety_checks_passed(self):
        current_time = rospy.get_time()
        odometry_ok = (self.last_odometry_time is not None) and (
            current_time - self.last_odometry_time <= self.odometry_timeout
        )
        altitude_ok = (self.last_altitude_time is not None) and (
            current_time - self.last_altitude_time <= self.altitude_timeout
        )
        min_altitude_ok = (self.altitude is not None) and (
            self.altitude >= self.min_altitude
        )
        battery_ok = (self.battery_voltage is not None) and (
            self.battery_voltage >= self.min_battery_voltage
        )
        dvl_ok = (self.last_dvl_time is not None) and (
            current_time - self.last_dvl_time <= self.dvl_timeout
        )

        return odometry_ok and altitude_ok and min_altitude_ok and battery_ok and dvl_ok

    def control_loop(self):
        self.control_inspector_enabled_pub.publish(
            Bool(data=self.control_inspector_enabled)
        )

        if self.control_inspector_enabled:
            if self.safety_checks_passed():
                self.control_enable_pub.publish(Bool(data=True))
            else:
                rospy.logerr("Safety checks failed! Disabling control.")
                self.control_inspector_enabled = False
                self.control_enable_pub.publish(Bool(data=False))
        else:
            self.control_enable_pub.publish(Bool(data=False))

    def run(self):
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("control_inspector_node")
        control_inspector_node = ControlInspectorNode()
        control_inspector_node.run()
    except rospy.ROSInterruptException:
        pass
