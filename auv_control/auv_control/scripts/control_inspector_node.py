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
        self.power_sub = rospy.Subscriber(
            "power", Power, self.battery_callback, tcp_nodelay=True
        )
        self.dvl_sub = rospy.Subscriber(
            "dvl_enabled", Bool, self.dvl_callback, tcp_nodelay=True
        )
        self.inspector_enable_sub = rospy.Subscriber(
            "enable", Bool, self.handle_enable, tcp_nodelay=True
        )

        # Publishers
        self.control_enable_pub = rospy.Publisher(
            "control_inspector_enable", Bool, queue_size=1, tcp_nodelay=True
        )

        # Initialize variables
        self.last_odometry_time = None
        self.last_altitude_time = None
        self.altitude = None
        self.battery_voltage = None
        self.last_dvl_time = None
        self.enable = False

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 20)
        self.min_battery_voltage = rospy.get_param("~min_battery_voltage", 13.0)
        self.odometry_timeout = rospy.get_param("~odometry_timeout", 0.5)
        self.altitude_timeout = rospy.get_param("~altitude_timeout", 0.5)
        self.dvl_timeout = rospy.get_param("~dvl_timeout", 1.0)
        self.min_altitude = rospy.get_param("~min_altitude", 0.3)
        self.pool_depth = rospy.get_param("~pool_depth", 2.2)
        self.vertical_offset = rospy.get_param("~vertical_offset", 0.3)

    def odometry_callback(self, msg):
        time = rospy.get_time()
        self.last_odometry_time = time

        if msg.pose.pose.position.z:
            self.altitude = (
                self.pool_depth + msg.pose.pose.position.z - self.vertical_offset
            )
            self.last_altitude_time = time
        else:
            self.last_altitude_time = None
            rospy.logwarn("No altitude data in odometry message")

    def battery_callback(self, msg):
        self.battery_voltage = msg.voltage

    def dvl_callback(self, msg):
        if msg.data:
            self.last_dvl_time = rospy.get_time()

    def handle_enable(self, msg):
        self.enable = msg.data

    def safety_checks_passed(self):
        current_time = rospy.get_time()
        errors = []

        # Check odometry status
        if self.last_odometry_time is None:
            errors.append("No odometry data received")
        elif current_time - self.last_odometry_time > self.odometry_timeout:
            errors.append(
                f"Odometry timeout (last update: {current_time - self.last_odometry_time:.1f}s ago)"
            )

        # Check altitude status
        if self.last_altitude_time is None:
            errors.append("No altitude data received")
        elif current_time - self.last_altitude_time > self.altitude_timeout:
            errors.append(
                f"Altitude timeout (last update: {current_time - self.last_altitude_time:.1f}s ago)"
            )

        # Check minimum altitude
        if self.altitude is None:
            errors.append("Altitude data not available")
        elif self.altitude < self.min_altitude:
            errors.append(
                f"Altitude below minimum ({self.altitude:.1f} < {self.min_altitude})"
            )

        # Check battery voltage
        if self.battery_voltage is None:
            errors.append("Battery voltage data not available")
        elif self.battery_voltage < self.min_battery_voltage:
            errors.append(
                f"Battery voltage low ({self.battery_voltage:.1f} < {self.min_battery_voltage})"
            )

        # Check DVL status
        if self.last_dvl_time is None:
            errors.append("DVL not enabled or no data received")
        elif current_time - self.last_dvl_time > self.dvl_timeout:
            errors.append(
                f"DVL timeout (last successful ping was: {current_time - self.last_dvl_time:.1f}s ago)"
            )

        return (len(errors) == 0, errors)

    def control_loop(self):
        if self.enable:
            success, errors = self.safety_checks_passed()
            if success:
                self.control_enable_pub.publish(Bool(data=True))
            else:
                error_msg = "; ".join(errors)
                rospy.logerr(
                    f"[ControlInspectorNode] Safety checks failed: {error_msg}"
                )
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
