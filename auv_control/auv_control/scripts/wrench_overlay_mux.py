#!/usr/bin/env python3

import threading

import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from mini_slalom_core import overlay_wrench


class WrenchOverlayMux:
    def __init__(self):
        rospy.init_node("wrench_overlay_mux")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.nominal_timeout = float(rospy.get_param("~nominal_timeout", 0.3))
        self.visual_timeout = float(rospy.get_param("~visual_timeout", 0.3))
        self.active_timeout = float(rospy.get_param("~active_timeout", 0.5))
        self.body_frame = rospy.get_param("~body_frame", "taluy_mini/base_link")
        self.lock = threading.Lock()
        self.nominal = None
        self.nominal_received = rospy.Time()
        self.visual = None
        self.visual_received = rospy.Time()
        self.active = False
        self.active_received = rospy.Time()

        self.output_pub = rospy.Publisher("wrench", WrenchStamped, queue_size=1)
        rospy.Subscriber(
            "wrench_nominal", WrenchStamped, self.nominal_callback, queue_size=1
        )
        rospy.Subscriber(
            "slalom/visual_wrench",
            WrenchStamped,
            self.visual_callback,
            queue_size=1,
        )
        rospy.Subscriber("slalom/active", Bool, self.active_callback, queue_size=1)

    def nominal_callback(self, msg):
        with self.lock:
            self.nominal = msg
            self.nominal_received = rospy.Time.now()

    def visual_callback(self, msg):
        with self.lock:
            self.visual = msg
            self.visual_received = rospy.Time.now()

    def active_callback(self, msg):
        with self.lock:
            self.active = msg.data
            self.active_received = rospy.Time.now()

    @staticmethod
    def fresh(now, received, timeout):
        return received != rospy.Time() and (now - received).to_sec() <= timeout

    def build_output(self, now):
        with self.lock:
            nominal = self.nominal
            visual = self.visual
            nominal_fresh = self.fresh(now, self.nominal_received, self.nominal_timeout)
            visual_fresh = self.fresh(now, self.visual_received, self.visual_timeout)
            active_fresh = self.fresh(now, self.active_received, self.active_timeout)
            active = self.active and active_fresh

        output = WrenchStamped()
        output.header.stamp = now
        output.header.frame_id = self.body_frame
        if not nominal_fresh:
            rospy.logwarn_throttle(
                2.0, "Mini slalom wrench mux: nominal wrench is stale"
            )
            return output

        nominal_values = (
            nominal.wrench.force.x,
            nominal.wrench.force.y,
            nominal.wrench.force.z,
            nominal.wrench.torque.x,
            nominal.wrench.torque.y,
            nominal.wrench.torque.z,
        )
        visual_values = (
            visual.wrench.force.x if visual is not None else 0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            visual.wrench.torque.z if visual is not None else 0.0,
        )
        values = overlay_wrench(nominal_values, visual_values, active, visual_fresh)
        output.wrench.force.x, output.wrench.force.y, output.wrench.force.z = values[:3]
        (
            output.wrench.torque.x,
            output.wrench.torque.y,
            output.wrench.torque.z,
        ) = values[3:]
        if active and not visual_fresh:
            rospy.logwarn_throttle(
                1.0,
                "Mini slalom wrench mux: visual command is stale, stopping planar motion",
            )
        return output

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.output_pub.publish(self.build_output(rospy.Time.now()))
            rate.sleep()


if __name__ == "__main__":
    try:
        WrenchOverlayMux().spin()
    except rospy.ROSInterruptException:
        pass
