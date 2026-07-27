#!/usr/bin/env python3
"""Fake battery publisher for testing battery_popup_node.

Publishes BatteryState messages that simulate voltage dropping below threshold.
Usage:
  rosrun auv_teleop test_battery_pub.py [--topic /taluy/mainboard/power_sensor/power]

Sequence: 5s normal (16V) → 5s low (12V) → repeat
"""

import argparse

import rospy
from sensor_msgs.msg import BatteryState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="power", help="Battery topic name")
    parser.add_argument("--rate", type=float, default=2.0, help="Publish rate (Hz)")
    parser.add_argument("--low", type=float, default=12.0, help="Low voltage value")
    parser.add_argument(
        "--normal", type=float, default=16.0, help="Normal voltage value"
    )
    parser.add_argument("--cycle", type=float, default=5.0, help="Seconds per phase")
    args, _ = parser.parse_known_args()

    rospy.init_node("test_battery_pub", anonymous=True)
    pub = rospy.Publisher(args.topic, BatteryState, queue_size=10)
    rate = rospy.Rate(args.rate)

    rospy.loginfo(
        f"Publishing fake battery: normal={args.normal}V, low={args.low}V, "
        f"cycle={args.cycle}s, rate={args.rate}Hz, topic={args.topic}"
    )

    while not rospy.is_shutdown():
        t = rospy.get_time()
        # Alternate between normal and low every `cycle` seconds
        phase = int(t / args.cycle) % 2
        voltage = args.low if phase == 1 else args.normal

        msg = BatteryState()
        msg.header.stamp = rospy.Time.now()
        msg.voltage = voltage
        pub.publish(msg)

        phase_name = "LOW 🔴" if phase == 1 else "NORMAL 🟢"
        rospy.loginfo_throttle(1.0, f"Battery: {voltage:.1f}V ({phase_name})")
        rate.sleep()


if __name__ == "__main__":
    main()
