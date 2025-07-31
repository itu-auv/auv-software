#!/usr/bin/env python
"""
This script subscribes to the odometry topic and publishes the Euler orientation, plus the yaw in degrees for convenience."""


import rospy
import math
from nav_msgs.msg import Odometry
from auv_msgs.msg import EulerOrientation
from tf.transformations import euler_from_quaternion


class EulerOrientationPublisher:
    def __init__(self):
        rospy.init_node("euler_orientation_publisher", anonymous=True)

        # Publisher for Euler orientation
        self.euler_pub = rospy.Publisher(
            "euler_orientation", EulerOrientation, queue_size=10
        )

        # Subscriber to odometry
        self.odom_sub = rospy.Subscriber("odometry", Odometry, self.odom_callback)

        rospy.loginfo("Euler Orientation Publisher node started")

    def odom_callback(self, msg):
        # Extract quaternion from odometry message
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

        # Convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # Create and populate Euler orientation message
        euler_msg = EulerOrientation()
        euler_msg.roll = roll
        euler_msg.pitch = pitch
        euler_msg.yaw = yaw
        euler_msg.yaw_degrees = math.degrees(yaw)

        # Publish the message
        self.euler_pub.publish(euler_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = EulerOrientationPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
