#!/usr/bin/env python
"""
This script subscribes to the odometry topic and publishes the Euler orientation, plus the yaw in degrees for convenience."""


import rospy
import math
import angles
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

        # Normalize yaw to the range [-pi, pi]
        yaw_normalized = angles.normalize_angle(yaw)

        # Create and populate Euler orientation message
        euler_msg = EulerOrientation()
        euler_msg.roll = round(roll, 3)
        euler_msg.pitch = round(pitch, 3)
        euler_msg.yaw = round(yaw_normalized, 3)
        euler_msg.yaw_degrees = round(math.degrees(yaw_normalized), 3)

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
