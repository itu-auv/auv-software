#!/usr/bin/env python3
"""
Temporary test node for DVL bag replay.
Subscribes to /taluy_mini/dvl/velocity (Twist) from a bag file and publishes:
  - odom_dvl      : DVL velocity as odometry twist (vx, vy)
  - odom_pressure : constant depth z = -1
  - odom_imu      : identity orientation, zero angular velocity
  - odom_bno_imu  : identity orientation, zero angular velocity

All topics are remapped via the launch file to feed into the EKF.
DELETE THIS FILE AFTER TESTING.
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DvlBagTestNode:
    def __init__(self):
        rospy.init_node("dvl_bag_test_node", anonymous=True)

        # ── Parameters ──────────────────────────────────────────────
        self.frame_id = rospy.get_param("~frame_id", "odom")
        self.child_frame_id = rospy.get_param("~child_frame_id", "taluy/base_link")
        self.pressure_depth = rospy.get_param("~pressure_depth", -1.0)
        self.rate_hz = rospy.get_param("~rate", 20)

        # ── Publishers ──────────────────────────────────────────────
        self.dvl_odom_pub = rospy.Publisher("odom_dvl", Odometry, queue_size=10)
        self.pressure_odom_pub = rospy.Publisher(
            "odom_pressure", Odometry, queue_size=10
        )
        self.imu_odom_pub = rospy.Publisher("odom_imu", Odometry, queue_size=10)
        self.bno_imu_odom_pub = rospy.Publisher("odom_bno_imu", Odometry, queue_size=10)

        # ── Subscriber ──────────────────────────────────────────────
        self.dvl_sub = rospy.Subscriber(
            "dvl/velocity", Twist, self.dvl_callback, tcp_nodelay=True
        )

        # ── Pre-build reusable odometry messages ────────────────────
        self.dvl_odom_msg = self._make_odom()
        self.pressure_odom_msg = self._make_odom()
        self.imu_odom_msg = self._make_odom()
        self.bno_imu_odom_msg = self._make_odom()

        # DVL covariances (vx, vy used by EKF)
        self.dvl_odom_msg.twist.covariance[0] = 0.000015  # vx
        self.dvl_odom_msg.twist.covariance[7] = 0.000015  # vy
        self.dvl_odom_msg.twist.covariance[14] = 0.00005  # vz

        # Pressure covariance (z position used by EKF)
        self.pressure_odom_msg.pose.covariance[14] = 0.045  # z

        # IMU covariances – orientation (roll, pitch) and angular velocity
        HIGH_COV = 1e6
        for msg in (self.imu_odom_msg, self.bno_imu_odom_msg):
            cov_pose = np.zeros(36)
            cov_pose[0] = HIGH_COV
            cov_pose[7] = HIGH_COV
            cov_pose[14] = HIGH_COV
            # orientation covariance small for roll/pitch
            cov_pose[21] = 0.001  # roll
            cov_pose[28] = 0.001  # pitch
            cov_pose[35] = 0.001  # yaw
            msg.pose.covariance = cov_pose.tolist()

            cov_twist = np.zeros(36)
            cov_twist[0] = HIGH_COV
            cov_twist[7] = HIGH_COV
            cov_twist[14] = HIGH_COV
            cov_twist[21] = 0.001
            cov_twist[28] = 0.001
            cov_twist[35] = 0.001
            msg.twist.covariance = cov_twist.tolist()

            # identity quaternion (no rotation)
            msg.pose.pose.orientation.w = 1.0

        rospy.loginfo(
            "[dvl_bag_test] Node started – waiting for DVL velocity from bag…"
        )

    # ── Helpers ─────────────────────────────────────────────────────
    def _make_odom(self):
        msg = Odometry()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.pose.covariance = np.zeros(36).tolist()
        msg.twist.covariance = np.zeros(36).tolist()
        return msg

    # ── Helpers ──────────────────────────────────────────────────────
    def transform_vector(self, vector):
        """Rotate DVL velocity by +90° to correct sensor mounting offset."""
        theta = np.radians(90)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(rotation_matrix, np.array(vector))

    # ── Callbacks ───────────────────────────────────────────────────
    def dvl_callback(self, twist_msg):
        """Receive raw DVL velocity from bag and publish as odom_dvl."""
        now = rospy.Time.now()

        # Rotate DVL velocity to correct mounting offset (same as dvl_odometry_node)
        rotated = self.transform_vector(
            [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z]
        )
        twist_msg.linear.x = rotated[0]
        twist_msg.linear.y = rotated[1]
        twist_msg.linear.z = rotated[2]

        # DVL odometry – only twist is meaningful
        self.dvl_odom_msg.header.stamp = now
        self.dvl_odom_msg.twist.twist = twist_msg
        self.dvl_odom_pub.publish(self.dvl_odom_msg)

    # ── Periodic publishers ─────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()

            # Pressure: constant depth = -1
            self.pressure_odom_msg.header.stamp = now
            self.pressure_odom_msg.pose.pose.position.z = self.pressure_depth
            self.pressure_odom_pub.publish(self.pressure_odom_msg)

            # IMU: identity orientation, zero angular velocity
            self.imu_odom_msg.header.stamp = now
            self.imu_odom_pub.publish(self.imu_odom_msg)

            self.bno_imu_odom_msg.header.stamp = now
            self.bno_imu_odom_pub.publish(self.bno_imu_odom_msg)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = DvlBagTestNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
