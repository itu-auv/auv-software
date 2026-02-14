#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import yaml
from auv_common_lib.logging.terminal_color_utils import TerminalColors


HIGH_COVARIANCE = 1e6


class ImuToOdom:
    def __init__(self):
        rospy.init_node("vectornav_imu_to_odom", anonymous=True)

        self.imu_topic = rospy.get_param("~imu_topic", "imu/data")
        self.odom_topic = rospy.get_param("~odom_topic", "odom_vectornav")

        # Frame parameters
        self.odom_frame = rospy.get_param("~odom_frame", "odom_vectornav")
        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link_vectornav")

        # Subscriber and Publisher (only Xsens-style logic kept)
        self.imu_subscriber = rospy.Subscriber(
            self.imu_topic, Imu, self.imu_callback, tcp_nodelay=True
        )
        self.odom_publisher = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = self.odom_frame
        self.odom_msg.child_frame_id = self.base_link_frame

        # Build default covariance matrices
        self.default_pose_cov = np.zeros((6, 6))
        self.default_pose_cov[:3, :3] = np.eye(3) * HIGH_COVARIANCE

        self.default_twist_cov = np.zeros((6, 6))
        self.default_twist_cov[:3, :3] = np.eye(3) * HIGH_COVARIANCE

        # Initialize covariances with the default matrices
        self.odom_msg.pose.covariance = self.default_pose_cov.flatten().tolist()
        self.odom_msg.twist.covariance = self.default_twist_cov.flatten().tolist()

        # Timeout and state management
        self.last_imu_time = None
        self.imu_timeout = rospy.Duration(1.0)  # 1 second timeout
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_imu_timeout)

    def check_imu_timeout(self, event):
        if self.last_imu_time and (rospy.Time.now() - self.last_imu_time) > self.imu_timeout:
            rospy.logwarn_throttle(5, "spam ctrl-c or suicide")
            self.last_imu_time = None

    def insert_covariance_block(
        self,
        base_covariance_6x6,
        input_covariance_3x3_flat,
    ):
        """
        Inserts a 3x3 covariance block (provided as a flat list) into a 6x6 NumPy matrix
        and returns the resulting flattened 6x6 list.

        The 3x3 block is always inserted into the bottom-right quadrant (indices 3:6, 3:6),
        which corresponds to orientation in pose covariance or angular velocity in twist covariance.
        """
        updated_covariance_6x6 = base_covariance_6x6.copy()

        # Only insert the covariance block if it has 9 elements
        if len(input_covariance_3x3_flat) == 9:

            input_covariance_3x3_matrix = np.array(input_covariance_3x3_flat).reshape(
                3, 3
            )

            # Insert the 3x3 matrix into the bottom-right quadrant (angular/orientation part)
            updated_covariance_6x6[3:6, 3:6] = input_covariance_3x3_matrix
        else:
            rospy.logwarn_throttle(
                3,
                f"Received invalid IMU covariance size: {len(input_covariance_3x3_flat)}. "
                f"Using default covariance",
            )
        return updated_covariance_6x6.flatten().tolist()

    def update_pose_covariance(self, imu_orientation_covariance):
        return self.insert_covariance_block(
            self.default_pose_cov, imu_orientation_covariance
        )

    def update_twist_covariance(self, imu_angular_velocity_covariance):
        return self.insert_covariance_block(
            self.default_twist_cov, imu_angular_velocity_covariance
        )

    def imu_callback(self, imu_msg):
        """Single IMU callback kept (Xsens logic only)."""
        self.last_imu_time = rospy.Time.now()

        self.odom_msg.header.stamp = imu_msg.header.stamp

        # Update twist and twist covariance
        self.odom_msg.twist.twist.angular = Vector3(
            # TODO(@baykara): change
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
        )
        self.odom_msg.twist.covariance = self.update_twist_covariance(
            imu_msg.angular_velocity_covariance
        )

        # Update orientation and orientation covariance (Xsens-style inversion kept)
        self.odom_msg.pose.pose.orientation = (
            imu_msg.orientation
        )
        self.odom_msg.pose.covariance = self.update_pose_covariance(
            imu_msg.orientation_covariance
        )

        # Zero-out position and linear velocity
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        # Publish using the single publisher for this node
        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        imu_to_odom = ImuToOdom()
        imu_to_odom.run()
    except rospy.ROSInterruptException:
        pass
