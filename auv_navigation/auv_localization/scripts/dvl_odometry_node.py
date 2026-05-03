#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import math
import message_filters
import time
import tf
import tf.transformations
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class DvlToOdom:
    def __init__(self):
        rospy.init_node("dvl_to_odom_node", anonymous=True)

        self.enabled = rospy.get_param("~enabled", True)
        self.enable_service = rospy.Service(
            "dvl_to_odom_node/enable", SetBool, self.enable_cb
        )

        self.namespace = rospy.get_param("~namespace", "taluy")
        self.base_frame = rospy.get_param("~base_frame", f"{self.namespace}/base_link")
        self.dvl_frame = rospy.get_param(
            "~dvl_frame", f"{self.namespace}/base_link/dvl_link"
        )

        self.tf_listener = tf.TransformListener()
        self.dvl_yaw = self.get_dvl_yaw()

        self.linear_x_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_x", 0.000015
        )
        self.linear_y_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_y", 0.000015
        )
        self.linear_z_covariance = rospy.get_param(
            "sensors/dvl/covariance/linear_z", 0.00005
        )

        # Subscribers and Publishers
        self.dvl_velocity_subscriber = message_filters.Subscriber(
            "dvl/velocity_raw", Twist, tcp_nodelay=True
        )
        self.is_valid_subscriber = message_filters.Subscriber(
            "dvl/is_valid", Bool, tcp_nodelay=True
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.dvl_velocity_subscriber, self.is_valid_subscriber],
            queue_size=10,
            slop=0.1,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.dvl_callback)

        self.odom_publisher = rospy.Publisher("odom_dvl", Odometry, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.namespace + "/base_link"

        # Initialize covariances with default values
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance[0] = self.linear_x_covariance
        self.odom_msg.twist.covariance[7] = self.linear_y_covariance
        self.odom_msg.twist.covariance[14] = self.linear_z_covariance

        # Logging
        DVL_odometry_colored = TerminalColors.color_text(
            "DVL Odometry Calibration data loaded", TerminalColors.PASTEL_BLUE
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear x covariance: {self.linear_x_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear y covariance: {self.linear_y_covariance}"
        )
        rospy.loginfo(
            f"{DVL_odometry_colored} : linear z covariance: {self.linear_z_covariance}"
        )

        self.is_dvl_enabled = False

    def enable_cb(self, req):
        """Service callback to enable/disable DVL->Odom processing"""
        self.enabled = req.data
        state = "enabled" if self.enabled else "disabled"
        rospy.loginfo(f"DVL->Odom node {state} via service call.")
        return SetBoolResponse(success=True, message=f"DVL->Odom {state}")

    def get_dvl_yaw(self):
        rospy.loginfo(f"Waiting for TF: {self.base_frame} <- {self.dvl_frame}")
        try:
            self.tf_listener.waitForTransform(
                self.base_frame, self.dvl_frame, rospy.Time(0), rospy.Duration(10.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, self.dvl_frame, rospy.Time(0)
            )
            euler = tf.transformations.euler_from_quaternion(rot)
            yaw = euler[2]
            rospy.loginfo(f"Loaded {self.dvl_frame} yaw from TF: {yaw} radians")
            return yaw
        except Exception as e:
            rospy.logwarn(
                f"Could not get TF for {self.dvl_frame} relative to {self.base_frame}: {e}. Falling back to default -45 degrees."
            )
            return np.radians(-45)

    def transform_vector(self, vector):
        theta = self.dvl_yaw
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(rotation_matrix, np.array(vector))

    def dvl_callback(self, velocity_msg, is_valid_msg):
        self.is_dvl_enabled = True

        if not self.enabled:
            return

        # Sadece DVL verisi valid ise yayinla, valid degilse hicbir sey yapma (yeni hiz yayinlanmaz)
        if is_valid_msg.data:
            rotated_vector = self.transform_vector(
                [velocity_msg.linear.x, velocity_msg.linear.y, velocity_msg.linear.z]
            )
            velocity_msg.linear.x = rotated_vector[0]
            velocity_msg.linear.y = rotated_vector[1]
            velocity_msg.linear.z = rotated_vector[2]

            # Fill the odometry message
            self.odom_msg.header.stamp = rospy.Time.now()
            self.odom_msg.twist.twist = velocity_msg

            # Set position to zero as we are not computing it here
            self.odom_msg.pose.pose.position.x = 0.0
            self.odom_msg.pose.pose.position.y = 0.0
            self.odom_msg.pose.pose.position.z = 0.0

            # Publish the odometry message
            self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.is_dvl_enabled:
                self.odom_msg.header.stamp = rospy.Time.now()
                self.odom_msg.twist.twist = Twist()
                self.odom_publisher.publish(self.odom_msg)
            rate.sleep()
        rospy.spin()


if __name__ == "__main__":
    try:
        dvl_to_odom = DvlToOdom()
        dvl_to_odom.run()
    except rospy.ROSInterruptException:
        pass
