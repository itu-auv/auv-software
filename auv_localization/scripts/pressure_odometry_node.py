#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import auv_common_lib.transform.transformer
from auv_common_lib.logging.terminal_color_utils import TerminalColors
import tf


class PressureToOdom:
    def __init__(self):
        rospy.init_node("pressure_to_odom_node", anonymous=True)
        self.depth_calibration_offset = rospy.get_param(
            "sensors/external_pressure_sensor/depth_offset", 0.0
        )
        self.depth_calibration_covariance = rospy.get_param(
            "sensors/external_pressure_sensor/depth_covariance", 0.00005
        )
        self.odom_publisher = rospy.Publisher(
            "localization/odom_pressure", Odometry, queue_size=10
        )
        

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "taluy/base_link"
        
        self.transformer = auv_common_lib.transform.transformer.Transformer()

        # Initialize covariances with zeros
        self.odom_msg.pose.covariance = np.zeros(36).tolist()
        self.odom_msg.twist.covariance = np.zeros(36).tolist()
        self.odom_msg.pose.covariance[14] = self.depth_calibration_covariance

        # Log loaded parameters
        pressure_odometry_colored = TerminalColors.color_text(
            "Pressure Odometry Calibration data loaded", TerminalColors.PASTEL_GREEN
        )
        rospy.loginfo(
            f"{pressure_odometry_colored} : depth offset: {self.depth_calibration_offset}"
        )
        rospy.loginfo(
            f"{pressure_odometry_colored} : depth covariance: {self.depth_calibration_covariance}"
        )
        
        self.depth_subscriber = rospy.Subscriber("depth", Float32, self.depth_callback)
 
    def get_global_base_to_pressure_height(self):
        translation, _ = self.transformer.get_transform(
            "taluy/base_link", "taluy/base_link/external_pressure_sensor_link"
        )

        # Getting the rotation matrix
        _, global_rotation = self.transformer.get_transform("odom", "taluy/base_link")
        rotation_matrix = tf.transformations.quaternion_matrix(global_rotation)
        
        # Rotating
        sensor_position_vector = np.array([[translation[0], translation[1], translation[2], 1.0]])
        global_sensor_position_vector = np.dot(rotation_matrix, sensor_position_vector)
        return global_sensor_position_vector[2]

    def depth_callback(self, depth_msg):
        # Fill the odometry message with depth data as the z component of the linear position
        self.odom_msg.header.stamp = rospy.Time.now()

        # Calibrate depth with the offset
        calibrated_depth = depth_msg.data + self.depth_calibration_offset

        # Calculate the depth of the base link
        base_link_depth = calibrated_depth + self.get_global_base_to_pressure_height()

        # Update odom message with transformed depth
        self.odom_msg.pose.pose.position.z = base_link_depth
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0

        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = 0.0

        # Publish the odometry message
        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        pressure_to_odom = PressureToOdom()
        pressure_to_odom.run()
    except rospy.ROSInterruptException:
        pass
