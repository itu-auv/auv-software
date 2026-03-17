#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import auv_common_lib.transform.transformer
import tf


class PressureToOdom:
    def __init__(self):
        rospy.init_node("pressure_to_odom_node", anonymous=True)

        self.odom_orientation = None
        self.depth_data = None
        self.base_to_pressure_translation = None

        # params
        self.namespace = rospy.get_param("~namespace", "taluy")
        self.base_frame = self.namespace + "/base_link"
        self.pressure_sensor_frame = (
            self.namespace + "/base_link/external_pressure_sensor_link"
        )

        self.depth_calibration_offset = rospy.get_param(
            "sensors/external_pressure_sensor/depth_offset", -0.10
        )
        self.depth_calibration_covariance = rospy.get_param(
            "sensors/external_pressure_sensor/depth_covariance", 0.05
        )
        self.pool_depth = rospy.get_param("/env/pool_depth", 2.2)
        self.min_valid_depth = rospy.get_param("~min_valid_depth", 0.1)
        self.max_valid_depth = rospy.get_param("~max_valid_depth", -self.pool_depth)

        # tf transformer
        self.transformer = auv_common_lib.transform.transformer.Transformer()

        # publishers and subscribers
        self.odom_publisher = rospy.Publisher("odom_pressure", Odometry, queue_size=10)
        self.odom_orientation_subscriber = rospy.Subscriber(
            "odometry",
            Odometry,
            self.odom_callback,
            tcp_nodelay=True,
        )
        self.depth_subscriber = rospy.Subscriber(
            "depth", Float32, self.depth_callback, tcp_nodelay=True
        )

        # initialize odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.namespace + "/base_link"
        self.odom_msg.pose.covariance[14] = self.depth_calibration_covariance

        rospy.loginfo("[PressureToOdom] Node initialized with parameters:")
        rospy.loginfo(f"  depth_calibration_offset: {self.depth_calibration_offset}")
        rospy.loginfo(
            f"  depth_calibration_covariance: {self.depth_calibration_covariance}"
        )
        rospy.loginfo(f"  pool_depth: {self.pool_depth}")
        rospy.loginfo(f"  min_valid_depth: {self.min_valid_depth}")
        rospy.loginfo(f"  max_valid_depth: {self.max_valid_depth}")

    def odom_callback(self, odom_msg):
        self.odom_orientation = odom_msg.pose.pose.orientation

    def get_base_to_pressure_height(self):
        if self.base_to_pressure_translation is None:
            try:
                trans, _ = self.transformer.get_transform(
                    self.base_frame, self.pressure_sensor_frame
                )
                self.base_to_pressure_translation = np.array(trans).flatten()
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                rospy.logwarn(
                    f"{self.pressure_sensor_frame} TF not available: {e}; using zero offset."
                )
                return 0.0

        translation = self.base_to_pressure_translation

        if self.odom_orientation is None:
            return float(translation[2])

        # compute rotated z-offset based on current orientation
        orientation = self.odom_orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(quat)[:3, :3]
        rotated_vector = rotation_matrix.dot(translation)
        return float(rotated_vector[2])

    def depth_callback(self, msg):
        self.depth_data = msg.data
        self.process_pressure_depth()

    def process_pressure_depth(self):
        if self.depth_data is None:
            rospy.logwarn_throttle(5.0, "Waiting for depth data...")
            return

        pressure_depth_calibrated = self.depth_data + self.depth_calibration_offset
        pressure_depth = pressure_depth_calibrated + self.get_base_to_pressure_height()

        is_pressure_valid = (
            self.max_valid_depth <= pressure_depth <= self.min_valid_depth
        )

        if not is_pressure_valid:
            rospy.logwarn_throttle(
                3.0,
                f"Pressure depth out of valid range [{self.max_valid_depth:.3f}, {self.min_valid_depth:.3f}]: {pressure_depth:.3f}",
            )
            return

        self.publish_odometry(pressure_depth)

    def publish_odometry(self, depth):
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.pose.pose.position.z = depth

        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        pressure_to_odom = PressureToOdom()
        pressure_to_odom.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in pressure_to_odom_node: {str(e)}")
