#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import auv_common_lib.transform.transformer
from auv_common_lib.logging.terminal_color_utils import TerminalColors
import tf


class PressureToOdom:
    def __init__(self):
        rospy.init_node("pressure_to_odom_node", anonymous=True)

        # Initialize class variables
        self.imu_data = None
        self.depth_data = None
        self.dvl_data = None
        self.base_to_pressure_translation = None
        self.base_to_dvl_translation = None

        # Load parameters
        self._load_parameters()

        # Setup publishers and subscribers
        self._setup_publishers()
        self._setup_subscribers()

        # Initialize odometry message
        self.odom_msg = self._initialize_odometry_message()

        # Log initialization
        self._log_initialization()

    def _load_parameters(self):
        """Load all ROS parameters"""
        # Sensor calibration and parameters
        self.depth_calibration_offset = rospy.get_param(
            "sensors/external_pressure_sensor/depth_offset", -0.10
        )
        self.depth_calibration_covariance = rospy.get_param(
            "sensors/external_pressure_sensor/depth_covariance", 0.05
        )
        self.pool_depth = rospy.get_param("/env/pool_depth", 2.2)

        # Validation thresholds
        self.min_valid_altitude = rospy.get_param("~min_valid_altitude", 0.3)
        self.max_valid_altitude = rospy.get_param(
            "~max_valid_altitude", self.pool_depth
        )
        self.min_valid_depth = rospy.get_param("~min_valid_depth", 0.0)
        self.max_valid_depth = rospy.get_param("~max_valid_depth", -self.pool_depth)

        # TF transformer
        self.transformer = auv_common_lib.transform.transformer.Transformer()

    def _setup_publishers(self):
        """Initialize all publishers"""
        self.odom_publisher = rospy.Publisher("odom_pressure", Odometry, queue_size=10)
        self.dvl_depth_publisher = rospy.Publisher("dvl_depth", Float32, queue_size=10)

    def _setup_subscribers(self):
        """Initialize all subscribers"""
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )
        self.depth_subscriber = rospy.Subscriber(
            "depth", Float32, self.depth_callback, tcp_nodelay=True
        )
        self.dvl_subscriber = rospy.Subscriber(
            "dvl/altitude", Float32, self.dvl_callback, tcp_nodelay=True
        )

    def _initialize_odometry_message(self):
        """Initialize and configure the odometry message"""
        odom_msg = Odometry()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "taluy/base_link"

        # Initialize covariances
        odom_msg.pose.covariance = np.zeros(36).tolist()
        odom_msg.twist.covariance = np.zeros(36).tolist()

        # Set depth covariance (index 14 corresponds to z-position)
        odom_msg.pose.covariance[14] = self.depth_calibration_covariance

        return odom_msg

    def _log_initialization(self):
        """Log initialization information"""
        pressure_odometry_colored = TerminalColors.color_text(
            "Pressure Odometry Calibration data loaded", TerminalColors.PASTEL_GREEN
        )
        rospy.loginfo(
            f"{pressure_odometry_colored} : depth offset: {self.depth_calibration_offset}"
        )
        rospy.loginfo(
            f"{pressure_odometry_colored} : depth covariance: {self.depth_calibration_covariance}"
        )
        rospy.loginfo(f"{pressure_odometry_colored} : pool_depth: {self.pool_depth}")

    def imu_callback(self, imu_msg):
        """Callback for IMU data to get current orientation"""
        self.imu_data = imu_msg

    def _get_sensor_height(self, frame_id, cache_attr):
        """
        Generic method to get height offset for any sensor
        Args:
            frame_id: TF frame ID of the sensor
            cache_attr: Attribute name where to cache the translation
        Returns:
            float: Z-offset of the sensor in base_link frame
        """
        # Try to fetch and cache the TF once
        if getattr(self, cache_attr) is None:
            try:
                trans, _ = self.transformer.get_transform("taluy/base_link", frame_id)
                arr = np.array(trans)
                # flatten any nested structure to 1D [x, y, z]
                setattr(self, cache_attr, arr.flatten())
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    10, f"{frame_id} TF not available: {e}; using zero offset."
                )
                return 0.0

        translation = getattr(self, cache_attr)

        # If no IMU data arrived yet, use the static Z offset
        if self.imu_data is None:
            rospy.logwarn_throttle(
                10, "No IMU data received yet. Using default orientation."
            )
            return float(translation[2])

        # Compute rotated Z-offset based on current orientation
        orientation = self.imu_data.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(quat)[:3, :3]
        rotated_vector = rotation_matrix.dot(translation)
        return float(rotated_vector[2])

    def get_base_to_pressure_height(self):
        """Get pressure sensor height with respect to base_link"""
        return self._get_sensor_height(
            "taluy/base_link/external_pressure_sensor_link",
            "base_to_pressure_translation",
        )

    def get_base_to_dvl_height(self):
        """Get DVL height with respect to base_link"""
        return self._get_sensor_height(
            "taluy/base_link/dvl_link", "base_to_dvl_translation"
        )

    def depth_callback(self, msg):
        """Callback for depth messages"""
        self.depth_data = msg.data
        self.process_fused_depth()

    def dvl_callback(self, msg):
        """Callback for DVL altitude messages"""
        self.dvl_data = msg.data
        self.process_fused_depth()

    def process_fused_depth(self):
        """Process and fuse depth and DVL data"""
        if self.depth_data is None and self.dvl_data is None:
            rospy.logwarn_throttle(5.0, "Waiting for both depth and DVL data...")
            return

        try:
            final_depth = None

            pressure_depth = None
            is_pressure_valid = False
            if self.depth_data is not None:
                pressure_depth_calibrated = (
                    self.depth_data + self.depth_calibration_offset
                )
                pressure_depth = (
                    pressure_depth_calibrated + self.get_base_to_pressure_height()
                )
                is_pressure_valid = (
                    self.max_valid_depth <= pressure_depth <= self.min_valid_depth
                )
            else:
                rospy.logwarn_throttle(
                    5.0, "Depth (Bar30) missing; will rely on dvl/altitude."
                )

            dvl_depth = None
            is_dvl_valid = False
            if self.dvl_data is not None:
                dvl_altitude = self.dvl_data
                dvl_depth = (
                    dvl_altitude - self.pool_depth
                ) + self.get_base_to_dvl_height()
                self.dvl_depth_publisher.publish(dvl_depth)
                is_dvl_valid = (
                    self.min_valid_altitude <= dvl_altitude <= self.max_valid_altitude
                )
            else:
                rospy.logwarn_throttle(
                    5.0, "DVL altitude missing; will rely on pressure sensor."
                )

            if is_pressure_valid:
                final_depth = pressure_depth
                rospy.logdebug_throttle(
                    1.0, f"Using valid pressure depth: {final_depth:.3f}"
                )
            elif is_dvl_valid:
                final_depth = dvl_depth
                rospy.logwarn_throttle(
                    1.0, f"Pressure invalid/missing, using DVL depth: {final_depth:.3f}"
                )
            else:
                rospy.logerr_throttle(
                    1.0, "Available sensor(s) gave invalid depth; not publishing."
                )
                return

            self._publish_odometry(final_depth)

        except Exception as e:
            rospy.logerr(f"Error in process_fused_depth: {str(e)}")

    def _publish_odometry(self, depth):
        """Publish odometry message with the given depth"""
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.pose.pose.position.z = depth

        # Set other components to zero as they are not provided by these sensors
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0
        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = 0.0

        self.odom_publisher.publish(self.odom_msg)

    def run(self):
        """Main run loop"""
        rospy.spin()


if __name__ == "__main__":
    try:
        pressure_to_odom = PressureToOdom()
        pressure_to_odom.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in pressure_to_odom_node: {str(e)}")
