#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
import tf.transformations
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import auv_common_lib.transform.transformer
from auv_common_lib.logging.terminal_color_utils import TerminalColors


class PressureToOdom:
    def __init__(self):
        rospy.init_node("pressure_to_odom_node", anonymous=True)

        self.namespace = rospy.get_namespace().strip("/")
        self.body_frame = f"{self.namespace}/base_link"

        # Initialize class variables
        self.imu_data = None
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
        """Initialize all subscribers and synchronizers"""
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )

        # Synchronized subscribers with time synchronization
        depth_sub = message_filters.Subscriber("depth", Float32)
        dvl_sub = message_filters.Subscriber("dvl/altitude", Float32)

        # Approximate time synchronizer with 100ms time difference allowance
        ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, dvl_sub],
            queue_size=10,
            slop=0.1,  # 100ms
            allow_headerless=True,
        )
        ts.registerCallback(self.fused_depth_callback)

    def _initialize_odometry_message(self):
        """Initialize and configure the odometry message"""
        odom_msg = Odometry()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = self.body_frame

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
                trans, _ = self.transformer.get_transform(self.body_frame, frame_id)
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
            f"{self.body_frame}/external_pressure_sensor_link",
            "base_to_pressure_translation",
        )

    def get_base_to_dvl_height(self):
        """Get DVL height with respect to base_link"""
        return self._get_sensor_height(
            f"{self.body_frame}/dvl_link", "base_to_dvl_translation"
        )

    def fused_depth_callback(self, depth_msg, dvl_msg):
        """Callback for synchronized depth and DVL messages"""
        try:
            # 1. Get calibrated pressure depth
            pressure_depth_calibrated = depth_msg.data + self.depth_calibration_offset
            pressure_depth = (
                pressure_depth_calibrated + self.get_base_to_pressure_height()
            )

            # 2. Convert DVL altitude to depth (altitude is positive up from seabed)
            dvl_altitude = dvl_msg.data
            dvl_depth = (dvl_altitude - self.pool_depth) + self.get_base_to_dvl_height()
            self.dvl_depth_publisher.publish(dvl_depth)

            # 3. Validate both sensor readings
            is_pressure_valid = (
                self.max_valid_depth <= pressure_depth <= self.min_valid_depth
            )
            is_dvl_valid = (
                self.min_valid_altitude <= dvl_altitude <= self.max_valid_altitude
            )

            final_depth = 0.0
            publish = False

            if is_pressure_valid:
                final_depth = pressure_depth
                publish = True
                rospy.logdebug_throttle(
                    1.0, f"Using valid pressure depth: {final_depth:.3f}"
                )
            elif is_dvl_valid:
                final_depth = dvl_depth
                publish = True
                rospy.logwarn_throttle(
                    1.0, f"Pressure sensor invalid, using DVL depth: {final_depth:.3f}"
                )
            else:
                rospy.logerr_throttle(
                    1.0,
                    "Both pressure and DVL depths are invalid! No odometry published.",
                )

            if publish:
                self._publish_odometry(final_depth)
        except Exception as e:
            rospy.logerr(f"Error in fused_depth_callback: {str(e)}")

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
