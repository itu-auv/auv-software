#!/usr/bin/env python3

import rospy
import tf
import tf.transformations
import numpy as np
import yaml
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
from auv_localization.srv import CalibrateIMU, CalibrateIMUResponse

HIGH_COVARIANCE = 1e6


class ImuToOdom:
    def __init__(self):
        rospy.init_node("imu_to_odom_node", anonymous=True)

        self.namespace = rospy.get_param("~namespace", "taluy")
        self.imu_calibration_data_path = rospy.get_param(
            "~imu_calibration_path", "config/imu_calibration_data.yaml"
        )

        self.base_frame = rospy.get_param("~base_frame", f"{self.namespace}/base_link")
        self.imu_frame = rospy.get_param(
            "~imu_frame", f"{self.namespace}/base_link/imu"
        )
        self.gravity_magnitude = rospy.get_param(
            "~gravity_magnitude", rospy.get_param("/env/gravity", 9.81)
        )

        self.tf_listener = tf.TransformListener()

        self.imu_to_base_q = self.get_frame_rotation(self.imu_frame)
        self.imu_to_base_rotation_matrix = None
        self.base_to_imu_q = None
        if self.imu_to_base_q is not None:
            self.imu_to_base_rotation_matrix = tf.transformations.quaternion_matrix(
                self.imu_to_base_q
            )[:3, :3]
            self.base_to_imu_q = tf.transformations.quaternion_inverse(
                self.imu_to_base_q
            )

        # Subscribers and Publishers
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )
        self.odom_publisher = rospy.Publisher("odom_imu", Odometry, queue_size=10)
        self.imu_publisher = rospy.Publisher("imu", Imu, queue_size=10)

        # Initialize the odometry message
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = self.base_frame

        # Build default covariance matrices
        self.default_pose_cov = np.zeros((6, 6))
        self.default_pose_cov[:3, :3] = np.eye(3) * HIGH_COVARIANCE

        self.default_twist_cov = np.zeros((6, 6))
        self.default_twist_cov[:3, :3] = np.eye(3) * HIGH_COVARIANCE

        # Initialize covariances with the default matrices
        self.odom_msg.pose.covariance = self.default_pose_cov.flatten().tolist()
        self.odom_msg.twist.covariance = self.default_twist_cov.flatten().tolist()

        # Variables for bias correction
        self.angular_velocity_bias = np.zeros(3)
        self.linear_acceleration_bias = np.zeros(3)
        self.calibrating_angular_velocity = False
        self.angular_velocity_calibration_data = []
        self.calibrating_linear_acceleration = False
        self.linear_acceleration_calibration_data = []

        # Load calibration data if available
        self.load_calibration_data()

        # Service for IMU calibration
        self.calibration_service = rospy.Service(
            "calibrate_imu", CalibrateIMU, self.calibrate_imu
        )
        self.linear_acceleration_calibration_service = rospy.Service(
            "calibrate_imu_acceleration",
            CalibrateIMU,
            self.calibrate_linear_acceleration,
        )

    def get_frame_rotation(self, frame_id):
        rospy.loginfo(f"Waiting for TF: {self.base_frame} <- {frame_id}")
        try:
            self.tf_listener.waitForTransform(
                self.base_frame, frame_id, rospy.Time(0), rospy.Duration(10.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, frame_id, rospy.Time(0)
            )
            rospy.loginfo(f"Loaded {frame_id} rotation from TF: {rot}")
            return np.array(rot)
        except Exception as e:
            rospy.logwarn(
                f"Could not get TF for {frame_id} relative to {self.base_frame}: {e}. No frame rotation will be applied."
            )
            return None

    def quaternion_multiply(self, q1, q2):
        x0, y0, z0, w0 = q1
        x1, y1, z1, w1 = q2

        x_new = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y_new = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        z_new = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        w_new = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1

        return np.array([x_new, y_new, z_new, w_new])

    def insert_covariance_block(self, base_covariance_6x6, input_covariance_3x3_flat):
        updated_covariance_6x6 = base_covariance_6x6.copy()
        if len(input_covariance_3x3_flat) == 9 and input_covariance_3x3_flat[0] != -1:
            input_covariance_3x3_matrix = np.array(input_covariance_3x3_flat).reshape(
                3, 3
            )
            updated_covariance_6x6[3:6, 3:6] = input_covariance_3x3_matrix
        return updated_covariance_6x6.flatten().tolist()

    def rotate_vector(self, vector):
        if self.imu_to_base_rotation_matrix is None:
            return vector
        return np.dot(self.imu_to_base_rotation_matrix, vector)

    def rotate_covariance(self, covariance_flat):
        if len(covariance_flat) != 9 or covariance_flat[0] == -1:
            return list(covariance_flat)

        if self.imu_to_base_rotation_matrix is None:
            return list(covariance_flat)

        covariance_matrix = np.array(covariance_flat).reshape(3, 3)
        rotated_covariance = np.dot(
            self.imu_to_base_rotation_matrix,
            np.dot(covariance_matrix, self.imu_to_base_rotation_matrix.T),
        )
        return rotated_covariance.flatten().tolist()

    def get_expected_gravity(self, orientation_q):
        norm = np.linalg.norm(orientation_q)
        if norm == 0.0:
            return np.zeros(3)

        normalized_orientation_q = orientation_q / norm
        rotation_matrix = tf.transformations.quaternion_matrix(
            normalized_orientation_q
        )[:3, :3]
        return np.dot(rotation_matrix.T, np.array([0.0, 0.0, self.gravity_magnitude]))

    def calibrate_bias(
        self, duration, calibration_name, flag_attr, data_attr, bias_attr
    ):
        rospy.loginfo(
            f"{calibration_name} calibration started for {duration} seconds..."
        )
        setattr(self, flag_attr, True)
        setattr(self, data_attr, [])
        rospy.sleep(duration)
        setattr(self, flag_attr, False)

        calibration_data = getattr(self, data_attr)
        if len(calibration_data) == 0:
            return CalibrateIMUResponse(
                success=False, message=f"{calibration_name} calibration failed"
            )

        bias = np.mean(calibration_data, axis=0)
        setattr(self, bias_attr, bias)
        self.save_calibration_data()
        rospy.loginfo(f"{calibration_name} calibration completed. Bias: {bias}")
        return CalibrateIMUResponse(
            success=True, message=f"{calibration_name} calibration successful"
        )

    def imu_callback(self, imu_msg):
        raw_orientation_q = np.array(
            [
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w,
            ]
        )
        raw_angular_velocity = np.array(
            [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ]
        )
        raw_linear_acceleration = np.array(
            [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
            ]
        )

        if self.calibrating_angular_velocity:
            self.angular_velocity_calibration_data.append(raw_angular_velocity.tolist())

        if self.calibrating_linear_acceleration:
            self.linear_acceleration_calibration_data.append(
                (
                    raw_linear_acceleration
                    - self.get_expected_gravity(raw_orientation_q)
                ).tolist()
            )

        self.odom_msg.header.stamp = imu_msg.header.stamp

        corrected_angular_velocity = raw_angular_velocity - self.angular_velocity_bias
        corrected_linear_acceleration = (
            raw_linear_acceleration - self.linear_acceleration_bias
        )

        orientation_q = raw_orientation_q.copy()
        rotated_orientation_covariance = self.rotate_covariance(
            imu_msg.orientation_covariance
        )
        rotated_angular_velocity_covariance = self.rotate_covariance(
            imu_msg.angular_velocity_covariance
        )
        rotated_linear_acceleration_covariance = self.rotate_covariance(
            imu_msg.linear_acceleration_covariance
        )

        if self.imu_to_base_q is not None:
            # q_base^odom = q_imu^odom * q_base^imu = q_imu^odom * inv(q_imu^base)
            orientation_q = self.quaternion_multiply(orientation_q, self.base_to_imu_q)

            corrected_angular_velocity = self.rotate_vector(corrected_angular_velocity)
            corrected_linear_acceleration = self.rotate_vector(
                corrected_linear_acceleration
            )

        self.odom_msg.twist.twist.angular.x = corrected_angular_velocity[0]
        self.odom_msg.twist.twist.angular.y = corrected_angular_velocity[1]
        self.odom_msg.twist.twist.angular.z = corrected_angular_velocity[2]

        self.odom_msg.pose.pose.orientation = Quaternion(
            x=orientation_q[0],
            y=orientation_q[1],
            z=orientation_q[2],
            w=orientation_q[3],
        )

        self.odom_msg.pose.covariance = self.insert_covariance_block(
            self.default_pose_cov, rotated_orientation_covariance
        )
        self.odom_msg.twist.covariance = self.insert_covariance_block(
            self.default_twist_cov, rotated_angular_velocity_covariance
        )

        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0
        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        output_imu_msg = Imu()
        output_imu_msg.header.stamp = imu_msg.header.stamp
        output_imu_msg.header.frame_id = (
            self.base_frame
            if self.imu_to_base_rotation_matrix is not None
            else imu_msg.header.frame_id
        )
        output_imu_msg.orientation = Quaternion(
            x=orientation_q[0],
            y=orientation_q[1],
            z=orientation_q[2],
            w=orientation_q[3],
        )
        output_imu_msg.orientation_covariance = rotated_orientation_covariance
        output_imu_msg.angular_velocity = Vector3(
            x=corrected_angular_velocity[0],
            y=corrected_angular_velocity[1],
            z=corrected_angular_velocity[2],
        )
        output_imu_msg.angular_velocity_covariance = rotated_angular_velocity_covariance
        output_imu_msg.linear_acceleration = Vector3(
            x=corrected_linear_acceleration[0],
            y=corrected_linear_acceleration[1],
            z=corrected_linear_acceleration[2],
        )
        output_imu_msg.linear_acceleration_covariance = (
            rotated_linear_acceleration_covariance
        )

        self.imu_publisher.publish(output_imu_msg)
        self.odom_publisher.publish(self.odom_msg)

    def calibrate_imu(self, req):
        return self.calibrate_bias(
            req.duration,
            "IMU angular velocity bias",
            "calibrating_angular_velocity",
            "angular_velocity_calibration_data",
            "angular_velocity_bias",
        )

    def calibrate_linear_acceleration(self, req):
        return self.calibrate_bias(
            req.duration,
            "IMU linear acceleration bias",
            "calibrating_linear_acceleration",
            "linear_acceleration_calibration_data",
            "linear_acceleration_bias",
        )

    def save_calibration_data(self):
        calibration_data = {
            "drift": self.angular_velocity_bias.tolist(),
            "linear_acceleration_bias": self.linear_acceleration_bias.tolist(),
        }
        try:
            with open(self.imu_calibration_data_path, "w") as f:
                yaml.dump(calibration_data, f)
            rospy.loginfo("Calibration data saved.")
        except Exception as e:
            rospy.logerr(f"Failed to save calibration data: {e}")

    def load_calibration_data(self):
        try:
            with open(self.imu_calibration_data_path, "r") as f:
                calibration_data = yaml.safe_load(f) or {}

            self.angular_velocity_bias = np.array(
                calibration_data.get(
                    "angular_velocity_bias",
                    calibration_data.get("drift", [0.0, 0.0, 0.0]),
                )
            )
            self.linear_acceleration_bias = np.array(
                calibration_data.get("linear_acceleration_bias", [0.0, 0.0, 0.0])
            )
            rospy.loginfo(
                "IMU Calibration data loaded. Angular velocity bias: %s, "
                "linear acceleration bias: %s",
                self.angular_velocity_bias,
                self.linear_acceleration_bias,
            )
        except FileNotFoundError:
            rospy.logwarn("No calibration data found.")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        imu_to_odom = ImuToOdom()
        imu_to_odom.run()
    except rospy.ROSInterruptException:
        pass
