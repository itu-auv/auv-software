import numpy as np
import rospy
import tf2_ros
import tf.transformations
import sensor_msgs.msg
import auv_common_lib.transform.transformer


class IMUSpatialTransformer:
    def __init__(self, source_frame: str):
        self.angular_rate = np.zeros((3, 1))

        self.transformer = auv_common_lib.transform.transformer.Transformer()
        self.source_frame = source_frame

        rospy.Subscriber("imu/data", sensor_msgs.msg.Imu, self.imu_data_callback)

    def imu_data_callback(self, data: sensor_msgs.msg.Imu):
        angular_rate_raw = np.array(
            [
                data.angular_velocity.x,
                data.angular_velocity.y,
                data.angular_velocity.z,
            ]
        )

        _, imu_rotation = self.transformer.get_transform(
            self.source_frame, data.header.frame_id
        )
        imu_rotation_matrix = tf.transformations.quaternion_matrix(imu_rotation)[:3, :3]
        self.angular_rate = np.matmul(imu_rotation_matrix, angular_rate_raw)

    def transform_linear_velocity(
        self, linear_velocity: np.array, frame_id: str
    ) -> np.ndarray:

        dvl_translation, dvl_rotation = self.transformer.get_transform(
            self.source_frame, frame_id
        )
        dvl_rotation_matrix = tf.transformations.quaternion_matrix(dvl_rotation)[:3, :3]
        dvl_velocity_in_base_link = np.matmul(dvl_rotation_matrix, linear_velocity)
        dvl_translation = np.array(dvl_translation)
        linear_velocity_due_to_rotation = np.cross(self.angular_rate, dvl_translation)

        return dvl_velocity_in_base_link - linear_velocity_due_to_rotation
