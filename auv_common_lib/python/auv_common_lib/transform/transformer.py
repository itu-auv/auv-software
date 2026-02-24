import numpy as np
import rospy
import tf2_ros
import tf.transformations


class Transformer:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_transform(self, source_frame: str, target_frame: str) -> tuple:

        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(), rospy.Duration(4.0)
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rate = rospy.Rate(1.0)
            while not rospy.is_shutdown():
                rospy.logerr(
                    "Waiting for {}->{} transform".format(source_frame, target_frame)
                )
                rate.sleep()

        rotation_matrix = tf.transformations.quaternion_matrix(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]
        )[:3, :3]

        translation_vector = np.array(
            [
                [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ]
            ]
        )

        return translation_vector, rotation_matrix
