import numpy as np
import rospy
import tf2_ros
import tf.transformations


def is_transform_fresh(transform, freshness_threshold=rospy.Duration(0.2)):
    stamp = transform.header.stamp
    if stamp == rospy.Time(0):
        return True

    age = abs((rospy.Time.now() - stamp).to_sec())
    return age <= freshness_threshold.to_sec()


def lookup_fresh_transform(
    tf_buffer,
    target_frame,
    source_frame,
    timeout=rospy.Duration(0.2),
    freshness_threshold=rospy.Duration(0.2),
):
    transform = tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        rospy.Time(0),
        timeout,
    )
    if not is_transform_fresh(transform, freshness_threshold):
        age = abs((rospy.Time.now() - transform.header.stamp).to_sec())
        raise tf2_ros.ExtrapolationException(
            f"Transform from {source_frame} to {target_frame} is stale "
            f"({age:.3f}s > {freshness_threshold.to_sec():.3f}s)"
        )

    return transform


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
                trans.transform.rotation.w,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
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
