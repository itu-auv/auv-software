from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros


def is_transform_fresh(transform, node, freshness_threshold=Duration(seconds=0.2)):
    stamp = Time.from_msg(transform.header.stamp)
    if stamp.nanoseconds == 0:
        return True

    age = abs((node.get_clock().now() - stamp).nanoseconds / 1e9)
    return age <= freshness_threshold.nanoseconds / 1e9


def lookup_fresh_transform(
    tf_buffer,
    node,
    target_frame,
    source_frame,
    timeout=Duration(seconds=0.2),
    freshness_threshold=Duration(seconds=0.2),
):
    transform = tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        Time(),
        timeout,
    )
    if not is_transform_fresh(transform, node, freshness_threshold):
        age = abs(
            (node.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds
            / 1e9
        )
        raise tf2_ros.ExtrapolationException(
            f"Transform from {source_frame} to {target_frame} is stale "
            f"({age:.3f}s > {freshness_threshold.nanoseconds / 1e9:.3f}s)"
        )

    return transform
