import rospy
import tf2_ros

_tf_buffer = None
_tf_listener = None


def get_tf_buffer(cache_time=None):
    """
    Returns a singleton TF buffer/listener
    """
    global _tf_buffer, _tf_listener

    if _tf_buffer is None:
        if cache_time is None:
            cache_time = rospy.Duration(15.0)

        _tf_buffer = tf2_ros.Buffer(cache_time=cache_time)
        _tf_listener = tf2_ros.TransformListener(_tf_buffer)
        rospy.loginfo(
            "Global TF Buffer and Listener initialized with cache_time={}s".format(
                cache_time.to_sec()
            )
        )

    return _tf_buffer
