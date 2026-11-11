import rclpy
from auv_common_lib.ros import get_param, has_param, get_node
from rclpy.duration import Duration
import tf2_ros

_tf_buffer = None


class _TfBufferHandle:
    def __init__(self, cache_time):
        self._cache_time = cache_time
        self._buffer = None
        self._listener = None
        self.reset(cache_time, initialized=True)

    def reset(self, cache_time=None, initialized=False):
        if cache_time is not None:
            self._cache_time = cache_time

        if self._listener is not None:
            self._listener.unregister()
        self._buffer = tf2_ros.Buffer(cache_time=self._cache_time)
        self._listener = tf2_ros.TransformListener(self._buffer, get_node())
        action = "initialized" if initialized else "reset"
        get_node().get_logger().info(
            "Global TF Buffer and Listener {} with cache_time={}s".format(
                action, self._cache_time.nanoseconds / 1e9
            )
        )

    def __getattr__(self, name):
        return getattr(self._buffer, name)


def get_tf_buffer(cache_time=None):
    """
    Returns a singleton TF buffer/listener
    """
    global _tf_buffer

    if _tf_buffer is None:
        if cache_time is None:
            cache_time = Duration(seconds=15.0)

        _tf_buffer = _TfBufferHandle(cache_time)

    return _tf_buffer


def reset_tf_buffer(cache_time=None):
    """
    Resets the singleton TF buffer while keeping existing state references valid.
    """
    global _tf_buffer
    if _tf_buffer is None:
        return get_tf_buffer(cache_time)

    handle = get_tf_buffer(cache_time)
    handle.reset(cache_time)
    return handle


_base_link_cache = None


def get_base_link():
    """
    Returns the base_link frame name from ROS param (singleton, read once).
    e.g. 'taluy/base_link' or 'taluy_mini/base_link'
    """
    global _base_link_cache

    if _base_link_cache is None:
        if not has_param(get_node(), "~base_link"):
            get_node().get_logger().warning(
                "~base_link param not set, falling back to default frame 'taluy/base_link'"
            )
        _base_link_cache = get_param(get_node(), "~base_link", "taluy/base_link")
        get_node().get_logger().info(
            "Base link frame set to: {}".format(_base_link_cache)
        )

    return _base_link_cache
