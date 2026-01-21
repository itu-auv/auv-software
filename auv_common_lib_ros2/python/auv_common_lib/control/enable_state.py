import rospy
import std_msgs


class ControlEnableHandler:
    def __init__(self, timeout_duration):
        self.enable = False
        self.timeout_duration = rospy.Duration(timeout_duration)
        self.last_enable_time = rospy.Time.now()
        rospy.Subscriber(
            "enable", std_msgs.msg.Bool, self._enable_callback, tcp_nodelay=True
        )

    def _enable_callback(self, msg):
        self.enable = msg.data
        if self.enable:
            self.last_enable_time = rospy.Time.now()

    def is_enabled(self) -> bool:
        current_time = rospy.Time.now()
        is_timeouted = (
            current_time - self.last_enable_time
        ).to_sec() >= self.timeout_duration.to_sec()

        return self.enable and not is_timeouted
