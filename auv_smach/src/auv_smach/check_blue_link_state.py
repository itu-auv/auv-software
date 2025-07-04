import smach
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


class CheckBlueLinkState(smach.State):
    def __init__(self, source_frame: str = "odom", timeout: float = 2.0):
        smach.State.__init__(
            self,
            outcomes=["found_blue", "not_found", "preempted", "aborted"],
            output_keys=["blue_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.blue_frame = "bin/blue_link"

    def execute(self, userdata):
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()

        while not self.preempt_requested():
            if rospy.Time.now() - start_time > self.timeout:
                return "not_found"

            try:
                if self.tf_buffer.can_transform(
                    self.source_frame,
                    self.blue_frame,
                    rospy.Time(0),
                    rospy.Duration(0.1),
                ):
                    rospy.loginfo(
                        f"[CheckBlueLinkState] Transform from '{self.source_frame}' to '{self.blue_frame}' found."
                    )
                    userdata.blue_frame = self.blue_frame
                    return "found_blue"
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue
            rate.sleep()

        return "preempted"
