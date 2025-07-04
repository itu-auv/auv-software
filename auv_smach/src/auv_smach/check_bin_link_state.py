import smach
import rospy
import tf2_ros


class CheckBinLinkState(smach.State):
    def __init__(
        self,
        source_frame: str = "odom",
        timeout: float = 2.0,
        target_frame: str = "bin/blue_link",
    ):
        smach.State.__init__(
            self,
            outcomes=["found", "not_found", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_frame = target_frame

    def execute(self, userdata):
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()

        while not self.preempt_requested():
            if rospy.Time.now() - start_time > self.timeout:
                return "not_found"

            try:
                if self.tf_buffer.can_transform(
                    self.source_frame,
                    self.target_frame,
                    rospy.Time(0),
                    rospy.Duration(0.1),
                ):
                    rospy.loginfo(
                        f"[CheckBinLinkState] Transform from '{self.source_frame}' to '{self.target_frame}' found."
                    )
                    userdata.found_frame = self.target_frame
                    return "found"
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue
            rate.sleep()

        return "preempted"
