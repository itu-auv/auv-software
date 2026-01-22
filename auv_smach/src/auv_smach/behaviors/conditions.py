import py_trees
import rospy
import tf2_ros


class IsTransformAvailable(py_trees.behaviour.Behaviour):
    """
    Checks if a TF transform is available between two frames.
    Returns SUCCESS if transform exists, FAILURE otherwise.
    """

    def __init__(
        self,
        name: str,
        source_frame: str,
        target_frame: str,
        timeout: float = 0.05,
    ):
        super().__init__(name)
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.timeout = timeout

    def setup(self, **kwargs):
        """Setup TF buffer and listener."""
        try:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            rospy.logdebug(f"[{self.name}] Setup complete")
            return True
        except Exception as e:
            rospy.logerr(f"[{self.name}] Setup error: {e}")
            return False

    def update(self):
        """Check if the transform is available."""
        try:
            can_transform = self.tf_buffer.can_transform(
                self.source_frame,
                self.target_frame,
                rospy.Time(0),
                rospy.Duration(self.timeout),
            )

            if can_transform:
                rospy.logdebug(
                    f"[{self.name}] Transform available: {self.source_frame} -> {self.target_frame}"
                )
                return py_trees.common.Status.SUCCESS
            else:
                rospy.logdebug(
                    f"[{self.name}] Transform NOT available: {self.source_frame} -> {self.target_frame}"
                )
                return py_trees.common.Status.FAILURE

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logdebug(f"[{self.name}] TF error: {e}")
            return py_trees.common.Status.FAILURE
