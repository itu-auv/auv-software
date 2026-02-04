import rospy
import smach
import py_trees
import py_trees_ros
from auv_smach.behaviors.gate_tree import create_gate_tree


class NavigateThroughGateState(smach.State):
    """
    SMACH State wrapper for the Gate Behavior Tree.
    """

    def __init__(
        self,
        gate_depth: float,
        gate_search_depth: float,
        gate_exit_angle: float = 0.0,
        roll_depth: float = -0.8,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.gate_depth = gate_depth
        self.gate_search_depth = gate_search_depth
        self.gate_exit_angle = gate_exit_angle
        self.roll_depth = roll_depth

        # Get ROS parameters for maneuvers
        self.enable_roll = rospy.get_param("~roll", True)
        self.enable_yaw = rospy.get_param("~yaw", False)
        self.enable_coin_flip = rospy.get_param("~coin_flip", False)

        self.tree = None
        self.behaviour_tree = None

    def execute(self, userdata):
        rospy.loginfo("[NavigateThroughGateState] Starting Behavior Tree execution...")

        # 1. Create the Tree with parameters
        self.tree = create_gate_tree(
            gate_depth=self.gate_depth,
            gate_search_depth=self.gate_search_depth,
            roll_depth=self.roll_depth,
            gate_exit_angle=self.gate_exit_angle,
            enable_roll=self.enable_roll,
            enable_yaw=self.enable_yaw,
            enable_coin_flip=self.enable_coin_flip,
        )

        # 2. Initialize the Tree Engine
        self.behaviour_tree = py_trees_ros.trees.BehaviourTree(self.tree)
        self.behaviour_tree.setup(timeout=15.0)

        # 3. Tick Loop
        rate = rospy.Rate(10)  # 10 Hz

        try:
            while not rospy.is_shutdown():
                if self.preempt_requested():
                    self.service_preempt()
                    self.behaviour_tree.interrupt()
                    return "preempted"

                self.behaviour_tree.tick()

                # Check Status
                status = self.tree.status

                if status == py_trees.common.Status.SUCCESS:
                    rospy.loginfo("[NavigateThroughGateState] Behavior Tree succeeded!")
                    return "succeeded"

                if status == py_trees.common.Status.FAILURE:
                    rospy.logerr("[NavigateThroughGateState] Behavior Tree failed!")
                    return "aborted"

                # Optional: Feedback logging
                # py_trees.display.print_ascii_tree(self.tree, show_status=True)

                rate.sleep()

            # While loop exited due to ROS shutdown
            if rospy.is_shutdown():
                rospy.logwarn(
                    "[NavigateThroughGateState] ROS shutdown - task incomplete"
                )
            return "aborted"

        finally:
            # Cleanup tree resources
            if self.behaviour_tree:
                self.behaviour_tree.shutdown()
