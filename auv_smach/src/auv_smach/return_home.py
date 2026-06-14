from auv_smach.tf_utils import get_tf_buffer, get_base_link
from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from auv_navigation.path_planning.path_planners import PathPlanners
from geometry_msgs.msg import PoseStamped
from auv_smach.common import (
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    AlignFrame,
    DynamicPathState,
)
from auv_smach.acoustic import AcousticTransmitter


class ReturnHomeState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.base_link = get_base_link()

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "SET_RETURN_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_RETURN_DEPTH",
                SetDepthState(
                    depth=-0.3,
                ),
                transitions={
                    "succeeded": "LOOK_AT_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_ODOM",
                SearchForPropState(
                    look_at_frame="odom",
                    alignment_frame="look_at_odom",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.4,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_ODOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_ODOM",
                DynamicPathState(
                    plan_target_frame="odom",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_ENTRANCE",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="odom",
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[ReturnHome] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
