from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import PlanPath, PlanPathRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from geometry_msgs.msg import PoseStamped
from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    ExecutePathState,
    ClearObjectMapState,
    SearchForPropState,
    AlignFrame,
    SetPlanState,
    SetPlanningNotActive,
    DynamicPathState,
    SetDetectionFocusState,
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.initialize import DelayState, OdometryEnableState, ResetOdometryState


class PublishSlalomWaypointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "publish_slalom_waypoints", Trigger, request=TriggerRequest()
        )


class NavigateThroughSlalomState(smach.State):
    def __init__(self, slalom_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_SLALOM_DEPTH",
                SetDepthState(
                    depth=slalom_depth,
                    sleep_duration=rospy.get_param("~set_depth_sleep_duration", 4.0),
                ),
                transitions={
                    "succeeded": "PUBLISH_SLALOM_WAYPOINTS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PUBLISH_SLALOM_WAYPOINTS",
                PublishSlalomWaypointsState(),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_SLALOM_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_SLALOM_ENTRANCE",
                DynamicPathState(
                    plan_target_frame="slalom_entrance",
                ),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_TO_SLALOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_TO_SLALOM",
                SetDetectionFocusState(focus_object="pipe"),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_WP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_WP_1",
                DynamicPathState(
                    plan_target_frame="slalom_waypoint_1",
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_WP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_WP_2",
                DynamicPathState(
                    plan_target_frame="slalom_waypoint_2",
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_WP_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_WP_3",
                DynamicPathState(
                    plan_target_frame="slalom_waypoint_3",
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_EXIT",
                DynamicPathState(
                    plan_target_frame="slalom_exit",
                ),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug("[NavigateThroughSlalomState] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
