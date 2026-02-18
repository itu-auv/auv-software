from auv_smach.tf_utils import get_tf_buffer
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
from auv_smach.initialize import DelayState

from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.initialize import DelayState, OdometryEnableState, ResetOdometryState


class PublishSlalomWaypointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_slalom_waypoints",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class StopSlalomWaypointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_slalom_waypoints",
            SetBool,
            request=SetBoolRequest(data=False),
        )


class NavigateThroughSlalomState(smach.State):
    def __init__(
        self,
        slalom_depth: float,
        slalom_exit_angle: float = 0.0,
        slalom_mode: str = "close",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.slalom_exit_angle = slalom_exit_angle
        self.slalom_mode = slalom_mode

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

            succeeded_transition = (
                "SET_DETECTION_FOCUS_TO_SLALOM"
                if self.slalom_mode == "close"
                else "DYNAMIC_PATH_TO_SLALOM_ENTRANCE"
            )
            smach.StateMachine.add(
                "PUBLISH_SLALOM_WAYPOINTS",
                PublishSlalomWaypointsState(),
                transitions={
                    "succeeded": succeeded_transition,
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
                    "succeeded": "ALIGN_TO_SLALOM_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SLALOM_ENTRANCE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_entrance",
                    confirm_duration=1.0,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_SLALOM_ENTRANCE_BACKED",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_SLALOM_ENTRANCE_BACKED",
                DynamicPathState(
                    plan_target_frame="slalom_entrance_backed",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_SLALOM_ENTRANCE_BACKED",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SLALOM_ENTRANCE_BACKED",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_entrance_backed",
                    confirm_duration=3.0,
                ),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_TO_SLALOM",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_TO_SLALOM",
                SetDetectionFocusState(focus_object="pipe"),
                transitions={
                    "succeeded": "SEARCH_FOR_RED_PIPE",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_RED_PIPE",
                SearchForPropState(
                    look_at_frame="red_pipe_link",
                    alignment_frame="search_for_red_pipe",
                    full_rotation=False,
                    timeout=30.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "LOOK_LEFT",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "LOOK_LEFT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="search_for_red_pipe",
                    angle_offset=0.5,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.2,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                ),
                transitions={
                    "succeeded": "LOOK_RIGHT",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "LOOK_RIGHT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="search_for_red_pipe",
                    angle_offset=-0.5,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.2,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                ),
                transitions={
                    "succeeded": "LOCK_ON_WAYPOINT_1",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "LOCK_ON_WAYPOINT_1",
                SearchForPropState(
                    look_at_frame="slalom_waypoint_1",
                    alignment_frame="look_at_waypoint_1",
                    full_rotation=False,
                    timeout=30.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_TO_NONE",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_TO_NONE",
                SetDetectionFocusState(focus_object="none"),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_WP_1",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_WP_1",
                DynamicPathState(
                    plan_target_frame="slalom_waypoint_1",
                    max_linear_velocity=0.4,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_WP_1_INTER",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_WP_1_INTER",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_waypoint_1_inter",
                    dist_threshold=0.2,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=10.0,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_WP_2",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_WP_2",
                DynamicPathState(
                    plan_target_frame="slalom_waypoint_2",
                    max_linear_velocity=0.12,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_WP_2_INTER",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_WP_2_INTER",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_waypoint_2_inter",
                    dist_threshold=0.2,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=10.0,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_WP_3",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_WP_3",
                DynamicPathState(
                    plan_target_frame="slalom_waypoint_3",
                    max_linear_velocity=0.12,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_EXIT",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_EXIT",
                DynamicPathState(
                    plan_target_frame="slalom_exit",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_SLALOM_EXIT",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SLALOM_EXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_exit",
                    confirm_duration=0.0,
                    angle_offset=self.slalom_exit_angle,
                ),
                transitions={
                    "succeeded": "STOP_SLALOM_WAYPOINTS",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "STOP_SLALOM_WAYPOINTS",
                StopSlalomWaypointsState(),
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
