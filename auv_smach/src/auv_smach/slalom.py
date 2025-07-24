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
            self, "publish_slalom_waypoints", Trigger, request=TriggerRequest()
        )


class SlalomDetectionToWAYpointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "slalom_detection_to_waypoints", Trigger, request=TriggerRequest()
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
                    "succeeded": "SET_DECTION_FOCUS_TO_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DECTION_FOCUS_TO_PIPE",
                SetDetectionFocusState(focus_object="pipe"),
                transitions={
                    "succeeded": "LOOK_AT_RED_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_RED_PIPE",
                SearchForPropState(
                    look_at_frame="slalom_red_pipe_candidate",
                    alignment_frame="red_pipe_search_frame",
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "FOCUS_TO_NONE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_TO_NONE",
                SetDetectionFocusState(focus_object="none"),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_SLALOM_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_SLALOM_ENTRANCE",
                DynamicPathState(
                    plan_target_frame="slalom_entrance_red",
                ),
                transitions={
                    "succeeded": "WAIT_FOR_SLALOM_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SLALOM_DETECTION",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "RED_PIPE_RELALOCATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RED_PIPE_RELALOCATE",
                SetDetectionFocusState(focus_object="pipe"),
                transitions={
                    "succeeded": "RED_PIPE_RELALOCATE_SERVICE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RED_PIPE_RELALOCATE_SERVICE",
                SlalomDetectionToWAYpointsState(),  # This service will handle the detection and reallocation of the red pipe
                transitions={
                    "succeeded": "DELAY_FOR_RED_PIPE_RELALOCATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DELAY_FOR_RED_PIPE_RELALOCATE",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "LOOK_LEFT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_LEFT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_entrance_red",
                    angle_offset=0.5,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.2,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.15,
                ),
                transitions={
                    "succeeded": "LOOK_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_RIGHT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_entrance_red",
                    angle_offset=-0.5,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.2,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.15,
                ),
                transitions={
                    "succeeded": "LOOK_AT_RED_PIPE_FOR_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_RED_PIPE_FOR_TRAJECTORY",
                SearchForPropState(
                    look_at_frame="red_pipe_link",
                    alignment_frame="red_pipe_search_frame",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "TO_NONE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TO_NONE",
                SetDetectionFocusState(focus_object="none"),
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
                    max_linear_velocity=0.4,
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
                    max_linear_velocity=0.12,
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
                    max_linear_velocity=0.12,
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
                    "succeeded": "ALIGN_TO_SLALOM_eXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SLALOM_eXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_exit",
                    confirm_duration=0.0,
                    angle_offset=-1.5,
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
