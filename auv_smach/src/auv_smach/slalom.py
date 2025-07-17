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
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.initialize import DelayState, OdometryEnableState, ResetOdometryState


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_slalom_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class NavigateThroughSlalomState(smach.State):
    def __init__(self, slalom_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.sim_mode = rospy.get_param("~sim", False)

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
                    "succeeded": "ENABLE_SLALOM_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_SLALOM_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "START_PLANNING_TO_SLALOM_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "START_PLANNING_TO_SLALOM_ENTRANCE",
                SetPlanState(target_frame="slalom_entrance"),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="dynamic_target"
                ),
                transitions={
                    "succeeded": "EXECUTE_SLALOM_PATH_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_SLALOM_PATH_ENTRANCE",
                ExecutePathState(),
                transitions={
                    "succeeded": "STOP_PLANNING",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "STOP_PLANNING",
                SetPlanningNotActive(),
                transitions={
                    "succeeded": "ALIGN_WP_1",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_WP_1",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_wp_1",
                    dist_threshold=0.2,
                    yaw_threshold=0.1,
                    timeout=7.0,
                ),
                transitions={
                    "succeeded": "ALIGN_WP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_WP_2",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_wp_2",
                    dist_threshold=0.2,
                    yaw_threshold=0.1,
                    timeout=7.0,
                ),
                transitions={
                    "succeeded": "ALIGN_WP_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_WP_3",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_wp_3",
                    dist_threshold=0.2,
                    yaw_threshold=0.1,
                    timeout=7.0,
                ),
                transitions={
                    "succeeded": "ALIGN_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_EXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="slalom_exit",
                    dist_threshold=0.2,
                    yaw_threshold=0.1,
                    timeout=7.0,
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
