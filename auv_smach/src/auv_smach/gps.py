from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import SetBool, SetBoolRequest
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


class ToggleGpsServiceState(smach_ros.ServiceState):
    def __init__(self, enable=True):
        smach_ros.ServiceState.__init__(
            self,
            "gps_target_frame_trigger",
            SetBool,
            request=SetBoolRequest(data=enable),
        )


class NavigateToGpsTargetState(smach.State):
    def __init__(
        self,
        gps_depth: float,
        gps_target_frame: str = "gps_target",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.gps_target_frame = gps_target_frame

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "TOGGLE_GPS_SERVICE",
                ToggleGpsServiceState(enable=True),
                transitions={
                    "succeeded": "WAIT_FOR_SERVICE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SERVICE",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "DISABLE_GPS_SERVICE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_GPS_SERVICE",
                ToggleGpsServiceState(enable=False),
                transitions={
                    "succeeded": "SET_GPS_DEPTH",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "SET_GPS_DEPTH",
                SetDepthState(
                    depth=gps_depth,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_GPS_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_GPS_TARGET",
                DynamicPathState(
                    plan_target_frame=self.gps_target_frame,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_GPS_TARGET",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_GPS_TARGET",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=self.gps_target_frame,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "SET_FINAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "SET_FINAL_DEPTH",
                SetDepthState(
                    depth=0.0,
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
        rospy.logdebug("[NavigateToGpsTargetState] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
