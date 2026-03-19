from auv_smach.tf_utils import get_tf_buffer, get_base_link
from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_navigation.path_planning.path_planners import PathPlanners
from auv_smach.common import (
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    AlignFrame,
    DynamicPathState,
    SetDetectionFocusState,
    LookAroundState,
)

from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.roll import TwoRollState, TwoYawState
from auv_smach.coin_flip import CoinFlipState
from auv_smach.acoustic import AcousticTransmitter
from std_msgs.msg import Bool


class MonitorVisibilityState(smach.State):
    def __init__(self, prop_name, timeout=3.0):
        smach.State.__init__(self, outcomes=["target_lost", "preempted"])
        self.prop_name = prop_name
        self.timeout = timeout
        self.last_seen_time = None
        self.subscriber = None

        namespace = rospy.get_namespace().strip("/")
        if not namespace:
            namespace = "taluy"  # fallback
        self.topic_name = f"/{namespace}/vision/front/{prop_name}_is_inside_image"

    def is_visible_cb(self, msg):
        if msg.data:
            self.last_seen_time = rospy.Time.now()

    def execute(self, userdata):
        self.last_seen_time = rospy.Time.now()
        self.subscriber = rospy.Subscriber(self.topic_name, Bool, self.is_visible_cb)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.preempt_requested():
                if self.subscriber:
                    self.subscriber.unregister()
                self.service_preempt()
                return "preempted"

            if self.last_seen_time is not None:
                if (rospy.Time.now() - self.last_seen_time).to_sec() > self.timeout:
                    rospy.logwarn(
                        f"[MonitorVisibility] Target has not been seen for {self.timeout} seconds! ({self.topic_name})"
                    )
                    if self.subscriber:
                        self.subscriber.unregister()
                    return "target_lost"

            rate.sleep()

        if self.subscriber:
            self.subscriber.unregister()
        return "preempted"


class AlignFrameWithVisibilityCheck(smach.Concurrence):
    """
    Runs the standard AlignFrame state concurrently with a MonitorVisibilityState.
    If the target is not seen for 'lost_timeout' seconds, the alignment is halted and returns 'target_lost'.
    """

    def __init__(
        self,
        source_frame,
        target_frame,
        prop_name=None,
        lost_timeout=3.0,
        **align_kwargs,
    ):
        smach.Concurrence.__init__(
            self,
            outcomes=["succeeded", "target_lost", "preempted", "aborted"],
            default_outcome="aborted",
            child_termination_cb=lambda so: True,  # Stop the other state when one finishes
            outcome_map={
                "succeeded": {"ALIGN": "succeeded"},
                "target_lost": {"MONITOR": "target_lost"},
                "aborted": {"ALIGN": "aborted"},
            },
        )

        # Use target_frame name if prop_name is not provided
        if prop_name is None:
            prop_name = target_frame

        with self:
            smach.Concurrence.add(
                "ALIGN",
                AlignFrame(
                    source_frame=source_frame, target_frame=target_frame, **align_kwargs
                ),
            )
            smach.Concurrence.add(
                "MONITOR",
                MonitorVisibilityState(prop_name=prop_name, timeout=lost_timeout),
            )


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_gate_single_frame_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PlanGatePathsState(smach.State):
    """State that plans the paths for the gate task"""

    def __init__(self, tf_buffer):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanGatePathsState] Preempt requested")
                return "preempted"

            path_planners = PathPlanners(
                self.tf_buffer
            )  # instance of PathPlanners with tf_buffer
            paths = path_planners.path_for_gate()
            if paths is None:
                return "aborted"

            userdata.planned_paths = paths
            return "succeeded"
        except Exception as e:
            rospy.logerr("[PlanGatePathsState] Error: %s", str(e))
            return "aborted"


class NavigateThroughGateMiniState(smach.State):
    def __init__(
        self,
        gate_depth: float,
        gate_search_depth: float,
        gate_exit_angle: float = 0.0,
        roll_depth: float = -0.8,
        target_animal: str = "gate_shark_link",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.base_link = get_base_link()
        self.roll = rospy.get_param("~roll", True)
        self.yaw = rospy.get_param("~yaw", False)
        self.coin_flip = rospy.get_param("~coin_flip", False)
        self.gate_look_at_frame = "gate_middle_part"
        self.gate_search_frame = "gate_search"
        self.gate_exit_angle = gate_exit_angle
        self.roll_depth = roll_depth
        self.target_animal = target_animal

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_INITIAL_GATE_DEPTH",
                SetDepthState(
                    depth=-0.5,
                ),
                transitions={
                    "succeeded": "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": (
                        "COIN_FLIP_STATE"
                        if self.coin_flip
                        else "SET_DETECTION_FOCUS_GATE"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "COIN_FLIP_STATE",
                CoinFlipState(),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_GATE",
                SetDetectionFocusState(focus_object="gate"),
                transitions={
                    "succeeded": "SET_ROLL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ROLL_DEPTH",
                SetDepthState(
                    depth=self.roll_depth,
                ),
                transitions={
                    "succeeded": "FIND_AND_AIM_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_GATE",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": (
                        "CALIFORNIA_ROLL"
                        if self.roll
                        else (
                            "TWO_YAW_STATE" if self.yaw else "SET_GATE_TRAJECTORY_DEPTH"
                        )
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CALIFORNIA_ROLL",
                TwoRollState(
                    roll_torque=50.0, gate_look_at_frame=self.gate_look_at_frame
                ),
                transitions={
                    "succeeded": "SET_GATE_TRAJECTORY_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TWO_YAW_STATE",
                TwoYawState(yaw_frame=self.gate_search_frame),
                transitions={
                    "succeeded": "SET_GATE_TRAJECTORY_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_TRAJECTORY_DEPTH",
                SetDepthState(depth=gate_search_depth),
                transitions={
                    "succeeded": "LOOK_AT_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_GATE",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame=self.gate_search_frame,
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=-1.0),
                transitions={
                    "succeeded": "ALIGN_FRAME_TO_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_TO_GATE",
                AlignFrameWithVisibilityCheck(
                    source_frame=self.base_link,
                    target_frame="gate_middle_part",
                    prop_name=self.target_animal,
                    lost_timeout=3.0,
                    angle_offset=self.gate_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=10.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "target_lost": "CANCEL_ALIGN_CONTROLLER",
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
        rospy.logdebug(
            "[NavigateThroughGateMiniState] Starting state machine execution."
        )

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
