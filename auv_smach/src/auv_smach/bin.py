from auv_smach.tf_utils import get_tf_buffer, get_base_link
from .initialize import *
import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
import tf2_ros
from auv_common_lib.transform import lookup_fresh_transform

from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    AlignFrame,
    SetDetectionState,
    DynamicPathState,
    SetDetectionFocusState,
    DropBallState,
    SetDetectionState,
    SetDetectionFocusBottomState,
    DynamicPathWithTransformCheck,
    AlignFrameWithTransformCheck,
)

from auv_smach.initialize import DelayState
from auv_smach.acoustic import AcousticTransmitter
from std_msgs.msg import Float32


class BallDropperSetAngleState(smach.State):
    """
    for real life gripper).
    """

    def __init__(self, angle_value: int):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.pub = rospy.Publisher(
            "/taluy/actuators/ball_dropper/set_angle", Float32, queue_size=1
        )
        self.angle_value = angle_value

    def execute(self, userdata) -> str:
        try:
            msg = Float32()
            msg.data = float(self.angle_value)
            for _ in range(3):
                self.pub.publish(msg)
                rospy.sleep(0.1)
            rospy.loginfo(
                f"[BallDropperSetAngleState] Published angle: {self.angle_value}"
            )
            return "succeeded"
        except Exception as e:
            rospy.logerr(f"[BallDropperSetAngleState] Error: {e}")
            return "aborted"


class BinTransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_bin_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class CheckForDropAreaState(smach.State):
    """Checks which bin drop area frame is available after DynamicPathWithTransformCheck.
    Writes the found frame name to userdata.found_frame.
    Priority order follows target_frames list (index 0 has highest priority)."""

    def __init__(
        self,
        source_frame: str = "odom",
        timeout: float = 2.0,
        target_frames=None,
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = get_tf_buffer()
        self.target_frames = target_frames or ["bin_shark_link", "bin_sawfish_link"]

        rospy.loginfo(
            f"[CheckForDropAreaState] Frame priority: {self.target_frames}"
        )

    def execute(self, userdata) -> str:
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while (rospy.Time.now() - start_time) < self.timeout:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            for frame in self.target_frames:
                try:
                    lookup_fresh_transform(
                        self.tf_buffer,
                        self.source_frame,
                        frame,
                        rospy.Duration(rospy.get_param("~tf_lookup_timeout", 0.2)),
                        rospy.Duration(rospy.get_param("~tf_freshness_threshold", 0.4)),
                    )
                    rospy.loginfo(
                        f"[CheckForDropAreaState] Transform from '{self.source_frame}' to '{frame}' found."
                    )
                    userdata.found_frame = frame
                    return "succeeded"
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    pass

            rate.sleep()

        rospy.logwarn(
            f"[CheckForDropAreaState] Timeout: No drop area transforms found after {self.timeout.to_sec()} seconds."
        )
        return "aborted"


class CheckTargetFrameAvailable(smach.State):
    """Checks if the found_frame matches the primary target frame.
    If yes, transitions to target_found. If not, transitions to need_search."""

    def __init__(self, target_frame: str):
        smach.State.__init__(
            self,
            outcomes=["target_found", "need_search", "preempted"],
            input_keys=["found_frame"],
        )
        self.target_frame = target_frame

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        found = userdata.found_frame
        if found == self.target_frame:
            rospy.loginfo(
                f"[CheckTargetFrameAvailable] Primary target '{self.target_frame}' found."
            )
            return "target_found"
        else:
            rospy.logwarn(
                f"[CheckTargetFrameAvailable] Found '{found}' but need '{self.target_frame}'. Searching..."
            )
            return "need_search"


class BottomSearchServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_bottom_search",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class DisableSearchAndSetFoundState(smach.State):
    def __init__(self, target_frame: str):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.target_frame = target_frame

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        userdata.found_frame = self.target_frame

        try:
            service_client = rospy.ServiceProxy("toggle_bottom_search", SetBool)
            service_client.wait_for_service(timeout=1.0)
            service_client.call(SetBoolRequest(data=False))
            return "succeeded"
        except Exception as e:
            rospy.logwarn(f"[DisableSearchAndSetFoundState] Service call failed: {e}")
            return "succeeded"


class BinSearchSequenceState(smach.StateMachine):
    def __init__(self, base_link: str, target_frame: str):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        with self:
            smach.StateMachine.add(
                "ENABLE_BOTTOM_SEARCH",
                BottomSearchServiceEnableState(True),
                transitions={
                    "succeeded": "ALIGN_SEARCH_FRONT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                }
            )

            smach.StateMachine.add(
                "ALIGN_SEARCH_FRONT",
                AlignFrameWithTransformCheck(
                    source_frame=base_link,
                    target_frame="bin_search_front",
                    transform_source_frame="odom",
                    transform_target_frame=target_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=20.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    transform_timeout=20.0,
                ),
                transitions={
                    "succeeded": "DISABLE_BOTTOM_SEARCH_SUCCESS",
                    "preempted": "DISABLE_BOTTOM_SEARCH_PREEMPTED",
                    "aborted": "ALIGN_SEARCH_LEFT",
                }
            )

            smach.StateMachine.add(
                "ALIGN_SEARCH_LEFT",
                AlignFrameWithTransformCheck(
                    source_frame=base_link,
                    target_frame="bin_search_left",
                    transform_source_frame="odom",
                    transform_target_frame=target_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=20.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    transform_timeout=20.0,
                ),
                transitions={
                    "succeeded": "DISABLE_BOTTOM_SEARCH_SUCCESS",
                    "preempted": "DISABLE_BOTTOM_SEARCH_PREEMPTED",
                    "aborted": "ALIGN_SEARCH_BACK",
                }
            )

            smach.StateMachine.add(
                "ALIGN_SEARCH_BACK",
                AlignFrameWithTransformCheck(
                    source_frame=base_link,
                    target_frame="bin_search_back",
                    transform_source_frame="odom",
                    transform_target_frame=target_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=20.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    transform_timeout=20.0,
                ),
                transitions={
                    "succeeded": "DISABLE_BOTTOM_SEARCH_SUCCESS",
                    "preempted": "DISABLE_BOTTOM_SEARCH_PREEMPTED",
                    "aborted": "ALIGN_SEARCH_RIGHT",
                }
            )

            smach.StateMachine.add(
                "ALIGN_SEARCH_RIGHT",
                AlignFrameWithTransformCheck(
                    source_frame=base_link,
                    target_frame="bin_search_right",
                    transform_source_frame="odom",
                    transform_target_frame=target_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=20.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    transform_timeout=20.0,
                ),
                transitions={
                    "succeeded": "DISABLE_BOTTOM_SEARCH_SUCCESS",
                    "preempted": "DISABLE_BOTTOM_SEARCH_PREEMPTED",
                    "aborted": "DISABLE_BOTTOM_SEARCH_ABORTED",
                }
            )

            smach.StateMachine.add(
                "DISABLE_BOTTOM_SEARCH_SUCCESS",
                DisableSearchAndSetFoundState(target_frame),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "succeeded",
                }
            )

            smach.StateMachine.add(
                "DISABLE_BOTTOM_SEARCH_PREEMPTED",
                BottomSearchServiceEnableState(False),
                transitions={
                    "succeeded": "preempted",
                    "preempted": "preempted",
                    "aborted": "preempted",
                }
            )

            smach.StateMachine.add(
                "DISABLE_BOTTOM_SEARCH_ABORTED",
                BottomSearchServiceEnableState(False),
                transitions={
                    "succeeded": "aborted",
                    "preempted": "preempted",
                    "aborted": "aborted",
                }
            )



class SetAlignToFoundState(smach.State):
    """Reads found_frame from userdata and dynamically creates an AlignFrame
    to align the ball_dropper source frame to the found target frame."""

    def __init__(
        self,
        source_frame: str = None,
        dist_threshold: float = 0.1,
        yaw_threshold: float = 0.1,
        confirm_duration: float = 4.0,
        timeout: float = 60.0,
        cancel_on_success: bool = False,
        keep_orientation: bool = True,
        max_linear_velocity: float = 0.1,
        max_angular_velocity: float = 0.1,
    ):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=["found_frame"],
        )
        if source_frame is None:
            source_frame = f"{get_base_link()}/ball_dropper_1_link"
        self.source_frame = source_frame
        self.dist_threshold = dist_threshold
        self.yaw_threshold = yaw_threshold
        self.confirm_duration = confirm_duration
        self.timeout = timeout
        self.cancel_on_success = cancel_on_success
        self.keep_orientation = keep_orientation
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        if "found_frame" not in userdata or not userdata.found_frame:
            rospy.logerr("[SetAlignToFoundState] No found_frame in userdata")
            return "aborted"

        target_frame = userdata.found_frame
        rospy.loginfo(f"[SetAlignToFoundState] Aligning to '{target_frame}'")

        align_state = AlignFrame(
            source_frame=self.source_frame,
            target_frame=target_frame,
            dist_threshold=self.dist_threshold,
            yaw_threshold=self.yaw_threshold,
            confirm_duration=self.confirm_duration,
            timeout=self.timeout,
            cancel_on_success=self.cancel_on_success,
            keep_orientation=self.keep_orientation,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
        )
        return align_state.execute(userdata)


###############################################################################
# BinSecondTrialState - State machine for managing the second trial attempt
# for the bin task. Handles path planning, execution, and verification of the second trial process.
###############################################################################

class BinSecondTrialState(smach.StateMachine):
    def __init__(
        self,
        bin_front_look_depth,
        bin_bottom_look_depth,
        target_frames=None,
    ):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        self.base_link = get_base_link()
        self.bin_search_frame = bin_search_frame

        with self:
            smach.StateMachine.add(
                "SET_SECOND_TRIAL_SEARCH_DEPTH",
                SetDepthState(depth=bin_front_look_depth),
                transitions={
                    "succeeded": "ALIGN_TO_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SECOND_TRIAL",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="bin_second_trial",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_AFTER_SECOND_TRIAL_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
           )
###############################################################################
# BinTaskState - Main state for managing the bin task. Handles the complete
# process including frame management, depth control, path planning, and ball dropping operations.
###############################################################################


class BinTaskState(smach.State):
    def __init__(
        self,
        bin_front_look_depth,
        bin_bottom_look_depth,
        target_frames=None,
        bin_exit_angle=0.0,
        bin_search_frame="bin_whole_link",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()
        self.bin_search_frame = bin_search_frame

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "OPEN_FRONT_CAMERA",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "FOCUS_ON_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_BIN",
                SetDetectionFocusState(focus_object="bin"),
                transitions={
                    "succeeded": "SET_BIN_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DEPTH",
                SetDepthState(depth=bin_front_look_depth),
                transitions={
                    "succeeded": "FIND_AND_AIM_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_BIN",
                SearchForPropState(
                    look_at_frame=self.bin_search_frame,  #should entegrate kde frame here
                    alignment_frame="bin_search",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ENABLE_BIN_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_ENABLE_BIN_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ENABLE_BIN_FRAME_PUBLISHER",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame="bin_close_approach",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_CLOSE_APPROACH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="bin_close_approach",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "SET_BIN_DROP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DROP_DEPTH",
                SetDepthState(depth=bin_bottom_look_depth),
                transitions={
                    "succeeded": "ENABLE_BOTTOM_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BOTTOM_DETECTION",
                SetDetectionState(camera_name="bottom", enable=True),
                transitions={
                    "succeeded": "FOCUS_ON_BIN_BOTTOM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_BIN_BOTTOM",
                SetDetectionFocusBottomState(focus_object="bin"),
                transitions={
                    "succeeded": "DYNAMIC_PATH_WITH_TRANSFORM_CHECK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_WITH_TRANSFORM_CHECK",
                DynamicPathWithTransformCheck(
                    plan_target_frame="bin_far_trial",
                    transform_source_frame="odom",
                    allow_mutli_check_goal=True,
                    transform_target_frame=target_frames,
                    max_linear_velocity=0.2,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=2.0,
                    target_frames=target_frames,
                ),
                transitions={
                    "succeeded": "CHECK_TARGET_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_TARGET_FRAME",
                CheckTargetFrameAvailable(
                    target_frame=target_frames[0],
                ),
                transitions={
                    "target_found": "ALIGN_TO_PRIMARY_TARGET",
                    "need_search": "SEARCH_FOR_TARGET",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_TARGET",
                BinSearchSequenceState(
                    base_link=self.base_link,
                    target_frame=target_frames[0],
                ),
                transitions={
                    "succeeded": "ALIGN_TO_PRIMARY_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_PRIMARY_TARGET",
                SetAlignToFoundState(
                    source_frame=self.base_link + "/ball_dropper_1_link",
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "SET_BALL_DROP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BALL_DROP_DEPTH",
                SetDepthState(depth=-0.4),
                transitions={
                    "succeeded": "DROP_BALL_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_1",
                BallDropperSetAngleState(angle_value=50.0),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_1",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "ALLIGN_TO_SECOND_BASKET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALLIGN_TO_SECOND_BASKET",
                AlignFrame(
                    source_frame=self.base_link+"/ball_dropper_2_link",
                    target_frame=target_frames[0]+"_0",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DROP_BALL_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_2",
                BallDropperSetAngleState(angle_value=-50.0),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_2",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "BALL_DROP_DEFAULT_POSE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "BALL_DROP_DEFAULT_POSE",
                BallDropperSetAngleState(angle_value=0.0),
                transitions={
                    "succeeded": "ALIGN_TO_BIN_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BIN_EXIT",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="bin_exit",
                    angle_offset=bin_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "TRANSMIT_ACOUSTIC_3",
                    "preempted": "preempted",
                    "aborted": "CANCEL_ALIGN_CONTROLLER",
                },
            )
            smach.StateMachine.add(
                "TRANSMIT_ACOUSTIC_3",
                AcousticTransmitter(acoustic_data=3),
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
            # smach.StateMachine.add(
            #     "BIN_SECOND_TRIAL",
            #     BinSecondTrialState(
            #         bin_front_look_depth,
            #         bin_bottom_look_depth,
            #         target_frames,
            #     ),
            #     transitions={
            #         "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
            #         "preempted": "preempted",
            #         "aborted": "DROP_BALL_1",
            #     },
            # )

    def execute(self, userdata):
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
