from auv_smach.tf_utils import get_tf_buffer
from .initialize import *
import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
import tf2_ros

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
)

from auv_smach.initialize import DelayState
from auv_smach.acoustic import AcousticTransmitter


class CheckForDropAreaState(smach.State):
    def __init__(
        self,
        source_frame: str = "odom",
        timeout: float = 2.0,
        target_selection: str = "shark",
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = get_tf_buffer()
        self.target_selection = target_selection
        # Set frame order based on target_selection
        if self.target_selection == "shark":
            self.target_frames = ["bin_shark_link", "bin_sawfish_link"]
        elif self.target_selection == "sawfish":
            self.target_frames = ["bin_sawfish_link", "bin_shark_link"]
        else:
            self.target_frames = ["bin_shark_link", "bin_sawfish_link"]

        rospy.loginfo(
            f"[CheckForDropAreaState] Target selection: {self.target_selection}, Frame priority: {self.target_frames}"
        )

    def execute(self, userdata) -> str:
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while (rospy.Time.now() - start_time) < self.timeout:
            if self.preempt_requested():
                return "preempted"

            # Check for both blue and red bin frames
            for frame in self.target_frames:
                if self.tf_buffer.can_transform(
                    self.source_frame, frame, rospy.Time(0), self.timeout
                ):
                    rospy.loginfo(
                        f"[CheckForDropAreaState] Target selection '{self.target_selection}': Transform from '{self.source_frame}' to '{frame}' found."
                    )
                    userdata.found_frame = frame
                    return "succeeded"

            rate.sleep()

        rospy.logwarn(
            f"[CheckForDropAreaState] Timeout: No drop area transforms found after {self.timeout.to_sec()} seconds."
        )
        return "aborted"


class SetAlignToFoundState(smach.State):
    def __init__(
        self,
        source_frame: str = "taluy/base_link/ball_dropper_link",
        dist_threshold: float = 0.05,
        yaw_threshold: float = 0.1,
        confirm_duration: float = 5.0,
        timeout: float = 0.0,
        cancel_on_success: bool = False,
        keep_orientation: bool = False,
    ):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.dist_threshold = dist_threshold
        self.yaw_threshold = yaw_threshold
        self.confirm_duration = confirm_duration
        self.timeout = timeout
        self.cancel_on_success = cancel_on_success
        self.keep_orientation = keep_orientation

    def execute(self, userdata):
        if self.preempt_requested():
            return "preempted"

        if "found_frame" not in userdata or not userdata.found_frame:
            rospy.logerr("[SetAlignToFoundState] No found_frame in userdata")
            return "aborted"

        target_frame = userdata.found_frame
        rospy.loginfo(f"[SetAlignToFoundState] Setting align target to {target_frame}")

        align_state = AlignFrame(
            source_frame=self.source_frame,
            target_frame=target_frame,
            dist_threshold=self.dist_threshold,
            yaw_threshold=self.yaw_threshold,
            confirm_duration=self.confirm_duration,
            timeout=self.timeout,
            cancel_on_success=self.cancel_on_success,
            keep_orientation=self.keep_orientation,
        )
        return align_state.execute(userdata)


class BinTransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_bin_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


###############################################################################
# BinSecondTrialState - State machine for managing the second trial attempt
# for the bin task. Handles path planning, execution, and verification of the second trial process.
###############################################################################


class BinSecondTrialState(smach.StateMachine):
    def __init__(
        self,
        bin_front_look_depth,
        bin_bottom_look_depth,
        target_selection="shark",
    ):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )

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
                    source_frame="taluy/base_link",
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
            smach.StateMachine.add(
                "CHECK_DROP_AREA_AFTER_SECOND_TRIAL_ALIGNMENT",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=1.0,
                    target_selection=target_selection,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "ENABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                BinTransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "CANCEL_ALIGN_FOR_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_FOR_SEARCH",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "FIND_AND_AIM_BIN_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_BIN_SECOND_TRIAL",
                SearchForPropState(
                    look_at_frame="bin_whole_link",
                    alignment_frame="bin_search",
                    full_rotation=False,
                    timeout=30.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                BinTransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "SET_SECOND_TRIAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_SECOND_TRIAL_DEPTH",
                SetDepthState(depth=bin_bottom_look_depth),
                transitions={
                    "succeeded": "ALIGN_TO_SECOND_FAR_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SECOND_FAR_TRIAL",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="bin_far_trial",
                    angle_offset=0.0,
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_AFTER_SECOND_FAR_TRIAL_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_AFTER_SECOND_FAR_TRIAL_ALIGNMENT",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=2.0,
                    target_selection=target_selection,
                ),
                transitions={
                    "succeeded": "succeeded",
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
        target_selection="shark",
        bin_exit_angle=0.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

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
                    look_at_frame="bin_whole_link",
                    alignment_frame="bin_search",
                    full_rotation=False,
                    timeout=30.0,
                    source_frame="taluy/base_link",
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
                    source_frame="taluy/base_link",
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
                    "succeeded": "DYNAMIC_PATH_TO_BIN_WHOLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_BIN_WHOLE",
                DynamicPathState(
                    plan_target_frame="bin_whole_estimated",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BIN_ESTIMATED",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BIN_ESTIMATED",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="bin_whole_estimated",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=60.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_AFTER_BIN_ESTIMATED_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_AFTER_BIN_ESTIMATED_ALIGNMENT",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=1.0,
                    target_selection=target_selection,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "DYNAMIC_PATH_TO_FAR_TRIAL",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_FAR_TRIAL",
                DynamicPathState(
                    plan_target_frame="bin_far_trial",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_FAR_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_FAR_TRIAL",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="bin_far_trial",
                    angle_offset=0.0,
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=60.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_AFTER_FAR_TRIAL_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_AFTER_FAR_TRIAL_ALIGNMENT",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=1.0,
                    target_selection=target_selection,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "BIN_SECOND_TRIAL",
                },
            )
            ############################
            # ALIGN TO FOUND DROP AREA #
            ############################
            smach.StateMachine.add(
                "SET_ALIGN_TO_FOUND_DROP_AREA",
                SetAlignToFoundState(
                    source_frame="taluy/base_link/ball_dropper_link",
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_FOR_FINAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_FOR_FINAL",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=1.0,
                    target_selection=target_selection,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA_FINAL",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "BIN_SECOND_TRIAL",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_TO_FOUND_DROP_AREA_FINAL",
                SetAlignToFoundState(
                    source_frame="taluy/base_link/ball_dropper_link",
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=20.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_NONE_FOR_BOTTOM_CAM",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_NONE_FOR_BOTTOM_CAM",
                SetDetectionState(camera_name="bottom", enable=False),
                transitions={
                    "succeeded": "SET_BALL_DROP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BALL_DROP_DEPTH",
                SetDepthState(depth=-1.25),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA_AFTER_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_TO_FOUND_DROP_AREA_AFTER_DEPTH",
                SetAlignToFoundState(
                    source_frame="taluy/base_link/ball_dropper_link",
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=20.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "DROP_BALL_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_1",
                DropBallState(),
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
                    "succeeded": "DROP_BALL_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_2",
                DropBallState(),
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
                    "succeeded": "ELEVATE_BEFORE_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ELEVATE_BEFORE_EXIT",
                SetDepthState(depth=-1.0),
                transitions={
                    "succeeded": "ALIGN_TO_BIN_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BIN_EXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
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
            smach.StateMachine.add(
                "BIN_SECOND_TRIAL",
                BinSecondTrialState(
                    bin_front_look_depth,
                    bin_bottom_look_depth,
                    target_selection,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "DROP_BALL_1",
                },
            )

    def execute(self, userdata):
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
