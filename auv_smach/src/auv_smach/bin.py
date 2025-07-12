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
)

from auv_smach.initialize import DelayState

from auv_navigation.path_planning.path_planners import PathPlanners


class DropBallState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "ball_dropper/drop",
            Trigger,
            request=TriggerRequest(),
        )


class CheckForDropAreaState(smach.State):
    def __init__(self, source_frame: str = "odom", timeout: float = 2.0):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_frames = ["bin/blue_link", "bin/red_link"]

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
                        f"[CheckForDropAreaState] Transform from '{self.source_frame}' to '{frame}' found."
                    )
                    userdata.found_frame = frame
                    return "succeeded"

            rate.sleep()

        rospy.logwarn(
            f"[CheckForDropAreaState] Timeout: No drop area transforms found after {self.timeout.to_sec()} seconds."
        )
        return "aborted"


class SetAlignToFoundState(smach.State):
    def __init__(self, source_frame: str):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"], input_keys=["found_frame"]
        )
        self.source_frame = source_frame

    def execute(self, userdata):
        if self.preempt_requested():
            return "preempted"

        if "found_frame" not in userdata or not userdata.found_frame:
            rospy.logerr("[SetAlignToFoundState] No found_frame in userdata")
            return "aborted"

        rospy.loginfo(
            f"[SetAlignToFoundState] Setting align target to {userdata.found_frame}"
        )

        align_state = SetAlignControllerTargetState(
            source_frame=self.source_frame,
            target_frame=userdata.found_frame,
            keep_orientation=True,
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
    def __init__(self, tf_buffer, bin_front_look_depth, bin_bottom_look_depth):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        self.tf_buffer = tf_buffer
        self.bin_front_look_depth = bin_front_look_depth
        self.bin_bottom_look_depth = bin_bottom_look_depth

        with self:
            smach.StateMachine.add(
                "SET_SECOND_TRIAL_SEARCH_DEPTH",
                SetDepthState(depth=bin_front_look_depth, sleep_duration=3.0),
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
                CheckForDropAreaState(source_frame="odom", timeout=1.0),
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
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
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
                SetDepthState(depth=bin_bottom_look_depth, sleep_duration=3.0),
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
                CheckForDropAreaState(source_frame="odom", timeout=2.0),
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
    def __init__(self, bin_front_look_depth, bin_bottom_look_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        with self.state_machine:
            smach.StateMachine.add(
                "SET_BIN_DEPTH",
                SetDepthState(depth=bin_front_look_depth, sleep_duration=3.0),
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
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
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
                DelayState(delay_time=1.0),
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
                    confirm_duration=10.0,
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
                SetDepthState(depth=bin_bottom_look_depth, sleep_duration=3.0),
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
                    "succeeded": "ALIGN_TO_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BIN",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="bin_whole_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=60.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_AFTER_BIN_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_AFTER_BIN_ALIGNMENT",
                CheckForDropAreaState(source_frame="odom", timeout=1.0),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "ALIGN_TO_BIN_ESTIMATED",
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
                CheckForDropAreaState(source_frame="odom", timeout=1.0),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "ALIGN_TO_FAR_TRIAL",
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
                CheckForDropAreaState(source_frame="odom", timeout=1.0),
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
                SetAlignToFoundState(source_frame="taluy/base_link/ball_dropper_link"),
                transitions={
                    "succeeded": "WAIT_FOR_DROP_AREA_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_DROP_AREA_ALIGNMENT",
                DelayState(delay_time=15.0),
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
                    "succeeded": "DISABLE_BOTTOM_CAMERA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BOTTOM_CAMERA",
                SetDetectionState(camera_name="bottom", enable=False),
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
                    self.tf_buffer, bin_front_look_depth, bin_bottom_look_depth
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
