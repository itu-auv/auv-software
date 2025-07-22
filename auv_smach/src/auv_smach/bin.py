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


class CheckForDropAreaState(smach.State):
    def __init__(
        self,
        source_frame: str,
        timeout: float,
        target_selection: str,
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Set frame order based on target_selection
        if target_selection == "shark":
            self.target_frames = ["bin_shark_link", "bin_sawfish_link"]
        elif target_selection == "sawfish":
            self.target_frames = ["bin_sawfish_link", "bin_shark_link"]
        else:
            self.target_frames = ["bin_shark_link", "bin_sawfish_link"]

    def execute(self, userdata) -> str:
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while (rospy.Time.now() - start_time) < self.timeout:
            if self.preempt_requested():
                return "preempted"

            # Check for both blue and red bin frames
            for frame in self.target_frames:
                try:
                    if self.tf_buffer.can_transform(
                        self.source_frame, frame, rospy.Time(0), rospy.Duration(0.1)
                    ):
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
                    continue

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


class BinSecondTrialState(smach.StateMachine):
    def __init__(
        self,
        tf_buffer,
    ):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        self.tf_buffer = tf_buffer
        self.target_selection = rospy.get_param("~target_selection", "shark")

        # Get params
        bin_second_trial_params = rospy.get_param("~bin_second_trial", {})
        set_second_trial_search_depth_params = bin_second_trial_params.get(
            "set_second_trial_search_depth", {}
        )
        align_to_second_trial_params = bin_second_trial_params.get(
            "align_to_second_trial", {}
        )
        find_and_aim_bin_params = bin_second_trial_params.get("find_and_aim_bin", {})
        set_second_trial_depth_params = bin_second_trial_params.get(
            "set_second_trial_depth", {}
        )
        align_to_second_far_trial_params = bin_second_trial_params.get(
            "align_to_second_far_trial", {}
        )

        with self:
            smach.StateMachine.add(
                "SET_SECOND_TRIAL_SEARCH_DEPTH",
                SetDepthState(
                    depth=set_second_trial_search_depth_params.get("depth", -1.2),
                    sleep_duration=3.0,
                ),
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
                    angle_offset=align_to_second_trial_params.get("angle_offset", 0.0),
                    dist_threshold=align_to_second_trial_params.get(
                        "dist_threshold", 0.1
                    ),
                    yaw_threshold=align_to_second_trial_params.get(
                        "yaw_threshold", 0.1
                    ),
                    confirm_duration=align_to_second_trial_params.get(
                        "confirm_duration", 2.0
                    ),
                    timeout=60.0,
                    cancel_on_success=align_to_second_trial_params.get(
                        "cancel_on_success", False
                    ),
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
                    timeout=2.0,
                    target_selection=self.target_selection,
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
                    full_rotation=find_and_aim_bin_params.get("full_rotation", False),
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=find_and_aim_bin_params.get("rotation_speed", 0.2),
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
                SetDepthState(
                    depth=set_second_trial_depth_params.get("depth", -1.2),
                    sleep_duration=3.0,
                ),
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
                    angle_offset=align_to_second_far_trial_params.get(
                        "angle_offset", 0.0
                    ),
                    dist_threshold=align_to_second_far_trial_params.get(
                        "dist_threshold", 0.05
                    ),
                    yaw_threshold=align_to_second_far_trial_params.get(
                        "yaw_threshold", 0.1
                    ),
                    confirm_duration=align_to_second_far_trial_params.get(
                        "confirm_duration", 3.0
                    ),
                    timeout=60.0,
                    cancel_on_success=align_to_second_far_trial_params.get(
                        "cancel_on_success", False
                    ),
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_FOUND_SECOND_FAR_TRIAL_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_FOUND_SECOND_FAR_TRIAL_ALIGNMENT",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=2.0,
                    target_selection=self.target_selection,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class BinTaskState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Get params
        bin_task_params = rospy.get_param("~bin_task", {})
        self.target_selection = rospy.get_param("~target_selection", "shark")
        set_bin_depth_params = bin_task_params.get("set_bin_depth", {})
        find_and_aim_bin_params = bin_task_params.get("find_and_aim_bin", {})
        wait_for_enable_bin_frame_publisher_params = bin_task_params.get(
            "wait_for_enable_bin_frame_publisher", {}
        )
        align_to_close_approach_params = bin_task_params.get(
            "align_to_close_approach", {}
        )
        set_bin_drop_depth_params = bin_task_params.get("set_bin_drop_depth", {})
        align_to_bin_whole_params = bin_task_params.get("align_to_bin_whole", {})
        align_to_bin_estimated_params = bin_task_params.get(
            "align_to_bin_estimated", {}
        )
        align_to_far_trial_params = bin_task_params.get("align_to_far_trial", {})
        set_align_to_found_drop_area_params = bin_task_params.get(
            "set_align_to_found_drop_area", {}
        )
        wait_for_aligning_drop_area_params = bin_task_params.get(
            "wait_for_aligning_drop_area", {}
        )
        wait_for_ball_drop_1_params = bin_task_params.get("wait_for_ball_drop_1", {})
        wait_for_ball_drop_2_params = bin_task_params.get("wait_for_ball_drop_2", {})

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
                SetDepthState(
                    depth=set_bin_depth_params.get("depth", -1.2),
                    sleep_duration=3.0,
                ),
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
                    full_rotation=find_and_aim_bin_params.get("full_rotation", False),
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=find_and_aim_bin_params.get("rotation_speed", 0.2),
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
                DelayState(
                    delay_time=wait_for_enable_bin_frame_publisher_params.get(
                        "delay_time", 1.0
                    )
                ),
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
                    angle_offset=align_to_close_approach_params.get(
                        "angle_offset", 0.0
                    ),
                    dist_threshold=align_to_close_approach_params.get(
                        "dist_threshold", 0.1
                    ),
                    yaw_threshold=align_to_close_approach_params.get(
                        "yaw_threshold", 0.1
                    ),
                    confirm_duration=align_to_close_approach_params.get(
                        "confirm_duration", 3.0
                    ),
                    timeout=60.0,
                    cancel_on_success=align_to_close_approach_params.get(
                        "cancel_on_success", False
                    ),
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
                SetDepthState(
                    depth=set_bin_drop_depth_params.get("depth", -0.7),
                    sleep_duration=3.0,
                ),
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
                    plan_target_frame="bin_whole_link",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BIN_WHOLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BIN_WHOLE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="bin_whole_link",
                    angle_offset=align_to_bin_whole_params.get("angle_offset", 0.0),
                    dist_threshold=align_to_bin_whole_params.get("dist_threshold", 0.1),
                    yaw_threshold=align_to_bin_whole_params.get("yaw_threshold", 0.1),
                    confirm_duration=align_to_bin_whole_params.get(
                        "confirm_duration", 1.0
                    ),
                    timeout=60.0,
                    cancel_on_success=align_to_bin_whole_params.get(
                        "cancel_on_success", False
                    ),
                    keep_orientation=align_to_bin_whole_params.get(
                        "keep_orientation", True
                    ),
                ),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_AFTER_BIN_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_AFTER_BIN_ALIGNMENT",
                CheckForDropAreaState(
                    source_frame="odom",
                    timeout=1.0,
                    target_selection=self.target_selection,
                ),
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
                    angle_offset=align_to_bin_estimated_params.get("angle_offset", 0.0),
                    dist_threshold=align_to_bin_estimated_params.get(
                        "dist_threshold", 0.1
                    ),
                    yaw_threshold=align_to_bin_estimated_params.get(
                        "yaw_threshold", 0.1
                    ),
                    confirm_duration=align_to_bin_estimated_params.get(
                        "confirm_duration", 1.0
                    ),
                    timeout=60.0,
                    cancel_on_success=align_to_bin_estimated_params.get(
                        "cancel_on_success", False
                    ),
                    keep_orientation=align_to_bin_estimated_params.get(
                        "keep_orientation", False
                    ),
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
                    timeout=2.0,
                    target_selection=self.target_selection,
                ),
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
                    angle_offset=align_to_far_trial_params.get("angle_offset", 0.0),
                    dist_threshold=align_to_far_trial_params.get(
                        "dist_threshold", 0.05
                    ),
                    yaw_threshold=align_to_far_trial_params.get("yaw_threshold", 0.1),
                    confirm_duration=align_to_far_trial_params.get(
                        "confirm_duration", 2.0
                    ),
                    timeout=60.0,
                    cancel_on_success=align_to_far_trial_params.get(
                        "cancel_on_success", False
                    ),
                    keep_orientation=align_to_far_trial_params.get(
                        "keep_orientation", False
                    ),
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
                    target_selection=self.target_selection,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "BIN_SECOND_TRIAL",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_TO_FOUND_DROP_AREA",
                SetAlignToFoundState(
                    source_frame=set_align_to_found_drop_area_params.get(
                        "source_frame", "taluy/base_link/ball_dropper_link"
                    )
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_DROP_ALIGNMENT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_DROP_ALIGNMENT",
                DelayState(
                    delay_time=wait_for_aligning_drop_area_params.get(
                        "delay_time", 15.0
                    )
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
                DelayState(
                    delay_time=wait_for_ball_drop_1_params.get("delay_time", 5.0)
                ),
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
                DelayState(
                    delay_time=wait_for_ball_drop_2_params.get("delay_time", 3.0)
                ),
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
                    self.tf_buffer,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "DROP_BALL_1",
                },
            )

    def execute(self, userdata):
        outcome = self.state_machine.execute()
        if outcome is None:
            return "preempted"
        return outcome
