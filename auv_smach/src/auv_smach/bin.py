import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
import rospy
import tf2_ros
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SetDepthState,
    DropBallState,
    SearchForPropState,
)
from auv_smach.initialize import DelayState


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
        rate = rospy.Rate(10)

        userdata.found_frame = None

        while not self.preempt_requested():
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

        userdata.found_frame = None

        return "preempted"


class BinTransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_bin_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


def AlignAndCheckConcurrently(align_target_frame):

    def child_term_cb(outcome_map):
        if outcome_map["CHECK"] == "succeeded":
            return True
        if outcome_map["ALIGN"] is not None and outcome_map["ALIGN"] != "preempted":
            return True
        return False

    outcome_map = {
        "found": {"CHECK": "succeeded"},
        "not_found": {"ALIGN": "succeeded"},
        "aborted": {"ALIGN": "aborted"},
        "preempted": {"ALIGN": "preempted"},
    }

    concurrence_state = smach.Concurrence(
        outcomes=["found", "not_found", "aborted", "preempted"],
        default_outcome="aborted",
        output_keys=["found_frame"],
        child_termination_cb=child_term_cb,
        outcome_map=outcome_map,
    )

    with concurrence_state:
        smach.Concurrence.add(
            "ALIGN",
            AlignFrame(
                source_frame="taluy/base_link",
                target_frame=align_target_frame,
                dist_threshold=0.1,
                yaw_threshold=0.1,
                confirm_duration=1.0,
                timeout=60.0,
                cancel_on_success=False,
            ),
        )
        smach.Concurrence.add(
            "CHECK",
            CheckForDropAreaState(source_frame="odom"),
            remapping={"found_frame": "found_frame"},
        )

    return concurrence_state


class BinTaskState(smach.State):
    def __init__(self, bin_task_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "SET_BIN_DEPTH",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DEPTH",
                SetDepthState(depth=bin_task_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "FIND_AND_AIM_BIN",
                    "aborted": "aborted",
                    "preempted": "preempted",
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
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "ALIGN_AND_CHECK_INITIAL",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_AND_CHECK_INITIAL",
                AlignAndCheckConcurrently(align_target_frame="bin_whole_link"),
                transitions={
                    "found": "ALIGN_DROPPER_TO_TARGET",
                    "not_found": "ALIGN_AND_CHECK_FAR",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
                remapping={"found_frame": "found_frame"},
            )

            smach.StateMachine.add(
                "ALIGN_AND_CHECK_FAR",
                AlignAndCheckConcurrently(align_target_frame="bin_far_trial"),
                transitions={
                    "found": "ALIGN_DROPPER_TO_TARGET",
                    "not_found": "ALIGN_AND_CHECK_SECOND_TRIAL",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
                remapping={"found_frame": "found_frame"},
            )

            smach.StateMachine.add(
                "ALIGN_AND_CHECK_SECOND_TRIAL",
                AlignAndCheckConcurrently(align_target_frame="bin_second_trial"),
                transitions={
                    "found": "ALIGN_DROPPER_TO_TARGET",
                    "not_found": "DROP_BALL_1",  # This transition will skip the dropper alignment if no frame is found after all trials
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
                remapping={"found_frame": "found_frame"},
            )

            smach.StateMachine.add(
                "ALIGN_DROPPER_TO_TARGET",
                AlignFrame(
                    source_frame="taluy/base_link/ball_dropper_link",
                    target_frame="bin/blue_link",  # Provide a placeholder target_frame
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=20.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_DROP_AREA",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
                remapping={"target_frame": "found_frame"},
            )

            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_DROP_AREA",
                DelayState(delay_time=12.0),
                transitions={
                    "succeeded": "DROP_BALL_1",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )

            smach.StateMachine.add(
                "DROP_BALL_1",
                DropBallState(),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_1",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_1",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "DROP_BALL_2",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_2",
                DropBallState(),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_2",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_2",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "aborted": "aborted",
                    "preempted": "preempted",
                },
            )

    def execute(self, userdata):
        outcome = self.state_machine.execute()
        return outcome if outcome is not None else "preempted"
