import smach
import rospy
from auv_smach.tf_utils import get_tf_buffer, get_base_link
from auv_smach.common import (
    SetDepthState,
    SearchForPropState,
    DynamicPathWithTransformAndVisibilityCheck,
    CancelAlignControllerState,
    SetDetectionState,
    SetDetectionFocusBottomState,
    SetDetectionFocusState,
    AlignFrameWithVisibilityCheck,
)


class BinTaskMiniState(smach.State):
    def __init__(
        self,
        bin_search_depth: float = -0.5,
        bin_drop_depth: float = -1.0,
        target_animal: str = "bin_shark_link",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.base_link = get_base_link()
        self.bin_search_depth = bin_search_depth
        self.bin_drop_depth = bin_drop_depth
        self.target_animal = target_animal
        self.bin_look_at_frame = "bin_whole_link"
        self.bin_alignment_frame = "bin_search"

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "SET_FRONT_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FRONT_FOCUS",
                SetDetectionFocusState(focus_object="bin"),
                transitions={
                    "succeeded": "ENABLE_BOTTOM_CAMERA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BOTTOM_CAMERA",
                SetDetectionState(camera_name="bottom", enable=True),
                transitions={
                    "succeeded": "SET_BOTTOM_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_BOTTOM_FOCUS",
                SetDetectionFocusBottomState(focus_object="bin"),
                transitions={
                    "succeeded": "SET_SEARCH_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_SEARCH_DEPTH",
                SetDepthState(
                    depth=self.bin_search_depth,
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
                    look_at_frame=self.bin_look_at_frame,
                    alignment_frame=self.bin_alignment_frame,
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_TO_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_FRAME_TO_BIN",
                DynamicPathWithTransformAndVisibilityCheck(
                    plan_target_frame=self.bin_look_at_frame,
                    transform_source_frame=self.base_link,
                    transform_target_frame=self.target_animal,
                    align_source_frame=self.base_link,
                    prop_name=self.bin_look_at_frame,
                    lost_timeout=6.0,
                    transform_timeout=20.0,
                ),
                transitions={
                    "succeeded": "ALIGN_PRECISELY_TO_BIN",
                    "target_lost": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_PRECISELY_TO_BIN",
                AlignFrameWithVisibilityCheck(
                    source_frame=self.base_link,
                    target_frame="bin_shark_link",
                    prop_name="bin_shark_link",
                    lost_timeout=3.0,
                    confirm_duration=10.0,
                    timeout=20.0,
                    cancel_on_success=True,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "SET_DROP_DEPTH",
                    "target_lost": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_DROP_DEPTH",
                SetDepthState(
                    depth=self.bin_drop_depth,
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
        rospy.logdebug("[DropMarkerInBinMiniState] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
