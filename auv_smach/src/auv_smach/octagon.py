from .initialize import *
import smach
import smach_ros
from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SetDetectionFocusState,
    DynamicPathState,
    AlignFrame,
    SearchForPropState,
    SetDetectionState,
)
from auv_smach.initialize import DelayState
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest


class MoveGripperServiceState(smach_ros.ServiceState):
    def __init__(self, id: int):
        smach_ros.ServiceState.__init__(
            self,
            "move_gripper",
            Trigger,
            request=TriggerRequest(),
        )


class OctagonFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_octagon_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class OctagonTaskState(smach.State):
    def __init__(self, octagon_depth: float, animal: str):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.griper_mode = False
        self.animal_frame = f"gate_{animal}_link"
        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_INITIAL_DEPTH",
                SetDepthState(depth=-1.2, sleep_duration=4.0),
                transitions={
                    "succeeded": "FOCUS_ON_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_OCTAGON",
                SetDetectionFocusState(focus_object="octagon"),
                transitions={
                    "succeeded": "FIND_AIM_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AIM_OCTAGON",
                SearchForPropState(
                    look_at_frame="octagon_link",
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame="taluy/base_link",
                    rotation_speed=-0.2,
                ),
                transitions={
                    "succeeded": "ENABLE_OCTAGON_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_OCTAGON_FRAME_PUBLISHER",
                OctagonFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_OCTAGON_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_OCTAGON_FRAME",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "CLOSE_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_DETECTION",
                SetDetectionFocusState(focus_object="none"),
                transitions={
                    "succeeded": "CLOSE_OCTAGON_PUBLİSHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_OCTAGON_PUBLİSHER",
                OctagonFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "SET_OCTAGON_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(depth=octagon_depth, sleep_duration=4.0),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame="octagon_closer_link",
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
                    target_frame="octagon_closer_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    cancel_on_success=False,
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
                    "succeeded": "GO_TO_OCTAGON_LINK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "GO_TO_OCTAGON_LINK",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": (
                        "SURFACE_TO_ANIMAL_DEPTH"
                        if not self.griper_mode
                        else "MOVE_GRIPPER"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "MOVE_GRIPPER",
                MoveGripperServiceState(id=0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACE_TO_ANIMAL_DEPTH",
                SetDepthState(depth=-0.27, sleep_duration=4.0),
                transitions={
                    "succeeded": "SEARCH_FOR_ANIMAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_ANIMAL",
                SetDetectionFocusState(focus_object="gate"),
                transitions={
                    "succeeded": "FIND_AIM_ANIMAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AIM_ANIMAL",
                SearchForPropState(
                    look_at_frame=self.animal_frame,
                    alignment_frame="animal_search_frame",
                    full_rotation=True,
                    set_frame_duration=10.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "GO_BACK_TO_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "GO_BACK_TO_OCTAGON",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_link",
                    angle_offset=0.0,
                    dist_threshold=0.3,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=60.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "LOOK_TO_ANIMAL_AGAIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_TO_ANIMAL_AGAIN",
                SearchForPropState(
                    look_at_frame=self.animal_frame,
                    alignment_frame="animal_search_frame",
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "SURFACE_TO_SURFACE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACE_TO_SURFACE",
                SetDepthState(depth=-0.05, sleep_duration=7.0),
                transitions={
                    "succeeded": "SET_FINAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FINAL_DEPTH",
                SetDepthState(depth=octagon_depth, sleep_duration=4.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER_OCTAGON",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
