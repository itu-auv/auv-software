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
    def __init__(self, octagon_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.griper_mode=False
        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(depth=octagon_depth, sleep_duration=4.0),
                transitions={
                    "succeeded": "FOCUS_ON_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_BIN",
                SetDetectionFocusState(focus_object="octagon_link"),
                transitions={
                    "succeeded": "DISABLE_OCTAGON_FRAME_PUBLISHER_ON_SUCCESS",
                    "preempted": "prteempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_BIN",
                SearchForPropState(
                    look_at_frame="octagon_link",
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    set_frame_duration=4.0,
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
                "ENABLE_OCTAGON_FRAME_PUBLISHER",
                OctagonFramePublisherServiceState(req=True),
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
                    "succeeded": "FOCUS_ON_BIN",
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
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER",
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
                "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame="octagon_link",
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
                    target_frame="octagon_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "SURFACE_START"
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
