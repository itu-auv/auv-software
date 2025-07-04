from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SetDepthState,
)

from auv_smach.initialize import DelayState

from auv_smach.common import LaunchTorpedoState, SearchForPropState


class TorpedoTargetFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_target_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TorpedoRealsenseTargetFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_realsense_target_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TorpedoFireFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_hole_target_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TorpedoTaskState(smach.State):
    def __init__(
        self,
        torpedo_map_depth,
        torpedo_target_frame,
        torpedo_realsense_target_frame,
        torpedo_fire_frame,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoTargetFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "SET_TORPEDO_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_DEPTH",
                SetDepthState(depth=torpedo_map_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "FIND_AND_AIM_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_TORPEDO",
                SearchForPropState(
                    look_at_frame="torpedo_map_link",
                    alignment_frame="torpedo_map_travel_start",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_TO_TORPEDO_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_TO_TORPEDO_TARGET",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=torpedo_target_frame,
                    # angle_offset=-1.57,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=5.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_TORPEDO_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoTargetFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ROTATE_FOR_REALSENSE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROTATE_FOR_REALSENSE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=torpedo_target_frame,
                    angle_offset=3.14,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    timeout=20.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                TorpedoRealsenseTargetFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRAME",
                DelayState(delay_time=10.0),
                transitions={
                    "succeeded": "DISABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                TorpedoRealsenseTargetFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "TURN_TO_LAUNCH_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TURN_TO_LAUNCH_TORPEDO",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=torpedo_realsense_target_frame,
                    angle_offset=-1.57,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=10.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                TorpedoFireFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "SET_FIRE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FIRE_DEPTH",
                SetDepthState(
                    depth=0.0, sleep_duration=5.0, frame_id=torpedo_fire_frame
                ),
                transitions={
                    "succeeded": "DISABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
            	"DISABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                TorpedoFireFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ALIGN_TO_TORPEDO_FIRE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
	    )
            smach.StateMachine.add(
                "ALIGN_TO_TORPEDO_FIRE_FRAME",
                AlignFrame(
                    source_frame="taluy/base_link/torpedo_upper_link",
                    target_frame=torpedo_fire_frame,
                    angle_offset=-1.57,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=10.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_1",
                LaunchTorpedoState(id=1),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_LAUNCH",
                DelayState(delay_time=6.0),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_2",
                LaunchTorpedoState(id=2),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_2_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_2_LAUNCH",
                DelayState(delay_time=6.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER_FINAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER_FINAL",
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
