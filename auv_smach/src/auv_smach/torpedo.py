from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest
from auv_msgs.srv import (
    SetString,
    SetStringRequest,
)

from auv_smach.common import (
    SetFrameLookingAtState,
    AlignFrame,
    CancelAlignControllerState,
    SetDepthState,
)

from auv_smach.initialize import DelayState

from auv_smach.common import (
    LaunchTorpedoState,
)


class TorpedoFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_frames",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TorpedoFrameNameService(smach_ros.ServiceState):
    def __init__(self, req: str):
        smach_ros.ServiceState.__init__(
            self,
            "set_torpedo_frame",
            SetString,
            request=SetStringRequest(data=req),
        )


class TorpedoTaskState(smach.State):
    def __init__(self, torpedo_map_depth, torpedo_target_link):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoFramePublisherServiceState(req=True),
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
                    "succeeded": "SET_TORPEDO_TRAVEL_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_TRAVEL_START",
                SetFrameLookingAtState(
                    base_frame="taluy/base_link",
                    look_at_frame="torpedo_map_link",
                    target_frame="torpedo_map_travel_start",
                ),
                transitions={
                    "succeeded": "SET_TORPEDO_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_ALIGN_CONTROLLER_TARGET",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="torpedo_map_travel_start",
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    timeout=5.0,
                    cancel_on_success=False,
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
                    target_frame=torpedo_target_link,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    timeout=20.0,
                    cancel_on_success=False,
                ),
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
                    target_frame=torpedo_target_link,
                    angle_offset=1.57,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    timeout=5.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "SET_TORPEDO_FRAME_NAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_FRAME_NAME",
                # TODO: replace name
                TorpedoFrameNameService(req="torpedo_map_link_0"),
                transitions={
                    "succeeded": "WAIT_FOR_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRAME",
                DelayState(delay_time=5.0),
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
                    target_frame=torpedo_target_link,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    timeout=5.0,
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
