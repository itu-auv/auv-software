from .initialize import *
import smach
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest

from auv_smach.common import (
    AlignFrame,
    SetDepthState,
)


class EnablePipeFramePublisherState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "pipe_frame_publisher/enable",
            Trigger,
            request=TriggerRequest(),
        )


class DisablePipeFramePublisherState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "pipe_frame_publisher/disable",
            Trigger,
            request=TriggerRequest(),
        )


class PipeTaskState(smach.State):
    def __init__(self, pipe_map_depth, pipe_target_frame):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_DEPTH",
                SetDepthState(depth=pipe_map_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "ENABLE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ENABLE_PUBLISHER",
                EnablePipeFramePublisherState(),
                transitions={
                    "succeeded": "ALIGN_TO_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_PIPE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=pipe_target_frame,
                    dist_threshold=0.2,
                    yaw_threshold=0.2,
                    confirm_duration=5.0,
                    timeout=60.0,
                    cancel_on_success=True,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "DISABLE_PUBLISHER",
                    "preempted": "DISABLE_PUBLISHER",
                    "aborted": "DISABLE_PUBLISHER",
                },
            )

            smach.StateMachine.add(
                "DISABLE_PUBLISHER",
                DisablePipeFramePublisherState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
