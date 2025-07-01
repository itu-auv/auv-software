from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    PlanPathToSingleFrameState,
    ExecutePlannedPathsState,
)
from auv_smach.initialize import DelayState


class OctagonTransformServiceEnableState(smach_ros.ServiceState):
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_OCTAGON_FRAME",
                OctagonTransformServiceEnableState(True),
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
                    "succeeded": "SEARCH_FOR_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_OCTAGON",
                SearchForPropState(
                    look_at_frame="octagon_link",
                    alignment_frame="octagon_search",
                    full_rotation=False,
                    set_frame_duration=6.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "PLAN_PATH_TO_CLOSER_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_PATH_TO_CLOSER_FRAME",
                PlanPathToSingleFrameState(
                    tf_buffer=self.tf_buffer,
                    target_frame="octagon_closer_link",
                    source_frame="taluy/base_link",
                ),
                transitions={
                    "succeeded": "DISABLE_OCTAGON_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_OCTAGON_FRAME",
                OctagonTransformServiceEnableState(False),
                transitions={
                    "succeeded": "SET_ALIGN_TO_DYNAMIC_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_TO_DYNAMIC_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="dynamic_target",
                ),
                transitions={
                    "succeeded": "EXECUTE_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_PATH",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "SET_ALIGN_TO_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_TO_OCTAGON",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="octagon_link",
                ),
                transitions={
                    "succeeded": "DELAY_BEFORE_SURFACE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DELAY_BEFORE_SURFACE",
                DelayState(delay_time=10.0),
                transitions={
                    "succeeded": "SET_DEPTH_MINUS_4",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DEPTH_MINUS_4",
                SetDepthState(depth=-0.4, sleep_duration=4.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_FOR_ROTATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_FOR_ROTATION",
                CancelAlignControllerState(),
                #    transitions={
                #        "succeeded": "SEARCH_FOR_OCTAGON_FULL_ROTATION",
                #        "preempted": "preempted",
                #        "aborted": "aborted",
                #    },
                # )
                # smach.StateMachine.add(
                #    "SEARCH_FOR_OCTAGON_FULL_ROTATION",
                #    SearchForPropState(
                #        look_at_frame="octagon_animal_image_link",
                #        alignment_frame="octagon_search",
                #        full_rotation=True,
                #        set_frame_duration=4.0,
                #        source_frame="taluy/base_link",
                #        rotation_speed=0.3,
                #    ),
                transitions={
                    "succeeded": "SET_DEPTH_0",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DEPTH_0",
                SetDepthState(depth=0.0, sleep_duration=3.0),
                transitions={
                    "succeeded": "SET_FINAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FINAL_DEPTH",
                SetDepthState(depth=-0.5, sleep_duration=4.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN",
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
