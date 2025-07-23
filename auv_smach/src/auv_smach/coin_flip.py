import smach
import rospy
from std_srvs.srv import SetBool, SetBoolRequest
from .initialize import *
import smach_ros
from auv_smach.common import (
    AlignFrame,
    SetDepthState,
)


class RescueCoinFlipServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_coin_flip_rescuer",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class CoinFlipState(smach.StateMachine):
    def __init__(self, gate_search_depth):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        with self:
            smach.StateMachine.add(
                "RESCUE_COIN_FLIP_SERVICE_ENABLE",
                RescueCoinFlipServiceEnableState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_RESCUE_COIN_FLIP_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_RESCUE_COIN_FLIP_FRAME",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "RESCUE_COIN_FLIP_SERVICE_DISABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESCUE_COIN_FLIP_SERVICE_DISABLE",
                RescueCoinFlipServiceEnableState(req=False),
                transitions={
                    "succeeded": "ALIGN_TO_RESCUE_COIN_FLIP_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_RESCUE_COIN_FLIP_FRAME",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="coin_flip_rescuer",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "SET_DEPTH_FOR_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DEPTH_FOR_GATE",
                SetDepthState(depth=gate_search_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "ALIGN_ORIENTATION_TO_RESCUE_COIN_FLIP_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_ORIENTATION_TO_RESCUE_COIN_FLIP_FRAME",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="coin_flip_rescuer",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=15.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
