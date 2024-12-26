#!/usr/bin/env python3

import rospy
import smach
import auv_smach
from auv_smach.initialize import InitializeState
from auv_smach.gate import NavigateThroughGateState
from auv_smach.red_buoy import RotateAroundBuoyState
from auv_smach.torpedo import TorpedoTaskState
from auv_smach.bin import BinTaskState
from auv_smach.octagon import OctagonTaskState
from std_msgs.msg import Bool
from auv_smach.common import (
    CancelAlignControllerState,
    SetDepthState,
)
from auv_smach.initialize import DelayState
import threading


class MainStateMachineNode:
    def __init__(self):
        self.previous_enabled = False

        # USER EDIT
        gate_depth = -1.75
        target_gate_link = "gate_blue_arrow_link"

        red_buoy_radius = 1.5
        red_buoy_depth = -0.7

        torpedo_map_radius = 1.5
        torpedo_map_depth = -1.3

        bin_whole_depth = -1.0

        octagon_depth = -1.0
        # USER EDIT

        # automatically select other params
        mission_selection_map = {
            "gate_red_arrow_link": {"red_buoy": "cw"},
            "gate_blue_arrow_link": {"red_buoy": "ccw"},
        }
        red_buoy_direction = mission_selection_map[target_gate_link]["red_buoy"]
        # automatically select other params

        rospy.Subscriber("/taluy/propulsion_board/status", Bool, self.enabled_callback)

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with self.sm:
            # Add the FullMissionState as the single state in this state machine
            smach.StateMachine.add(
                "INITIALIZE",
                InitializeState(),
                transitions={
                    "succeeded": "NAVIGATE_THROUGH_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_THROUGH_GATE",
                NavigateThroughGateState(gate_depth=gate_depth),
                transitions={
                    "succeeded": "NAVIGATE_AROUND_RED_BUOY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_AROUND_RED_BUOY",
                RotateAroundBuoyState(
                    radius=red_buoy_radius,
                    direction=red_buoy_direction,
                    red_buoy_depth=red_buoy_depth,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_SURFACING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_SURFACING",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "SURFACING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACING",
                SetDepthState(depth=0.3, sleep_duration=4.0),
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

        outcome = self.sm.execute()

        # rospy.Timer(rospy.Duration(0.1), self.start)

    def start(self, event=None):
        outcome = self.sm.execute()
        rospy.loginfo("Main state machine exited with outcome: %s", outcome)

    def enabled_callback(self, msg):
        falling_edge = self.previous_enabled and not msg.data

        self.previous_enabled = msg.data

        if falling_edge:
            self.sm.request_preempt()
            # restart
            rospy.Timer(rospy.Duration(0.1), self.start)


if __name__ == "__main__":
    rospy.init_node("main_state_machine")
    node = None
    try:
        node = MainStateMachineNode()
    except KeyboardInterrupt:
        if node is not None:
            node.sm.request_preempt()
