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
import threading
import smach_ros


class MainStateMachineNode:
    def __init__(self):
        self.previous_enabled = False

        # USER EDIT
        self.gate_depth = -0.9
        self.target_gate_link = "gate_blue_arrow_link"

        self.red_buoy_radius = 2.2
        self.red_buoy_depth = -0.7

        self.torpedo_map_radius = 1.5
        self.torpedo_map_depth = -1.3

        self.bin_whole_depth = -1.0

        self.octagon_depth = -1.0
        self.test_mode = False  # Flag or Change to True to enable test mode
        self.test_states = []  # List of states to test
        # USER EDIT

        # automatically select other params
        mission_selection_map = {
            "gate_red_arrow_link": {"red_buoy": "cw"},
            "gate_blue_arrow_link": {"red_buoy": "ccw"},
        }
        self.red_buoy_direction = mission_selection_map[self.target_gate_link]["red_buoy"]
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
                })


    def set_test_mode(self, test_mode, test_states=None):
        self.test_mode = test_mode
        if test_states is not None:
            self.test_states = test_states

    def execute_state_machine(self):
        # Map state names to their corresponding classes and parameters using variables defined at the start
        state_mapping = {
            "INITIALIZE": (InitializeState, {}),
            "NAVIGATE_THROUGH_GATE": (NavigateThroughGateState, {"gate_depth": self.gate_depth}),
            "NAVIGATE_AROUND_RED_BUOY": (RotateAroundBuoyState, {"radius": self.red_buoy_radius, "depth": self.red_buoy_depth, "direction": self.red_buoy_direction}),
            "NAVIGATE_TO_TORPEDO_TASK": (TorpedoTaskState, {"radius": self.torpedo_map_radius, "depth": self.torpedo_map_depth}),
            "NAVIGATE_TO_BIN_TASK": (BinTaskState, {"depth": self.bin_whole_depth}),
            "NAVIGATE_TO_OCTAGON_TASK": (OctagonTaskState, {"depth": self.octagon_depth}),
        }

        # Execute the state machine
        if self.test_mode and self.test_states:
            rospy.loginfo("Executing test state machine with states: %s", self.test_states)
            test_sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])
            with test_sm:
                for i, state_name in enumerate(self.test_states):
                    next_state = self.test_states[i + 1] if i + 1 < len(self.test_states) else "succeeded"
                    state_class, params = state_mapping.get(state_name, (None, {}))
                    if state_class:
                        smach.StateMachine.add(
                            state_name,
                            state_class(**params),
                            transitions={"succeeded": next_state, "preempted": "preempted", "aborted": "aborted"},
                        )
            outcome = test_sm.execute()
            rospy.loginfo("Test state machine exited with outcome: %s", outcome)
        else:
            rospy.loginfo("Executing main state machine")
            outcome = self.sm.execute()  # Execute the main state machine

        rospy.loginfo(f"Main state machine exited with outcome: {outcome}")

    def start(self, event=None):
        self.execute_state_machine()

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
        node.set_test_mode(True, ["NAVIGATE_AROUND_RED_BUOY"])
        #optionally set test mode
        node.start()
    except KeyboardInterrupt:
        if node is not None:
            node.sm.request_preempt()
