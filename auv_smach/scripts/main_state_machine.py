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


class MainStateMachineNode:
    def __init__(self):
        self.previous_enabled = False

        # USER EDIT
        self.gate_depth = -1.2
        self.target_gate_link = "gate_blue_arrow_link"

        self.red_buoy_radius = 2.2
        self.red_buoy_depth = -0.7

        self.torpedo_map_radius = 1.5
        self.torpedo_map_depth = -1.3

        self.bin_whole_depth = -1.0

        self.octagon_depth = -1.0
        # USER EDIT

        test_mode = rospy.get_param("~test_mode", False)
        # Get test states from ROS param
        if test_mode:
            state_map = rospy.get_param("~state_map")

            short_state_list = rospy.get_param("~test_states", "").split(",")

            # Parse state mapping
            state_mapping = {
                item.split(":")[0].strip(): item.split(":")[1].strip()
                for item in state_map.strip().split(",")
            }

            # Map test states to full names
            self.state_list = [
                state_mapping[state.strip()]
                for state in short_state_list
                if state.strip() in state_mapping
            ]
        else:
            self.state_list = rospy.get_param("~full_mission_states")

        # automatically select other params
        mission_selection_map = {
            "gate_red_arrow_link": {"red_buoy": "cw"},
            "gate_blue_arrow_link": {"red_buoy": "ccw"},
        }
        self.red_buoy_direction = mission_selection_map[self.target_gate_link][
            "red_buoy"
        ]
        # Subscribe to propulsion status
        rospy.Subscriber("propulsion_board/status", Bool, self.enabled_callback)

    def execute_state_machine(self):
        # Map state names to their corresponding classes and parameters
        state_mapping = {
            "INITIALIZE": (InitializeState, {}),
            "NAVIGATE_THROUGH_GATE": (
                NavigateThroughGateState,
                {"gate_depth": self.gate_depth},
            ),
            "NAVIGATE_AROUND_RED_BUOY": (
                RotateAroundBuoyState,
                {
                    "radius": self.red_buoy_radius,
                    "direction": self.red_buoy_direction,
                    "red_buoy_depth": self.red_buoy_depth,
                },
            ),
            "NAVIGATE_TO_TORPEDO_TASK": (
                TorpedoTaskState,
                {
                    "torpedo_map_radius": self.torpedo_map_radius,
                    "torpedo_map_depth": self.torpedo_map_depth,
                },
            ),
            "NAVIGATE_TO_BIN_TASK": (
                BinTaskState,
                {"bin_whole_depth": self.bin_whole_depth},
            ),
            "NAVIGATE_TO_OCTAGON_TASK": (
                OctagonTaskState,
                {"octagon_depth": self.octagon_depth},
            ),
        }

        # Validate and execute state machine
        if not self.state_list:
            rospy.logerr("No states to execute")
            return

        rospy.loginfo("Executing state machine with states: %s", self.state_list)
        sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with sm:
            for i, state_name in enumerate(self.state_list):
                next_state = (
                    self.state_list[i + 1]
                    if i + 1 < len(self.state_list)
                    else "succeeded"
                )
                state_class, params = state_mapping.get(state_name, (None, {}))

                if state_class is None:
                    rospy.logerr(f"Unknown state: {state_name}")
                    continue

                smach.StateMachine.add(
                    state_name,
                    state_class(**params),
                    transitions={
                        "succeeded": next_state,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

        # Execute the state machine
        try:
            outcome = sm.execute()
            rospy.loginfo(f"State machine exited with outcome: {outcome}")
        except Exception as e:
            rospy.logerr(f"Error executing state machine: {e}")

    def enabled_callback(self, msg):
        falling_edge = self.previous_enabled and not msg.data

        self.previous_enabled = msg.data

        if falling_edge:
            self.sm.request_preempt()
            # restart
            rospy.Timer(rospy.Duration(0.1), self.start)


if __name__ == "__main__":
    rospy.init_node("main_state_machine")
    try:
        node = MainStateMachineNode()
        node.execute_state_machine()
    except KeyboardInterrupt:
        rospy.loginfo("State machine node interrupted")
