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

        # Get target selection from YAML
        self.target_selection = rospy.get_param("~target_selection", "shark")
        self.mission_targets = rospy.get_param(
            "~mission_targets",
            {
                "shark": {
                    "gate_target_frame": "gate_shark_link",
                    "red_buoy_direction": "ccw",
                    "slalom_direction": "left_side",
                    "bin_target_frame": "bin_shark_link",
                    "torpedo_target_frame": "torpedo_hole_shark_link",
                    "octagon_target_frame": "octagon_shark_link",
                },
                "sawfish": {
                    "gate_target_frame": "gate_sawfish_link",
                    "red_buoy_direction": "cw",
                    "slalom_direction": "right_side",
                    "bin_target_frame": "bin_sawfish_link",
                    "torpedo_target_frame": "torpedo_hole_sawfish_link",
                    "octagon_target_frame": "octagon_sawfish_link",
                },
            },
        )

        self.gate_search_depth = -0.7
        self.gate_depth = -1.35

        self.red_buoy_radius = 2.2
        self.red_buoy_depth = -0.7

        self.torpedo_map_depth = -1.3
        self.torpedo_target_frame = "torpedo_target"
        self.torpedo_realsense_target_frame = "torpedo_target_realsense"
        self.torpedo_fire_frame = "torpedo_fire_frame"
        self.torpedo_shark_fire_frame = "torpedo_shark_fire_frame"
        self.torpedo_sawfish_fire_frame = "torpedo_sawfish_fire_frame"

        self.bin_front_look_depth = -1.2
        self.bin_bottom_look_depth = -0.7

        self.octagon_depth = -1.0

        # Set parameters based on target selection
        self.target_frames = self.mission_targets[self.target_selection]
        self.red_buoy_direction = self.target_frames["red_buoy_direction"]
        # self.slalom_direction = self.target_frames["slalom_direction"]
        # self.bin_target_frame = self.target_frames["bin_target_frame"]
        # self.torpedo_target_frame = self.target_frames["torpedo_target_frame"]
        # self.octagon_target_frame = self.target_frames["octagon_target_frame"]
        # self.gate_target_frame = self.target_frames["gate_target_frame"]

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

        # Subscribe to propulsion status
        rospy.Subscriber("propulsion_board/status", Bool, self.enabled_callback)

    def execute_state_machine(self):
        # Map state names to their corresponding classes and parameters
        state_mapping = {
            "INITIALIZE": (InitializeState, {}),
            "NAVIGATE_THROUGH_GATE": (
                NavigateThroughGateState,
                {
                    "gate_depth": self.gate_depth,
                    "gate_search_depth": self.gate_search_depth,
                },
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
                    "torpedo_map_depth": self.torpedo_map_depth,
                    "torpedo_target_frame": self.torpedo_target_frame,
                    "torpedo_realsense_target_frame": self.torpedo_realsense_target_frame,
                    "torpedo_fire_frames": (
                        [
                            self.torpedo_shark_fire_frame,
                            self.torpedo_sawfish_fire_frame,
                        ]
                        if self.target_selection == "shark"
                        else [
                            self.torpedo_sawfish_fire_frame,
                            self.torpedo_shark_fire_frame,
                        ]
                    ),
                },
            ),
            "NAVIGATE_TO_BIN_TASK": (
                BinTaskState,
                {
                    "bin_front_look_depth": self.bin_front_look_depth,
                    "bin_bottom_look_depth": self.bin_bottom_look_depth,
                    "target_selection": self.target_selection,
                },
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
