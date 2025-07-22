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

        # Parameters
        self.gate_params = rospy.get_param("~gate_task", {})
        self.red_buoy_params = rospy.get_param("~red_buoy_task", {})
        self.torpedo_params = rospy.get_param("~torpedo_task", {})
        self.bin_params = rospy.get_param("~bin_task", {})
        self.octagon_params = rospy.get_param("~octagon_task", {})
        self.mission_params = rospy.get_param("~mission_params", {})

        self.gate_depth = self.gate_params.get("gate_depth", -1.5)
        self.gate_search_depth = self.gate_params.get("gate_search_depth", -0.7)

        # Get target selection from YAML
        self.target_selection = self.mission_params.get("target_selection", "shark")
        self.mission_targets = self.mission_params.get("mission_targets", {})

        self.red_buoy_radius = self.red_buoy_params.get("radius", 2.2)
        self.red_buoy_depth = self.red_buoy_params.get("depth", -0.7)
        self.red_buoy_direction = self.mission_targets.get(
            self.target_selection, {}
        ).get("red_buoy_direction", "ccw")

        self.torpedo_map_depth = self.torpedo_params.get("map_depth", -1.3)
        self.torpedo_target_frame = self.torpedo_params.get(
            "target_frame", "torpedo_target"
        )
        self.torpedo_realsense_target_frame = self.torpedo_params.get(
            "realsense_target_frame", "torpedo_target_realsense"
        )
        self.torpedo_fire_frames = self.mission_targets.get(
            self.target_selection, {}
        ).get(
            "torpedo_fire_frames",
            ["torpedo_shark_fire_frame", "torpedo_sawfish_fire_frame"],
        )

        self.bin_front_look_depth = self.bin_params.get("front_look_depth", -1.2)
        self.bin_bottom_look_depth = self.bin_params.get("bottom_look_depth", -0.7)

        self.octagon_depth = self.octagon_params.get("depth", -1.0)

        test_mode = rospy.get_param("~test_mode", False)
        # Get test states from ROS param
        if test_mode:
            state_map = rospy.get_param("~state_map", {})
            test_states_str = rospy.get_param("~test_states", "")
            short_state_list = [state.strip() for state in test_states_str.split(",")]

            # Map test states to full names
            self.state_list = [
                state_map[state] for state in short_state_list if state in state_map
            ]
        else:
            self.state_list = rospy.get_param("~full_mission_states")

        # Subscribe to propulsion status
        rospy.Subscriber("propulsion_board/status", Bool, self.enabled_callback)

    def execute_state_machine(self):
        # Map state names to their corresponding classes and parameters
        state_mapping = {
            "INITIALIZE": (InitializeState, {}),
            "NAVIGATE_THROUGH_GATE": (NavigateThroughGateState, {}),
            "NAVIGATE_AROUND_RED_BUOY": (
                RotateAroundBuoyState,
                {
                    "radius": self.red_buoy_radius,
                    "direction": self.red_buoy_direction,
                    "red_buoy_depth": self.red_buoy_depth,
                },
            ),
            "NAVIGATE_TO_TORPEDO_TASK": (TorpedoTaskState, {}),
            "NAVIGATE_TO_BIN_TASK": (
                BinTaskState,
                {},
            ),
            "NAVIGATE_TO_OCTAGON_TASK": (
                OctagonTaskState,
                {},
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
