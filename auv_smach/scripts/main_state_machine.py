#!/usr/bin/env python3

import rospy
import smach
import auv_smach
import math
from auv_smach.initialize import InitializeState
from auv_smach.gate import NavigateThroughGateState
from auv_smach.slalom import NavigateThroughSlalomState
from auv_smach.red_buoy import RotateAroundBuoyState
from auv_smach.torpedo import TorpedoTaskState
from auv_smach.bin import BinTaskState
from auv_smach.octagon import OctagonTaskState
from auv_smach.return_home import NavigateReturnThroughGateState
from auv_smach.acoustic import AcousticTransmitter, AcousticReceiver
from auv_smach.pipeline import NavigateThroughPipelineState
from auv_smach.gps import NavigateToGpsTargetState
from std_msgs.msg import Bool
import threading
from dynamic_reconfigure.client import Client
from auv_bringup.cfg import SmachParametersConfig


class MainStateMachineNode:
    def __init__(self):
        self.previous_enabled = False

        # Initialize dynamic reconfigure client
        self.dynamic_reconfigure_client = Client(
            "smach_parameters_server",
            timeout=10,
            config_callback=self.dynamic_reconfigure_callback,
        )

        self.return_home_station = "bin_exit"

        # Get initial values from dynamic reconfigure
        self.selected_animal = "sawfish"
        self.slalom_mode = "close"

        # Exit angles in degrees (will be converted to radians)
        self.gate_exit_angle_deg = 0.0
        self.slalom_exit_angle_deg = 0.0
        self.bin_exit_angle_deg = 0.0
        self.torpedo_exit_angle_deg = 0.0

        # Get current configuration from server
        try:
            current_config = self.dynamic_reconfigure_client.get_configuration()
            if current_config:
                self.selected_animal = current_config.get("selected_animal", "sawfish")
                self.slalom_mode = current_config.get("slalom_mode", "close")
                self.gate_exit_angle_deg = current_config.get("gate_exit_angle", 0.0)
                self.slalom_exit_angle_deg = current_config.get(
                    "slalom_exit_angle", 0.0
                )
                self.bin_exit_angle_deg = current_config.get("bin_exit_angle", 0.0)
                self.torpedo_exit_angle_deg = current_config.get(
                    "torpedo_exit_angle", 0.0
                )
                rospy.loginfo(
                    f"Loaded current config: selected_animal={self.selected_animal}, slalom_mode={self.slalom_mode}, angles=({self.gate_exit_angle_deg}, {self.slalom_exit_angle_deg}, {self.bin_exit_angle_deg}, {self.torpedo_exit_angle_deg})"
                )
        except Exception as e:
            rospy.logwarn(f"Could not get current configuration: {e}")
            rospy.loginfo("Using default values")

        self.gate_search_depth = -0.7
        self.gate_depth = -1.35
        self.roll_depth = -0.8

        self.slalom_depth = -1.1

        self.red_buoy_radius = 2.2
        self.red_buoy_depth = -0.7

        self.torpedo_map_depth = -1.25
        self.torpedo_target_frame = "torpedo_target"
        self.torpedo_realsense_target_frame = "torpedo_target_realsense"
        self.torpedo_fire_frame = "torpedo_fire_frame"
        self.torpedo_shark_fire_frame = "torpedo_shark_fire_frame"
        self.torpedo_sawfish_fire_frame = "torpedo_sawfish_fire_frame"

        self.bin_front_look_depth = -1.3
        self.bin_bottom_look_depth = -0.7

        self.octagon_depth = -0.8

        self.pipeline_depth = -1.5

        # GPS parameters
        self.gps_depth = -1.0
        self.gps_target_frame = "gps_target"

        # Acoustic transmitter parameters
        self.acoustic_tx_data_value = 1
        self.acoustic_tx_publish_rate = 1.0  # Hz
        self.acoustic_tx_duration = 5.0  # seconds

        # Acoustic receiver parameters
        self.acoustic_rx_expected_data = [1, 2, 3]  # Accept any of these values
        self.acoustic_rx_timeout = 30.0  # seconds

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

    def dynamic_reconfigure_callback(self, config):
        """
        Dynamic reconfigure callback for updating mission parameters
        """
        if config is None:
            rospy.logwarn("Could not get parameters from server")
            return

        rospy.loginfo(
            "Received reconfigure request: selected_animal=%s, slalom_mode=%s, gate_exit_angle=%f, slalom_exit_angle=%f, bin_exit_angle=%f, torpedo_exit_angle=%f",
            config.selected_animal,
            config.slalom_mode,
            config.gate_exit_angle,
            config.slalom_exit_angle,
            config.bin_exit_angle,
            config.torpedo_exit_angle,
        )

        # Update parameters
        self.selected_animal = config.selected_animal
        self.slalom_mode = config.slalom_mode
        self.gate_exit_angle_deg = config.gate_exit_angle
        self.slalom_exit_angle_deg = config.slalom_exit_angle
        self.bin_exit_angle_deg = config.bin_exit_angle
        self.torpedo_exit_angle_deg = config.torpedo_exit_angle

    def execute_state_machine(self):
        # Convert degrees to radians
        gate_exit_angle_rad = math.radians(self.gate_exit_angle_deg)
        slalom_exit_angle_rad = math.radians(self.slalom_exit_angle_deg)
        bin_exit_angle_rad = math.radians(self.bin_exit_angle_deg)
        torpedo_exit_angle_rad = math.radians(self.torpedo_exit_angle_deg)

        rospy.loginfo(
            f"Exit angles (degrees): gate={self.gate_exit_angle_deg}, slalom={self.slalom_exit_angle_deg}, bin={self.bin_exit_angle_deg}, torpedo={self.torpedo_exit_angle_deg}"
        )

        # Create torpedo fire frames based on selected animal
        torpedo_fire_frames = (
            [self.torpedo_shark_fire_frame, self.torpedo_sawfish_fire_frame]
            if self.selected_animal == "shark"
            else [self.torpedo_sawfish_fire_frame, self.torpedo_shark_fire_frame]
        )
        rospy.loginfo(f"Torpedo fire frames order: {torpedo_fire_frames}")
        rospy.loginfo(
            f"Exit angles (radians): gate={gate_exit_angle_rad}, slalom={slalom_exit_angle_rad}, bin={bin_exit_angle_rad}, torpedo={torpedo_exit_angle_rad}"
        )

        # Map state names to their corresponding classes and parameters
        state_mapping = {
            "INITIALIZE": (InitializeState, {}),
            "NAVIGATE_THROUGH_GATE": (
                NavigateThroughGateState,
                {
                    "gate_depth": self.gate_depth,
                    "gate_search_depth": self.gate_search_depth,
                    "roll_depth": self.roll_depth,
                    "gate_exit_angle": gate_exit_angle_rad,
                },
            ),
            "NAVIGATE_THROUGH_SLALOM": (
                NavigateThroughSlalomState,
                {
                    "slalom_depth": self.slalom_depth,
                    "slalom_exit_angle": slalom_exit_angle_rad,
                    "slalom_mode": self.slalom_mode,
                },
            ),
            "NAVIGATE_TO_TORPEDO_TASK": (
                TorpedoTaskState,
                {
                    "torpedo_map_depth": self.torpedo_map_depth,
                    "torpedo_target_frame": self.torpedo_target_frame,
                    "torpedo_realsense_target_frame": self.torpedo_realsense_target_frame,
                    "torpedo_exit_angle": torpedo_exit_angle_rad,
                    "torpedo_fire_frames": torpedo_fire_frames,
                },
            ),
            "NAVIGATE_TO_BIN_TASK": (
                BinTaskState,
                {
                    "bin_front_look_depth": self.bin_front_look_depth,
                    "bin_bottom_look_depth": self.bin_bottom_look_depth,
                    "target_selection": self.selected_animal,
                    "bin_exit_angle": bin_exit_angle_rad,
                },
            ),
            "NAVIGATE_TO_OCTAGON_TASK": (
                OctagonTaskState,
                {
                    "octagon_depth": self.octagon_depth,
                    "animal": self.selected_animal,
                },
            ),
            "NAVIGATE_TO_GPS_TARGET": (
                NavigateToGpsTargetState,
                {
                    "gps_depth": self.gps_depth,
                    "gps_target_frame": self.gps_target_frame,
                },
            ),
            "ACOUSTIC_TRANSMITTER": (
                AcousticTransmitter,
                {},
            ),
            "ACOUSTIC_RECEIVER": (
                AcousticReceiver,
                {
                    "expected_data": self.acoustic_rx_expected_data,
                    "timeout": self.acoustic_rx_timeout,
                },
            ),
            "NAVIGATE_RETURN_THROUGH_GATE": (
                NavigateReturnThroughGateState,
                {"station_frame": self.return_home_station},
            ),
            "NAVIGATE_THROUGH_PIPELINE": (
                NavigateThroughPipelineState,
                {"pipeline_depth": self.pipeline_depth},
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
                        "aborted": next_state,
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
        rospy.sleep(1.0)
        rospy.loginfo(f"Final selected animal before execution: {node.selected_animal}")
        node.execute_state_machine()
    except KeyboardInterrupt:
        rospy.loginfo("State machine node interrupted")
