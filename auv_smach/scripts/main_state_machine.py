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
from auv_smach.waypoints import DynamicPathExecutionState
from std_msgs.msg import Bool
import threading
from dynamic_reconfigure.client import Client
from auv_bringup.cfg import SmachParametersConfig


DEFAULT_SELECTED_ROLE = "survey_repair"
DEFAULT_TORPEDO_MAP = "fire"
BIN_FIRE_FIRST_LIST_FRAMES = ["bin_fire_link", "bin_blood_link"]
BIN_BLOOD_FIRST_LIST_FRAMES = ["bin_blood_link", "bin_fire_link"]
LEFT_TOP_TORPEDO_FIRE_FRAMES = ["torpedo_left_fire", "torpedo_top_fire"]
RIGHT_BOTTOM_TORPEDO_FIRE_FRAMES = ["torpedo_right_fire", "torpedo_bottom_fire"]
ROLE_TO_BIN_TARGET_SELECTION = {
    "survey_repair": "shark",
    "search_rescue": "sawfish",
}
ROLE_TO_OCTAGON_TARGET_FRAME = {
    "survey_repair": "octagon_repair_link",
    "search_rescue": "octagon_rescue_link",
}
LEFT_TOP_TORPEDO_FIRE_FRAMES = [
    "torpedo_left_mid_fire_frame",
    "torpedo_top_mid_fire_frame",
]
RIGHT_BOTTOM_TORPEDO_FIRE_FRAMES = [
    "torpedo_bottom_right_fire_frame",
    "torpedo_bottom_mid_fire_frame",
]


class MainStateMachineNode:
    def __init__(self):
        self.sm = None
        self.previous_enabled = False

        # Initialize dynamic reconfigure client
        self.dynamic_reconfigure_client = Client(
            "smach_parameters_server",
            timeout=10,
            config_callback=self.dynamic_reconfigure_callback,
        )

        self.return_home_station = "bin_exit"

        # Get initial values from dynamic reconfigure
        self.selected_role = DEFAULT_SELECTED_ROLE
        self.torpedo_map = DEFAULT_TORPEDO_MAP
        self.slalom_mode = "close"
        self.slalom_direction = "left"
        self.octagon_start_from_table = False

        # Exit angles in degrees (will be converted to radians)
        self.gate_exit_angle_deg = 0.0
        self.slalom_exit_angle_deg = 0.0
        self.bin_exit_angle_deg = 0.0
        self.torpedo_exit_angle_deg = 0.0

        # Get current configuration from server
        try:
            current_config = self.dynamic_reconfigure_client.get_configuration()
            if current_config:
                self.selected_role = current_config.get(
                    "selected_role", DEFAULT_SELECTED_ROLE
                )
                self.torpedo_map = current_config.get(
                    "torpedo_map", DEFAULT_TORPEDO_MAP
                )
                self.slalom_mode = current_config.get("slalom_mode", "close")
                self.slalom_direction = current_config.get("slalom_direction", "left")
                self.gate_exit_angle_deg = current_config.get("gate_exit_angle", 0.0)
                self.slalom_exit_angle_deg = current_config.get(
                    "slalom_exit_angle", 0.0
                )
                self.bin_exit_angle_deg = current_config.get("bin_exit_angle", 0.0)
                self.torpedo_exit_angle_deg = current_config.get(
                    "torpedo_exit_angle", 0.0
                )
                rospy.loginfo(
                    f"Loaded current config: selected_role={self.selected_role}, torpedo_map={self.torpedo_map}, slalom_mode={self.slalom_mode}, angles=({self.gate_exit_angle_deg}, {self.slalom_exit_angle_deg}, {self.bin_exit_angle_deg}, {self.torpedo_exit_angle_deg})"
                )
        except Exception as e:
            rospy.logwarn(f"Could not get current configuration: {e}")
            rospy.loginfo("Using default values")

        self.gate_search_depth = -0.7
        self.gate_depth = -1.35
        self.roll_depth = -0.8

        self.gate_search_frame = "gate_search_rescue_link_kde"
        self.torpedo_search_frame = "torpedo_map_link_kde"
        self.bin_search_frame = "bin_basket_front_link_kde"
        self.octagon_search_frame = "octagon_link_kde"
        self.red_buoy_search_frame = "red_buoy_link_kde"
        self.slalom_search_frame = "slalom_red_pipe_link_kde"

        self.slalom_depth = -1.1

        self.red_buoy_radius = 2.2
        self.red_buoy_depth = -0.7

        self.torpedo_map_depth = -1.25
        self.torpedo_target_frame = "torpedo_target"
        self.torpedo_realsense_target_frame = "torpedo_target_realsense"

        self.bin_front_look_depth = -1.3
        self.bin_bottom_look_depth = -0.7

        self.octagon_depth = -0.8

        self.pipeline_depth = -0.75

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

        selected_role = config.selected_role
        rospy.loginfo(
            "Received reconfigure request: selected_role=%s, torpedo_map=%s, slalom_mode=%s, slalom_direction=%s, gate_exit_angle=%f, slalom_exit_angle=%f, bin_exit_angle=%f, torpedo_exit_angle=%f",
            selected_role,
            config.torpedo_map,
            config.slalom_mode,
            config.slalom_direction,
            config.gate_exit_angle,
            config.slalom_exit_angle,
            config.bin_exit_angle,
            config.torpedo_exit_angle,
        )

        # Update parameters
        self.selected_role = selected_role
        self.torpedo_map = config.torpedo_map
        self.slalom_mode = config.slalom_mode
        self.slalom_direction = config.slalom_direction
        self.gate_exit_angle_deg = config.gate_exit_angle
        self.slalom_exit_angle_deg = config.slalom_exit_angle
        self.bin_exit_angle_deg = config.bin_exit_angle
        self.torpedo_exit_angle_deg = config.torpedo_exit_angle

    def get_bin_target_frames(self):
        is_survey_repair = self.selected_role == DEFAULT_SELECTED_ROLE
        return (
            BIN_FIRE_FIRST_LIST_FRAMES
            if is_survey_repair
            else BIN_BLOOD_FIRST_LIST_FRAMES
        )

    def get_octagon_target_frame(self):
        return ROLE_TO_OCTAGON_TARGET_FRAME.get(
            self.selected_role,
            ROLE_TO_OCTAGON_TARGET_FRAME[DEFAULT_SELECTED_ROLE],
        )

    def get_torpedo_fire_frames(self):
        is_survey_repair = self.selected_role == DEFAULT_SELECTED_ROLE

        if self.torpedo_map == "fire":
            return (
                LEFT_TOP_TORPEDO_FIRE_FRAMES
                if is_survey_repair
                else RIGHT_BOTTOM_TORPEDO_FIRE_FRAMES
            )
        if self.torpedo_map == "blood":
            return (
                RIGHT_BOTTOM_TORPEDO_FIRE_FRAMES
                if is_survey_repair
                else LEFT_TOP_TORPEDO_FIRE_FRAMES
            )

        rospy.logwarn(
            "Unknown torpedo_map '%s'. Using default fire-frame order.",
            self.torpedo_map,
        )
        return LEFT_TOP_TORPEDO_FIRE_FRAMES

    @staticmethod
    def _is_path_token(name):
        return isinstance(name, str) and name.startswith("path") and name[4:].isdigit()

    def _build_path_state(self, token, gui_paths):
        cfg = (gui_paths or {}).get(token)
        if not cfg:
            rospy.logwarn(
                "No GUI path config found for '%s' at /waypoint_gui/paths. "
                "Draw it in the waypoint GUI before running this mission.",
                token,
            )
            return None

        waypoint_frames = cfg.get("waypoint_frames")
        if not waypoint_frames and "waypoints" in cfg:
            count = int(cfg["waypoints"])
            waypoint_frames = [f"{token}_wp{i + 1}" for i in range(count)]
        if not waypoint_frames:
            rospy.logwarn("GUI path '%s' has no waypoints; skipping.", token)
            return None

        reference_frame = cfg.get("reference_frame", f"{token}_ref")
        rospy.loginfo(
            "[main] %s: ref=%s, wps=%s", token, reference_frame, waypoint_frames
        )
        return DynamicPathExecutionState(
            path_name=token,
            reference_frame=reference_frame,
            waypoint_frames=waypoint_frames,
            final_align=True,
        )

    def execute_state_machine(self):
        # Convert degrees to radians
        gate_exit_angle_rad = math.radians(self.gate_exit_angle_deg)
        slalom_exit_angle_rad = math.radians(self.slalom_exit_angle_deg)
        bin_exit_angle_rad = math.radians(self.bin_exit_angle_deg)
        torpedo_exit_angle_rad = math.radians(self.torpedo_exit_angle_deg)

        rospy.loginfo(
            f"Exit angles (degrees): gate={self.gate_exit_angle_deg}, slalom={self.slalom_exit_angle_deg}, bin={self.bin_exit_angle_deg}, torpedo={self.torpedo_exit_angle_deg}"
        )

        bin_target_frames = self.get_bin_target_frames()
        rospy.loginfo(f"Bin target frames order: {bin_target_frames}")
        octagon_target_frame = self.get_octagon_target_frame()

        torpedo_fire_frames = self.get_torpedo_fire_frames()
        rospy.loginfo(f"Torpedo fire frames order: {torpedo_fire_frames}")
        rospy.loginfo(f"Octagon target frame: {octagon_target_frame}")
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
                    "gate_search_frame": self.gate_search_frame,
                },
            ),
            "NAVIGATE_THROUGH_SLALOM": (
                NavigateThroughSlalomState,
                {
                    "slalom_depth": self.slalom_depth,
                    "slalom_exit_angle": slalom_exit_angle_rad,
                    "slalom_direction": self.slalom_direction,
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
                    "torpedo_search_frame": self.torpedo_search_frame,
                },
            ),
            "NAVIGATE_TO_BIN_TASK": (
                BinTaskState,
                {
                    "bin_front_look_depth": self.bin_front_look_depth,
                    "bin_bottom_look_depth": self.bin_bottom_look_depth,
                    "target_frames": bin_target_frames,
                    "bin_exit_angle": bin_exit_angle_rad,
                    "bin_search_frame": self.bin_search_frame,
                },
            ),
            "NAVIGATE_TO_OCTAGON_TASK": (
                OctagonTaskState,
                {
                    "octagon_depth": self.octagon_depth,
                    "animal": self.selected_role,
                    "octagon_search_frame": self.octagon_search_frame,
                    "octagon_role_frame": octagon_target_frame,
                    "start_from_table": self.octagon_start_from_table,
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
                {
                    "station_frame": self.return_home_station,
                    "gate_search_frame": self.gate_search_frame,
                },
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
        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        gui_paths_ns = rospy.get_param("~gui_paths_rosparam", "/waypoint_gui/paths")
        gui_paths = rospy.get_param(gui_paths_ns, None) or {}
        if gui_paths:
            rospy.loginfo(
                "Loaded GUI path configs from %s: %s",
                gui_paths_ns,
                list(gui_paths.keys()),
            )

        with self.sm:
            for i, state_name in enumerate(self.state_list):
                next_state = (
                    self.state_list[i + 1]
                    if i + 1 < len(self.state_list)
                    else "succeeded"
                )

                if self._is_path_token(state_name):
                    state_instance = self._build_path_state(state_name, gui_paths)
                    if state_instance is None:
                        continue
                else:
                    state_class, params = state_mapping.get(state_name, (None, {}))
                    if state_class is None:
                        rospy.logerr(f"Unknown state: {state_name}")
                        continue
                    state_instance = state_class(**params)

                smach.StateMachine.add(
                    state_name,
                    state_instance,
                    transitions={
                        "succeeded": next_state,
                        "preempted": "preempted",
                        "aborted": next_state,
                    },
                )

        # Execute the state machine
        try:
            outcome = self.sm.execute()
            rospy.loginfo(f"State machine exited with outcome: {outcome}")
        except Exception as e:
            rospy.logerr(f"Error executing state machine: {e}")

    def enabled_callback(self, msg):
        falling_edge = self.previous_enabled and not msg.data

        self.previous_enabled = msg.data

        if falling_edge:
            # TODO: maybe add restart logic
            rospy.logerr("KILLSWITCH!")
            rospy.signal_shutdown("Force stopping state machine")


if __name__ == "__main__":
    rospy.init_node("main_state_machine")
    try:
        node = MainStateMachineNode()
        rospy.sleep(1.0)
        rospy.loginfo(f"Final selected role before execution: {node.selected_role}")
        node.execute_state_machine()
    except KeyboardInterrupt:
        rospy.loginfo("State machine node interrupted")
