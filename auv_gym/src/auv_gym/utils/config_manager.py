#!/usr/bin/env python3
"""
Config Manager for AUV-Gym Platform
Loads and validates YAML configuration files.
"""

import yaml
import os
from typing import Dict, Any, List
import numpy as np


class ConfigManager:
    """
    Manages configuration files for AUV-Gym environment.

    Responsibilities:
    - Load YAML config files
    - Validate required fields
    - Provide type-safe access to configuration values
    - Support for task-specific validation
    """

    # Required fields for all tasks
    REQUIRED_BASE_FIELDS = [
        "task",
        "action_type",
        "reward_weights",
        "max_episode_steps",
    ]

    # Task-specific required fields
    TASK_SPECIFIC_FIELDS = {
        "ResidualControl": ["pid_controller", "observation_keys"],
        "EndToEndControl": ["observation_keys"],
        "Navigation": ["navigation"],
    }

    # Valid task types
    VALID_TASKS = ["ResidualControl", "EndToEndControl", "Navigation"]

    # Valid action types
    VALID_ACTION_TYPES = ["wrench", "cmd_vel"]

    def __init__(self, config_path: str):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_path = config_path
        self.config = self._load_config()
        self._validate()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _validate(self):
        """Validate configuration file."""
        # Check required base fields
        for field in self.REQUIRED_BASE_FIELDS:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: '{field}'")

        # Check task type is valid
        task = self.config["task"]
        if task not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task type: '{task}'. " f"Valid options: {self.VALID_TASKS}"
            )

        # Check action type is valid
        action_type = self.config["action_type"]
        if action_type not in self.VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action_type: '{action_type}'. "
                f"Valid options: {self.VALID_ACTION_TYPES}"
            )

        # Task-specific validation
        if task in self.TASK_SPECIFIC_FIELDS:
            for field in self.TASK_SPECIFIC_FIELDS[task]:
                if field not in self.config:
                    raise ValueError(f"Task '{task}' requires field: '{field}'")

        if task == "Navigation":
            self._validate_navigation_config()

        # Validate reward weights
        self._validate_reward_weights()

        # Validate action scaling if present
        if "action_scaling" in self.config:
            self._validate_action_scaling()

        # Validate domain randomization config if present
        if "domain_randomization" in self.config:
            self._validate_domain_randomization()

    def _validate_reward_weights(self):
        """Validate reward weights section."""
        weights = self.config["reward_weights"]
        if not isinstance(weights, dict):
            raise ValueError("reward_weights must be a dictionary")

        # Check that all weights are numeric
        for key, value in weights.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Reward weight '{key}' must be numeric, got {type(value)}"
                )

    def _validate_action_scaling(self):
        """Validate action scaling array."""
        scaling = self.config["action_scaling"]
        if not isinstance(scaling, list):
            raise ValueError("action_scaling must be a list")

        if len(scaling) == 0:
            raise ValueError("action_scaling must contain at least one value")

    def _validate_domain_randomization(self):
        """Validate domain randomization configuration."""
        dr_config = self.config["domain_randomization"]

        if not isinstance(dr_config, dict):
            raise ValueError("domain_randomization must be a dictionary")

        # Validate physics DR if present
        if "physics" in dr_config:
            physics = dr_config["physics"]
            for key, value in physics.items():
                if "_range" in key:
                    if not isinstance(value, list) or len(value) != 2:
                        raise ValueError(
                            f"DR physics '{key}' must be [min, max], got {value}"
                        )
                    if value[0] > value[1]:
                        raise ValueError(f"DR physics '{key}': min must be <= max")

        # Validate controller DR if present
        if "controller" in dr_config:
            controller = dr_config["controller"]
            for key, value in controller.items():
                if "_range" in key:
                    if not isinstance(value, list) or len(value) != 2:
                        raise ValueError(
                            f"DR controller '{key}' must be [min, max], got {value}"
                        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            >>> config.get('task')
            'ResidualControl'
            >>> config.get('reward_weights.w_pose')
            1.0
            >>> config.get('nonexistent', 'default_value')
            'default_value'
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_observation_dim(self) -> int:
        """
        Calculate observation space dimension based on task type.

        Returns:
            Dimension of observation space
        """
        task = self.config["task"]

        if task == "ResidualControl":
            # error_pose(6) + error_vel(6) + action_pid(6)
            return 18
        elif task == "EndToEndControl":
            # error_pose(6) + error_vel(6)
            return 12
        elif task == "Navigation":
            dim = 0
            observation_keys = self.get("observation_keys", [])
            if "base_to_gate_transform" in observation_keys:
                dim += 6  # [dx, dy, dz, roll, pitch, yaw]
            if "body_velocity" in observation_keys:
                dim += 4  # [vx, vy, vz, yaw_rate]
            return dim
        else:
            raise ValueError(f"Unknown task type: {task}")

    def get_action_dim(self) -> int:
        """
        Get action space dimension.

        Returns:
            Dimension of action space
        """
        scaling = self.config.get("action_scaling")
        if isinstance(scaling, list) and len(scaling) > 0:
            return len(scaling)
        return 6  # Default fallback, should not happen with validated configs

    def get_action_bounds(self) -> tuple:
        """
        Get action space bounds.

        Returns:
            Tuple of (low, high) bounds for actions
        """
        # Actions are normalized to [-1, 1]
        return (-1.0, 1.0)

    def get_ros_namespace(self) -> str:
        """
        Get ROS namespace for topics.

        Returns:
            ROS namespace (default: 'taluy')
        """
        return self.config.get("ros_namespace", "taluy")

    def get_simulation_dt(self) -> float:
        """
        Get simulation time step.

        Returns:
            Simulation dt in seconds (default: 0.1)
        """
        return self.config.get("simulation_dt", 0.1)

    def is_residual_mode(self) -> bool:
        """Check if using residual control mode."""
        return self.config["task"] == "ResidualControl"

    def is_navigation_mode(self) -> bool:
        """Check if using navigation task."""
        return self.config["task"] == "Navigation"

    def _validate_navigation_config(self):
        """Validate navigation-specific settings."""
        nav_cfg = self.config.get("navigation")
        if not isinstance(nav_cfg, dict):
            raise ValueError("navigation section must be a dictionary")

        for key in ["source_frame", "target_frame"]:
            if key not in nav_cfg:
                raise ValueError(f"navigation config requires '{key}'")
            if not isinstance(nav_cfg[key], str) or not nav_cfg[key]:
                raise ValueError(f"navigation '{key}' must be a non-empty string")

        if "transform_timeout" in nav_cfg:
            timeout = nav_cfg["transform_timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0.0:
                raise ValueError("navigation.transform_timeout must be > 0")

    def has_domain_randomization(self) -> bool:
        """Check if domain randomization is enabled."""
        return "domain_randomization" in self.config

    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return (
            f"ConfigManager(\n"
            f"  task={self.config['task']},\n"
            f"  action_type={self.config['action_type']},\n"
            f"  obs_dim={self.get_observation_dim()},\n"
            f"  action_dim={self.get_action_dim()}\n"
            f")"
        )


# Utility function for quick config loading
def load_config(config_path: str) -> ConfigManager:
    """
    Quick utility to load a config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)
