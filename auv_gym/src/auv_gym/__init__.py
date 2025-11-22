"""
AUV-Gym: Reinforcement Learning Training Platform for AUVs
===========================================================

A Gymnasium-compatible RL training environment for Autonomous Underwater Vehicles.

Modules:
    - envs: Gym environment implementations
    - utils: Helper utilities (config manager, domain randomization, etc.)

Usage:
    from auv_gym.envs import AUVEnv
    from auv_gym.utils import ConfigManager

    config = ConfigManager('config/residual_control.yaml')
    env = AUVEnv(config)
"""

__version__ = "0.1.0"
__author__ = "Emin"

# Make key classes easily importable
from auv_gym.envs.auv_env import AUVEnv
from auv_gym.utils.config_manager import ConfigManager

__all__ = [
    "AUVEnv",
    "ConfigManager",
]
