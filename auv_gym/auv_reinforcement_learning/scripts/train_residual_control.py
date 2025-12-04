#!/usr/bin/env python3
"""
Training Script: Residual Control
==================================

Trains an RL agent to provide corrections to a PID baseline controller.
Uses SAC algorithm from Stable-Baselines3.

Usage:
    roslaunch auv_reinforcement_learning train_residual.launch
    or
    python train_residual_control.py
"""

import os
import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import rospkg

from auv_reinforcement_learning.envs import AUVEnv
from auv_reinforcement_learning.utils import ConfigManager


def main():
    """Main training loop."""
    rospy.init_node("auv_reinforcement_learning_training", anonymous=True)

    # Get package path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("auv_reinforcement_learning")

    # Load configuration
    config_path = os.path.join(pkg_path, "config", "residual_control.yaml")
    config = ConfigManager(config_path)
    rospy.loginfo(f"Loaded config: {config}")

    # Create environment
    env = AUVEnv(config)
    env = Monitor(env)  # Wrap with Monitor for logging

    # Create model
    rospy.loginfo("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=os.path.join(pkg_path, "logs", "sac_residual"),
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(pkg_path, "models", "checkpoints"),
        name_prefix="sac_residual",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(pkg_path, "models", "best"),
        log_path=os.path.join(pkg_path, "logs", "eval"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    rospy.loginfo("Starting training...")
    try:
        model.learn(
            total_timesteps=2_000_000,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
        )
    except KeyboardInterrupt:
        rospy.logwarn("Training interrupted by user")

    # Save final model
    final_model_path = os.path.join(pkg_path, "models", "sac_residual_final")
    model.save(final_model_path)
    rospy.loginfo(f"Training complete! Model saved to: {final_model_path}")

    env.close()


if __name__ == "__main__":
    main()
