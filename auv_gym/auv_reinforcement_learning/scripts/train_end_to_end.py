#!/usr/bin/env python3
"""
Training Script: End-to-End Control
====================================

Trains an RL agent to directly output thruster commands (no baseline controller).
Uses SAC algorithm from Stable-Baselines3.

Usage:
    python train_end_to_end.py
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
    rospy.init_node("auv_reinforcement_learning_training_e2e", anonymous=True)

    # Get package path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("auv_reinforcement_learning")

    # Load configuration
    config_path = os.path.join(pkg_path, "config", "end_to_end_control.yaml")
    config = ConfigManager(config_path)
    rospy.loginfo(f"Loaded config: {config}")

    # Create environment
    env = AUVEnv(config)
    env = Monitor(env)

    # Create model (might need different hyperparameters for E2E)
    rospy.loginfo("Creating SAC model for end-to-end control...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-3,  # Higher LR for learning from scratch
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",  # Automatic entropy tuning
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=os.path.join(pkg_path, "logs", "sac_e2e"),
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(pkg_path, "models", "checkpoints_e2e"),
        name_prefix="sac_e2e",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(pkg_path, "models", "best_e2e"),
        log_path=os.path.join(pkg_path, "logs", "eval_e2e"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train (longer training time for E2E)
    rospy.loginfo("Starting training (this will take longer than residual)...")
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
        )
    except KeyboardInterrupt:
        rospy.logwarn("Training interrupted by user")

    # Save final model
    final_model_path = os.path.join(pkg_path, "models", "sac_e2e_final")
    model.save(final_model_path)
    rospy.loginfo(f"Training complete! Model saved to: {final_model_path}")

    env.close()


if __name__ == "__main__":
    main()
