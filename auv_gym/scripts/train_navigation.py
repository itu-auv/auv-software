#!/usr/bin/env python3
"""
Training Script: Navigation
===========================

Trains an RL agent to command body-frame velocities that align the vehicle
with a detected gate frame using the navigation task configuration.
"""

import os
import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import rospkg

from auv_gym.envs import AUVEnv
from auv_gym.utils import ConfigManager


def main():
    """Main training loop for navigation mode."""
    rospy.init_node("auv_gym_training_navigation", anonymous=True)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("auv_gym")

    config_path = os.path.join(pkg_path, "config", "navigation.yaml")
    config = ConfigManager(config_path)
    rospy.loginfo(f"Loaded navigation config: {config}")

    env = AUVEnv(config)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    rospy.loginfo("Creating SAC model for navigation task...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=os.path.join(pkg_path, "logs", "sac_navigation"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=15000,
        save_path=os.path.join(pkg_path, "models", "checkpoints_navigation"),
        name_prefix="sac_navigation",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(pkg_path, "models", "best_navigation"),
        log_path=os.path.join(pkg_path, "logs", "eval_navigation"),
        eval_freq=7500,
        n_eval_episodes=10,
        deterministic=True,
    )

    rospy.loginfo("Starting navigation training...")
    try:
        model.learn(
            total_timesteps=3_000_000,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
        )
    except KeyboardInterrupt:
        rospy.logwarn("Navigation training interrupted by user")

    final_model_path = os.path.join(pkg_path, "models", "sac_navigation_final")
    model.save(final_model_path)
    rospy.loginfo(f"Navigation training complete! Model saved to: {final_model_path}")

    env.close()


if __name__ == "__main__":
    main()
