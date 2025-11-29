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

from auv_reinforcement_learning.envs import AUVEnv
from auv_reinforcement_learning.utils import ConfigManager


def main():
    """Main training loop for navigation mode."""
    rospy.init_node("auv_reinforcement_learning_training_navigation", anonymous=True)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("auv_reinforcement_learning")

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

    # Load pre-trained weights if provided
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_weight", help="Path to pre-trained BC weights", default=None
    )
    # We use parse_known_args because roslaunch might pass other args
    args, _ = parser.parse_known_args()

    if args.pretrained_weight and os.path.exists(args.pretrained_weight):
        rospy.loginfo(f"Loading pre-trained weights from {args.pretrained_weight}")
        bc_state_dict = torch.load(args.pretrained_weight)

        # We assume the BC model was a simple MLP: Input -> Hidden -> Output
        # We need to map this to SB3's Actor network.
        # SB3 SAC Actor structure (MlpPolicy):
        # model.policy.actor.latent_pi (Sequential) -> model.policy.actor.mu (Linear)

        # Attempt to map weights. This is heuristic and depends on exact architectures matching.
        # If sizes mismatch, this will fail (which is good).

        try:
            # Map Feature Extractor / Latent Pi
            # My MLP: net[0] (Linear), net[2] (Linear)
            # SB3: actor.latent_pi[0] (Linear), actor.latent_pi[2] (Linear)

            with torch.no_grad():
                # First Layer
                model.policy.actor.latent_pi[0].weight.copy_(
                    bc_state_dict["net.0.weight"]
                )
                model.policy.actor.latent_pi[0].bias.copy_(bc_state_dict["net.0.bias"])

                # Second Layer
                model.policy.actor.latent_pi[2].weight.copy_(
                    bc_state_dict["net.2.weight"]
                )
                model.policy.actor.latent_pi[2].bias.copy_(bc_state_dict["net.2.bias"])

                # Output Layer (Mu)
                # My MLP: net[4] (Linear)
                # SB3: actor.mu (Linear)
                model.policy.actor.mu.weight.copy_(bc_state_dict["net.4.weight"])
                model.policy.actor.mu.bias.copy_(bc_state_dict["net.4.bias"])

            rospy.loginfo("Successfully loaded pre-trained weights into SAC Actor!")

        except Exception as e:
            rospy.logerr(
                f"Failed to load weights: {e}. Check architecture compatibility."
            )
    else:
        rospy.loginfo(
            "No pre-trained weights provided or file not found. Starting from scratch."
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
