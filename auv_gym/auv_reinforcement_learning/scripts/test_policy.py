#!/usr/bin/env python3
"""
Test Script: Evaluate Trained Policy
=====================================

Loads a trained model and tests it in the environment.

Usage:
    python test_policy.py --model models/best/best_model.zip --config config/residual_control.yaml --episodes 10
"""

import os
import argparse
import rospy
import numpy as np
from stable_baselines3 import SAC
import rospkg

from auv_reinforcement_learning.envs import AUVEnv
from auv_reinforcement_learning.utils import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test trained AUV policy")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (relative to package root)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (relative to package root)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of test episodes"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render episodes (Gazebo visualization)"
    )
    return parser.parse_args()


def test_policy(model_path, config_path, num_episodes=10, deterministic=True):
    """
    Test a trained policy.

    Args:
        model_path: Path to trained model
        config_path: Path to config file
        num_episodes: Number of episodes to test
        deterministic: Use deterministic actions
    """
    # Load config
    config = ConfigManager(config_path)
    rospy.loginfo(f"Loaded config: {config}")

    # Create environment
    env = AUVEnv(config)
    rospy.loginfo("Environment created")

    # Load model
    model = SAC.load(model_path)
    rospy.loginfo(f"Model loaded from: {model_path}")

    # Test episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0

        rospy.loginfo(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        rospy.loginfo(f"Target: {info.get('target_position', 'N/A')}")

        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # Log progress
            if step % 50 == 0:
                rospy.loginfo(
                    f"Step {step}: pos_error={info['position_error']:.3f}m, "
                    f"reward={reward:.2f}"
                )

        # Episode finished
        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        if terminated:  # Goal reached
            success_count += 1
            rospy.loginfo(f"✓ SUCCESS! Reached goal in {step} steps")
        else:
            rospy.loginfo(f"✗ FAILED after {step} steps")

        rospy.loginfo(
            f"Total reward: {total_reward:.2f}, "
            f"Final error: {info['position_error']:.3f}m"
        )

    # Summary statistics
    rospy.loginfo("\n" + "=" * 50)
    rospy.loginfo("TEST SUMMARY")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"Episodes: {num_episodes}")
    rospy.loginfo(
        f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)"
    )
    rospy.loginfo(
        f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    rospy.loginfo(
        f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    rospy.loginfo(f"Best reward: {np.max(episode_rewards):.2f}")
    rospy.loginfo(f"Worst reward: {np.min(episode_rewards):.2f}")

    env.close()

    return {
        "success_rate": success_count / num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }


def main():
    """Main function."""
    args = parse_args()

    rospy.init_node("auv_reinforcement_learning_test", anonymous=True)

    # Get package path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("auv_reinforcement_learning")

    # Resolve paths
    model_path = os.path.join(pkg_path, args.model)
    config_path = os.path.join(pkg_path, args.config)

    # Check files exist
    if not os.path.exists(model_path):
        rospy.logerr(f"Model not found: {model_path}")
        return

    if not os.path.exists(config_path):
        rospy.logerr(f"Config not found: {config_path}")
        return

    # Test policy
    rospy.loginfo(f"Testing policy: {model_path}")
    rospy.loginfo(f"Config: {config_path}")

    results = test_policy(
        model_path,
        config_path,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
    )

    rospy.loginfo("Testing complete!")


if __name__ == "__main__":
    main()
