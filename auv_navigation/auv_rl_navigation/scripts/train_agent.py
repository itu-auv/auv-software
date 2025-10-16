#!/usr/bin/env python3
"""
Training Script for AUV RL Navigation
Trains PPO agent in the Gazebo simulation environment
"""

import rospy
import sys
import os
from auv_rl_navigation.environments.auv_nav_env import AUVNavEnv
from auv_rl_navigation.agents.ppo_agent import AUVPPOAgent, TrainingCallback


def main():
    """Main training function."""

    # Initialize ROS node
    rospy.init_node("auv_rl_trainer", anonymous=True)

    rospy.loginfo("=" * 60)
    rospy.loginfo("AUV RL Navigation Training")
    rospy.loginfo("=" * 60)

    # Get parameters from ROS parameter server
    total_timesteps = rospy.get_param("~total_timesteps", 500000)
    save_freq = rospy.get_param("~save_freq", 10000)
    model_save_path = rospy.get_param("~model_save_path", "./models/ppo_auv_best")
    final_model_path = rospy.get_param("~final_model_path", "./models/ppo_auv_final")

    # Environment parameters
    max_episode_steps = rospy.get_param("~max_episode_steps", 500)
    goal_tolerance = rospy.get_param("~goal_tolerance", 1.0)
    goal_frame = rospy.get_param("~goal_frame", "gate")
    visualize = rospy.get_param("~visualize", True)

    # Object frames to track
    object_frames_str = rospy.get_param(
        "~object_frames",
        "gate_shark_link,gate_sawfish_link,red_pipe_link,white_pipe_link,"
        "red_buoy,torpedo_map_link,bin_whole_link,octagon_link",
    )
    object_frames = [f.strip() for f in object_frames_str.split(",")]

    rospy.loginfo(f"Training Parameters:")
    rospy.loginfo(f"  Total timesteps: {total_timesteps}")
    rospy.loginfo(f"  Max episode steps: {max_episode_steps}")
    rospy.loginfo(f"  Goal tolerance: {goal_tolerance}m")
    rospy.loginfo(f"  Goal frame: {goal_frame}")
    rospy.loginfo(f"  Tracking {len(object_frames)} object types")
    rospy.loginfo(f"  Visualization: {'ON' if visualize else 'OFF'}")

    # Create environment
    rospy.loginfo("\nCreating environment...")
    try:
        env = AUVNavEnv(
            max_episode_steps=max_episode_steps,
            goal_tolerance=goal_tolerance,
            object_frames=object_frames,
            goal_frame=goal_frame,
            visualize=visualize,
        )
        rospy.loginfo("Environment created successfully!")
    except Exception as e:
        rospy.logerr(f"Failed to create environment: {e}")
        return

    # PPO configuration
    ppo_config = {
        "learning_rate": rospy.get_param("~learning_rate", 3e-4),
        "n_steps": rospy.get_param("~n_steps", 2048),
        "batch_size": rospy.get_param("~batch_size", 64),
        "n_epochs": rospy.get_param("~n_epochs", 10),
        "gamma": rospy.get_param("~gamma", 0.99),
        "gae_lambda": rospy.get_param("~gae_lambda", 0.95),
        "clip_range": rospy.get_param("~clip_range", 0.2),
        "ent_coef": rospy.get_param("~ent_coef", 0.01),
        "vf_coef": rospy.get_param("~vf_coef", 0.5),
        "max_grad_norm": rospy.get_param("~max_grad_norm", 0.5),
        "features_dim": rospy.get_param("~features_dim", 256),
    }

    rospy.loginfo("\nPPO Configuration:")
    for key, value in ppo_config.items():
        rospy.loginfo(f"  {key}: {value}")

    # Create agent
    rospy.loginfo("\nCreating PPO agent...")
    agent = AUVPPOAgent(env, config=ppo_config)
    rospy.loginfo("Agent created successfully!")

    # Create save directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    # Create callback for saving best models
    callback = TrainingCallback(
        check_freq=save_freq, save_path=model_save_path, verbose=1
    )

    # Start training
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("Starting training...")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"Monitor training with: tensorboard --logdir=./rl_logs/")
    rospy.loginfo("=" * 60 + "\n")

    try:
        agent.train(
            total_timesteps=total_timesteps,
            callback=callback,
            save_path=final_model_path,
        )

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("Training completed successfully!")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Best model saved to: {model_save_path}.zip")
        rospy.loginfo(f"Final model saved to: {final_model_path}.zip")

        # Evaluate the trained agent
        rospy.loginfo("\nEvaluating trained agent...")
        metrics = agent.evaluate(n_episodes=10)

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("Training session finished!")
        rospy.loginfo("=" * 60)

    except KeyboardInterrupt:
        rospy.logwarn("\nTraining interrupted by user!")
        rospy.loginfo(f"Saving current model to: {final_model_path}_interrupted.zip")
        agent.model.save(f"{final_model_path}_interrupted")

    except Exception as e:
        rospy.logerr(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        env.close()
        rospy.loginfo("Environment closed.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Training node terminated.")
        sys.exit(0)
