#!/usr/bin/env python3
"""
PPO Agent for AUV Navigation
Custom neural network architecture for processing 3D voxel grid + vector observations
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np


class Custom3DGridExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 3D voxel grid (10x10x2x8).
    Processes the spatial grid and concatenates with vector observations.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        """
        Initialize the feature extractor.

        Args:
            observation_space: Dictionary observation space containing:
                - 'grid': (10, 10, 2, 8) 3D voxel grid
                - 'goal_vector': (3,) relative goal position
                - 'velocity': (6,) vehicle velocity state
            features_dim: Output feature dimension
        """
        super(Custom3DGridExtractor, self).__init__(observation_space, features_dim)

        # Extract dimensions from observation space
        grid_shape = observation_space["grid"].shape  # (10, 10, 2, 8)
        goal_dim = observation_space["goal_vector"].shape[0]  # 3
        velocity_dim = observation_space["velocity"].shape[0]  # 6

        # Reshape grid: (10, 10, 2, 8) -> treat as (16 channels, 10, 10) for 2D CNN
        # We'll flatten the z-dimension (2) and object channels (8) -> 2*8=16 channels
        self.n_input_channels = grid_shape[2] * grid_shape[3]  # 2 * 8 = 16

        # 3D Grid processing: 2D CNN (treating z-layers and object channels as channels)
        self.grid_cnn = nn.Sequential(
            # Conv layer 1: 16 -> 32 channels
            nn.Conv2d(self.n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10x10 -> 5x5
            # Conv layer 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5x5 -> 2x2
            # Conv layer 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
        )

        # Calculate CNN output size
        # After convolutions: 128 channels * 1 * 1 = 128
        cnn_output_dim = 128

        # Vector processing: MLP for goal_vector and velocity
        vector_input_dim = goal_dim + velocity_dim  # 3 + 6 = 9
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        # Combine CNN and MLP features
        combined_dim = cnn_output_dim + 64  # 128 + 64 = 192
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Dictionary containing grid, goal_vector, and velocity

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, features_dim)
        """
        # Extract components
        grid = observations["grid"]  # (batch, 10, 10, 2, 8)
        goal_vector = observations["goal_vector"]  # (batch, 3)
        velocity = observations["velocity"]  # (batch, 6)

        # Reshape grid: (batch, 10, 10, 2, 8) -> (batch, 16, 10, 10)
        batch_size = grid.shape[0]
        grid_reshaped = grid.permute(
            0, 3, 4, 1, 2
        )  # (batch, 10, 10, 2, 8) -> (batch, 2, 8, 10, 10)
        grid_reshaped = grid_reshaped.reshape(batch_size, self.n_input_channels, 10, 10)

        # Process grid through CNN
        grid_features = self.grid_cnn(grid_reshaped)  # (batch, 128)

        # Concatenate goal and velocity vectors
        vector_input = torch.cat([goal_vector, velocity], dim=1)  # (batch, 9)

        # Process vectors through MLP
        vector_features = self.vector_mlp(vector_input)  # (batch, 64)

        # Combine features
        combined = torch.cat([grid_features, vector_features], dim=1)  # (batch, 192)

        # Final processing
        output = self.combined_mlp(combined)  # (batch, features_dim)

        return output


class TrainingCallback(BaseCallback):
    """
    Callback for logging training metrics and saving best models.
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        """
        Initialize callback.

        Args:
            check_freq: Frequency (in steps) to check for best model
            save_path: Path to save the best model
            verbose: Verbosity level
        """
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        """
        Called at each step.

        Returns:
            bool: If False, training will be stopped
        """
        if self.n_calls % self.check_freq == 0:
            # Get training metrics
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean(
                    [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                )
                mean_length = np.mean(
                    [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                )

                if self.verbose > 0:
                    print(f"Step: {self.num_timesteps}")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Mean episode length: {mean_length:.2f}")

                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"New best mean reward: {self.best_mean_reward:.2f}")
                        print(f"Saving model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class AUVPPOAgent:
    """
    PPO Agent wrapper for AUV navigation.
    """

    def __init__(self, env, config=None):
        """
        Initialize PPO agent.

        Args:
            env: Gym environment (AUVNavEnv)
            config: Configuration dictionary for PPO hyperparameters
        """
        self.env = env

        # Default configuration
        default_config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "features_dim": 256,
        }

        # Update with user config
        if config is not None:
            default_config.update(config)

        self.config = default_config

        # Policy architecture
        policy_kwargs = dict(
            features_extractor_class=Custom3DGridExtractor,
            features_extractor_kwargs=dict(features_dim=self.config["features_dim"]),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Actor and Critic networks
        )

        # Create PPO model
        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            clip_range_vf=self.config["clip_range_vf"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./rl_logs/",
        )

    def train(self, total_timesteps, callback=None, save_path="ppo_auv_navigation"):
        """
        Train the agent.

        Args:
            total_timesteps: Total number of training steps
            callback: Training callback
            save_path: Path to save the final model
        """
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        print(f"Configuration: {self.config}")

        # Train
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save final model
        self.model.save(save_path)
        print(f"Training completed. Model saved to {save_path}.zip")

    def load(self, model_path):
        """
        Load a trained model.

        Args:
            model_path: Path to the model file
        """
        self.model = PPO.load(model_path, env=self.env)
        print(f"Model loaded from {model_path}")

    def predict(self, observation, deterministic=True):
        """
        Predict action given observation.

        Args:
            observation: Current observation
            deterministic: If True, use deterministic policy (no exploration)

        Returns:
            action: Predicted action
            state: Internal state (if recurrent policy)
        """
        return self.model.predict(observation, deterministic=deterministic)

    def evaluate(self, n_episodes=10):
        """
        Evaluate the agent.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            dict: Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0

        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                if done and info.get("termination_reason") == "goal_reached":
                    success_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Reward={episode_reward:.2f}, Length={episode_length}, "
                f"Reason={info.get('termination_reason', 'unknown')}"
            )

        metrics = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": success_count / n_episodes,
        }

        print("\nEvaluation Results:")
        print(
            f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
        )
        print(f"Mean Episode Length: {metrics['mean_length']:.2f}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")

        return metrics
