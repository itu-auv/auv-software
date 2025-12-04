#!/usr/bin/env python3
"""
Training script for Behavior Cloning (BC) using the imitation library and Stable Baselines3.
loads an expert dataset, converts it to the required format, and trains a BC agent.
"""

import argparse
import os
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data.types import Transitions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Loading dataset from {dataset_path}...")
    try:
        data = np.load(dataset_path)
        observations = data["inputs"]
        actions = data["labels"]
        logger.info(f"Dataset loaded. Shape: Obs={observations.shape}, Acts={actions.shape}")
        return observations, actions
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def create_transitions(observations: np.ndarray, actions: np.ndarray) -> Transitions:
    """
    Convert observations and actions into imitation Transitions object.
    
    Since we only have (s, a) pairs, we create dummy next_obs and dones.

    Args:
        observations: Array of observations.
        actions: Array of actions.

    Returns:
        Transitions object containing the demonstration data.
    """
    
    # The imitation library expects data in the standard Reinforcement Learning transition format: (s, a, s', d) (observation, action, next_observation, done).
    # Create dummy next_obs and dones since they are not used for BC loss
    dummy_next_obs = np.zeros_like(observations)
    dummy_dones = np.zeros(len(observations), dtype=bool)
    
    return Transitions(
        obs=observations,
        acts=actions,
        infos=[{}] * len(observations),
        next_obs=dummy_next_obs,
        dones=dummy_dones
    )


def get_space(data: np.ndarray) -> gym.spaces.Box:
    """
    Create a gym Space based on the data shape.
    Assumes continuous space with infinite bounds.
    """
    dim = data.shape[1]
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def train_bc_agent(
    dataset_path: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    rng_seed: int = 0
) -> None:
    """
    Args:
        dataset_path: Path to the dataset.
        output_path: Path to save the trained policy.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        rng_seed: Random seed.
    """
    observations, actions = load_dataset(dataset_path)
    transitions = create_transitions(observations, actions)
    
    logger.info("Initializing BC trainer...")
    rng = np.random.default_rng(rng_seed)
    
    bc_trainer = bc.BC(
        observation_space=get_space(observations),
        action_space=get_space(actions),
        demonstrations=transitions,
        rng=rng,
        batch_size=batch_size,
        policy=None,  # use default MlpPolicy
    )
    
    logger.info(f"Training for {epochs} epochs...")
    bc_trainer.train(n_epochs=epochs)
    
    logger.info(f"Saving policy to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bc_trainer.policy.save(output_path)
    logger.info("Training completed and policy saved.")


def main():
    parser = argparse.ArgumentParser(description="Train BC Agent using imitation library")
    parser.add_argument("--dataset", required=True, help="Path to NPZ dataset")
    parser.add_argument(
        "--output_path",
        default="models/bc_policy.zip",
        help="Path to save trained model",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    train_bc_agent(
        dataset_path=args.dataset,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
