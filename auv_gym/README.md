# AUV-Gym

Reinforcement Learning training platform for Autonomous Underwater Vehicles (AUVs).

## Overview

AUV-Gym is a Gymnasium-compatible environment for training RL agents on AUV control and navigation tasks. It integrates with ROS1 and Gazebo simulation.

## Features

- **3 Task Types**:
  - Residual Control: RL adds corrections to PID baseline
  - End-to-End Control: RL directly outputs thruster commands
  - Navigation: RL commands body-frame velocities to align with the gate frames

- **Flexible Configuration**: YAML-based config system
- **Domain Randomization**: Physics and controller randomization for sim-to-real transfer
- **Stable-Baselines3 Integration**: Ready-to-use training scripts

## Installation

### Prerequisites

```bash
# ROS1 Noetic
# Gazebo
# uuv_simulator

# Python dependencies
pip install gymnasium stable-baselines3 pyyaml numpy tensorboard
```

### Build

```bash
cd ~/catkin_ws
catkin build auv_gym
source devel/setup.bash
```

## Quick Start

### 1. Start Gazebo Simulation

```bash
roslaunch auv_sim_bringup start_gazebo.launch
```

### 2. Train an Agent

```bash
# Residual control (fastest to train)
rosrun auv_gym train_residual_control.py

# End-to-end control
rosrun auv_gym train_end_to_end.py

# Navigation task
rosrun auv_gym train_navigation.py
```

### 3. Test Trained Policy

```bash
rosrun auv_gym test_policy.py \
    --model models/best/best_model.zip \
    --config config/residual_control.yaml \
    --episodes 10 \
    --deterministic
```

## Configuration

Edit YAML files in `config/`:

- `residual_control.yaml` - PID + RL hybrid
- `end_to_end_control.yaml` - Pure RL control
- `navigation.yaml` - TF-based gate alignment navigation

Example configuration structure:

```yaml
task: "ResidualControl"
action_type: "wrench"
action_scaling: [100.0, 100.0, 100.0, 20.0, 20.0, 20.0]

reward_weights:
  w_pose: 1.0
  w_vel: 0.5
  w_effort_rl: 0.1

max_episode_steps: 1000
simulation_dt: 0.1
```

## Navigation Task

The navigation mode observes the TF transform from `taluy/base_link` to `gate_sawfish_link` (or any frames you configure) and outputs body-frame velocity commands `[vx, vy, vz, yaw_rate]` via `/taluy/cmd_vel`. Customize the frames, tolerances, and reward weights in `config/navigation.yaml`, then start training with `rosrun auv_gym train_navigation.py`.

## Package Structure

```
auv_gym/
├── config/              # YAML configuration files
├── launch/              # ROS launch files
├── scripts/             # Training and testing scripts
├── src/auv_gym/
│   ├── envs/           # Environment implementations
│   └── utils/          # Utilities (config manager, etc.)
└── tests/              # Unit tests
```

## Usage Example (Python)

```python
from auv_gym.envs import AUVEnv
from auv_gym.utils import ConfigManager

# Load configuration
config = ConfigManager('config/residual_control.yaml')

# Create environment
env = AUVEnv(config)

# Training loop
obs, info = env.reset()
for _ in range(1000):
    action = policy(obs)  # Your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir auv_gym/logs/
```

## Contributing

See the main project plan in `AUV_GYM_PROJECT_PLAN.md` for development roadmap.

## License

MIT

## Authors

- Emin

## References

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [uuv_simulator](https://github.com/uuvsimulator/uuv_simulator)
