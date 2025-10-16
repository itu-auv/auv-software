# AUV RL Navigation

Reinforcement Learning-based navigation system for autonomous underwater vehicles (AUV).

## Overview

This package implements a PPO (Proximal Policy Optimization) based navigation agent that learns to navigate using:
- **3D Voxel Grid** (10×10×2×8): Vehicle-centric spatial representation of objects
- **Goal Vector**: Relative position to target
- **Vehicle State**: Velocity and orientation

## Installation

### 1. Install Python Dependencies

```bash
cd ~/catkin_ws/src/auv-software/auv_navigation/auv_rl_navigation
pip3 install -r requirements.txt
```

### 2. Build the Package

```bash
cd ~/catkin_ws
catkin build auv_rl_navigation
source devel/setup.bash
```

## Usage

### Training the Agent

1. **Start your simulation environment** (Gazebo/Holoocean with AUV and objects)

2. **Launch training:**
```bash
roslaunch auv_rl_navigation training.launch
```

3. **Monitor training with TensorBoard:**
```bash
tensorboard --logdir=./rl_logs/
```
Open browser at `http://localhost:6006`

### Configuration

Edit `config/rl_params.yaml` to adjust:
- Training parameters (timesteps, episode length)
- Environment settings (goal frame, object frames)
- PPO hyperparameters (learning rate, batch size, etc.)

### Trained Models

Models are saved to:
- `./models/ppo_auv_best.zip` - Best model during training
- `./models/ppo_auv_final.zip` - Final model after training

## Architecture

### Components

1. **WorldGridEncoder** (`observation/world_grid_encoder.py`)
   - Converts TF transforms to 10×10×2×8 voxel grid
   - Tracks 8 object types in 3D space
   - Real-time RViz visualization

2. **AUVNavEnv** (`environments/auv_nav_env.py`)
   - OpenAI Gym environment
   - Interfaces with ROS topics (cmd_vel, odom)
   - Reward shaping for navigation

3. **AUVPPOAgent** (`agents/ppo_agent.py`)
   - Custom CNN for spatial grid processing
   - MLP for vector observations
   - PPO algorithm for policy learning

### Observation Space

```python
{
  'grid': (10, 10, 2, 8),      # 3D voxel grid
  'goal_vector': (3,),          # [x, y, z] to goal
  'velocity': (6,)              # [vx, vy, vz, wx, wy, wz]
}
```

### Action Space

```python
[surge, sway, heave, yaw_rate]  # Range: [-1, 1]
```

Scaled to:
- Surge: ±1.0 m/s
- Sway: ±0.5 m/s
- Heave: ±0.5 m/s
- Yaw rate: ±0.5 rad/s

### Reward Function

- **+100**: Goal reached
- **+10×progress**: Moving towards goal
- **-10**: Episode timeout
- **-0.1**: Per-step penalty (efficiency)

## Visualization

The grid encoder publishes visualization markers to:
- `/auv_rl/grid_visualization` - 3D voxel grid
- `/auv_rl/object_markers` - Detected objects

View in RViz by adding MarkerArray displays for these topics.

## Parameters

Key parameters in `config/rl_params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 500000 | Total training steps |
| `max_episode_steps` | 500 | Max steps per episode |
| `goal_tolerance` | 1.0 | Distance to goal (m) |
| `goal_frame` | "gate" | Target TF frame |
| `learning_rate` | 0.0003 | PPO learning rate |
| `batch_size` | 64 | Training batch size |

## Troubleshooting

### TF Lookup Errors
Ensure all object frames are being published:
```bash
rosrun tf tf_echo taluy/base_link gate
```

### Low Performance
- Increase `total_timesteps` for longer training
- Adjust `learning_rate` (try 1e-4 or 1e-3)
- Check reward values with TensorBoard

### Memory Issues
- Reduce `n_steps` (e.g., 1024)
- Reduce `batch_size` (e.g., 32)
- Close visualization (`visualize: false`)

## Citation

If you use this code, please cite:
```
@software{auv_rl_navigation,
  title={AUV RL Navigation},
  author={ITU AUV Team},
  year={2025}
}
```
