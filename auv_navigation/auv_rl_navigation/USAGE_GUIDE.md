# AUV RL Navigation - Complete Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
cd ~/catkin_ws/src/auv-software/auv_navigation/auv_rl_navigation
pip3 install -r requirements.txt
```

### 2. Build Package
```bash
cd ~/catkin_ws
catkin build auv_rl_navigation
source devel/setup.bash
```

### 3. Test Components (Before Training)
```bash
# Start your simulation first!
roslaunch auv_rl_navigation test_components.launch
```

---

## Testing & Debugging

### Test Components Script
**Purpose:** Verify all components work before training

```bash
roslaunch auv_rl_navigation test_components.launch
```

**Tests performed:**
1. ✅ ROS Topics connectivity (`/taluy/odom`, `/taluy/cmd_vel`)
2. ✅ TF transforms (base_link → goal_frame)
3. ✅ World Grid Encoder (with visualization)
4. ✅ Gym Environment (reset, step, observations)
5. ✅ PPO Agent (network creation, prediction, short training)

**Skip specific tests:**
```bash
roslaunch auv_rl_navigation test_components.launch skip_agent:=true
```

**Expected output:**
```
====================================================================
# AUV RL Navigation - Component Tests
====================================================================

TEST 1: ROS Topics
    ✓ /taluy/odom - Found
    ✓ /taluy/cmd_vel - Found
    ✓ TF transform taluy/base_link → gate available
✅ ROS Topics Test PASSED

TEST 2: World Grid Encoder
    ✓ Grid encoder created successfully
    - Grid dimensions: 10x10x2
    - Coverage: ±3.5m
✅ Grid Encoder Test PASSED

[... more tests ...]

🎉 All tests passed! Ready for training.
```

---

## Training

### Basic Training
```bash
# Start simulation first!
roslaunch auv_rl_navigation training.launch
```

### Monitor Training Progress
Open **3 terminals**:

**Terminal 1 - Training:**
```bash
roslaunch auv_rl_navigation training.launch
```

**Terminal 2 - TensorBoard:**
```bash
cd ~/catkin_ws/src/auv-software/auv_navigation/auv_rl_navigation
tensorboard --logdir=./rl_logs/
# Open http://localhost:6006 in browser
```

**Terminal 3 - Live Plots:**
```bash
rosrun auv_rl_navigation plot_training_progress.py
```

This shows real-time plots:
- Episode rewards (with moving average)
- Episode lengths
- Success rate (rolling 20-episode window)
- Distance to goal over time

### Customize Training Parameters

Edit `config/rl_params.yaml`:
```yaml
# Quick test run
total_timesteps: 100000      # Shorter training
max_episode_steps: 200       # Shorter episodes

# Full training
total_timesteps: 1000000     # Longer training
max_episode_steps: 500       # Standard episodes
```

Or override via launch file:
```bash
roslaunch auv_rl_navigation training.launch total_timesteps:=1000000
```

---

## Deployment (Running Trained Agent)

### Basic Deployment
```bash
# Make sure your trained model exists!
roslaunch auv_rl_navigation deploy_agent.launch
```

### Specify Model Path
```bash
roslaunch auv_rl_navigation deploy_agent.launch \
    model_path:=/path/to/your/model.zip \
    goal_frame:=gate
```

### Control the Agent

**Enable/Disable agent:**
```bash
# Disable agent (stops sending commands)
rostopic pub /rl_agent_node/enable std_msgs/Bool "data: false"

# Enable agent
rostopic pub /rl_agent_node/enable std_msgs/Bool "data: true"
```

**Reset episode:**
```bash
rostopic pub /rl_agent_node/reset std_msgs/Bool "data: true"
```

### Monitor Deployed Agent

**Terminal 1 - Agent:**
```bash
roslaunch auv_rl_navigation deploy_agent.launch
```

**Terminal 2 - Live Action Monitor:**
```bash
rosrun auv_rl_navigation live_action_monitor.py
```

This shows:
- Agent's actions in real-time (surge, sway, heave, yaw)
- Distance to goal
- Reward per step
- Velocity commands sent to robot

**Terminal 3 - Training Progress (works for deployment too):**
```bash
rosrun auv_rl_navigation plot_training_progress.py
```

---

## Visualization in RViz

### Add Visualization Markers

1. Open RViz
2. Add MarkerArray displays:
   - Topic: `/auv_rl/grid_visualization` (3D voxel grid)
   - Topic: `/auv_rl/object_markers` (detected objects)

### What You'll See

- **Grid Boundary**: Gray wireframe showing the 10×10×2 grid
- **Occupied Cells**: Colored cubes showing where objects are detected
  - Color = object type
  - Opacity = proximity (closer = more opaque)
- **Object Markers**: Spheres at actual object positions with labels

---

## Configuration

### Key Parameters (`config/rl_params.yaml`)

#### Training
```yaml
total_timesteps: 500000      # Total training steps
max_episode_steps: 500       # Max steps per episode
save_freq: 10000             # Save model every N steps
```

#### Environment
```yaml
goal_frame: "gate"           # Target TF frame name
goal_tolerance: 1.0          # Success distance (meters)
visualize: true              # Enable RViz markers

# Objects to track (8 types)
object_frames: "gate_shark_link,gate_sawfish_link,red_pipe_link,..."
```

#### PPO Algorithm
```yaml
learning_rate: 0.0003        # Adam learning rate
n_steps: 2048                # Rollout buffer size
batch_size: 64               # Training batch size
gamma: 0.99                  # Discount factor
clip_range: 0.2              # PPO clip parameter
```

---

## Troubleshooting

### Problem: "Model file not found"
**Solution:**
```bash
# Check if model exists
ls -lh ./models/

# Train a model first
roslaunch auv_rl_navigation training.launch

# Or specify correct path
roslaunch auv_rl_navigation deploy_agent.launch \
    model_path:=$(rospack find auv_rl_navigation)/models/ppo_auv_best.zip
```

### Problem: "TF lookup failed"
**Solution:**
```bash
# Check TF tree
rosrun rqt_tf_tree rqt_tf_tree

# Verify frames exist
rosrun tf tf_echo taluy/base_link gate

# Update goal_frame in config
rosparam set /auv_rl_trainer/goal_frame "your_frame_name"
```

### Problem: "Low training performance"
**Solutions:**
1. **Increase training time:**
   ```yaml
   total_timesteps: 2000000  # Train longer
   ```

2. **Adjust learning rate:**
   ```yaml
   learning_rate: 0.0001  # Lower for stability
   ```

3. **Check reward values in TensorBoard:**
   - Should gradually increase over time
   - If stuck at negative values, reward function may need tuning

4. **Verify environment setup:**
   ```bash
   roslaunch auv_rl_navigation test_components.launch
   ```

### Problem: "Agent oscillates/unstable"
**Solutions:**
1. **Use deterministic policy:**
   ```bash
   roslaunch auv_rl_navigation deploy_agent.launch deterministic:=true
   ```

2. **Reduce control rate:**
   ```bash
   roslaunch auv_rl_navigation deploy_agent.launch rate:=5
   ```

3. **Check action scaling in `auv_nav_env.py`:**
   ```python
   cmd.linear.x = float(action[0]) * 0.5   # Reduce from 1.0 to 0.5
   ```

---

## Command Reference

### Launch Files

| Launch File | Purpose |
|-------------|---------|
| `test_components.launch` | Test all components before training |
| `training.launch` | Train RL agent |
| `deploy_agent.launch` | Run trained agent |

### Scripts

| Script | Purpose |
|--------|---------|
| `test_components.py` | Automated component testing |
| `train_agent.py` | Main training script |
| `rl_agent_node.py` | Deployment/inference node |
| `plot_training_progress.py` | Real-time training visualization |
| `live_action_monitor.py` | Real-time action monitoring |

### Topics (Deployment)

| Topic | Type | Description |
|-------|------|-------------|
| `/rl_agent_node/enable` | `std_msgs/Bool` | Enable/disable agent |
| `/rl_agent_node/reset` | `std_msgs/Bool` | Reset episode |
| `/rl_agent_node/reward` | `std_msgs/Float32` | Current reward |
| `/rl_agent_node/distance_to_goal` | `std_msgs/Float32` | Distance to goal |
| `/rl_agent_node/status` | `std_msgs/String` | Agent status |
| `/rl_agent_node/current_action` | `geometry_msgs/PoseStamped` | Current action |

---

## Typical Workflow

### 1. Development Phase
```bash
# Test components
roslaunch auv_rl_navigation test_components.launch

# Short test training (5 minutes)
roslaunch auv_rl_navigation training.launch total_timesteps:=50000

# Check if training works
tensorboard --logdir=./rl_logs/
```

### 2. Full Training
```bash
# Terminal 1: Training (8-12 hours)
roslaunch auv_rl_navigation training.launch

# Terminal 2: TensorBoard monitoring
tensorboard --logdir=./rl_logs/

# Terminal 3: Live plots
rosrun auv_rl_navigation plot_training_progress.py
```

### 3. Evaluation
```bash
# Load and evaluate trained model
roslaunch auv_rl_navigation deploy_agent.launch monitor:=true

# Watch live actions
rosrun auv_rl_navigation live_action_monitor.py
```

### 4. Deployment
```bash
# Run on real robot
roslaunch auv_rl_navigation deploy_agent.launch \
    model_path:=/path/to/best_model.zip \
    goal_frame:=target_buoy \
    start_enabled:=false  # Manual start

# Enable when ready
rostopic pub /rl_agent_node/enable std_msgs/Bool "data: true"
```

---

## Tips for Better Training

1. **Start with simple environments** - fewer obstacles
2. **Use curriculum learning** - gradually increase difficulty
3. **Monitor TensorBoard** - watch for learning progress
4. **Save checkpoints frequently** - `save_freq: 5000`
5. **Test on multiple goals** - vary `goal_frame` during training
6. **Adjust reward weights** - tune in `auv_nav_env.py`
7. **Use visualization** - helps debug issues
8. **Be patient** - good policies need 500K-1M steps

---

## Performance Expectations

### Training Time
- **100K steps:** ~2 hours → Basic exploration
- **500K steps:** ~10 hours → Decent navigation
- **1M steps:** ~20 hours → Good performance
- **2M+ steps:** ~40+ hours → Near-optimal

### Success Rates (after training)
- **Good model:** 70-90% success rate
- **Great model:** >90% success rate
- **Poor model:** <50% (needs more training)

### Convergence Signs
- Reward curve trending upward
- Episode length decreasing (finding direct paths)
- Success rate > 70%

---

## Support

For issues, check:
1. Test components first: `roslaunch auv_rl_navigation test_components.launch`
2. Review logs: `~/.ros/log/latest/`
3. Check TensorBoard for training metrics
4. Verify TF frames: `rosrun rqt_tf_tree rqt_tf_tree`
