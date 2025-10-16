# Gazebo Optimization for RL Training

## Running Gazebo Headless

### Quick Start

**Headless training (recommended):**
```bash
roslaunch auv_rl_navigation training.launch gazebo_gui:=false sim_speed:=2.0
```

**With GUI (for debugging):**
```bash
roslaunch auv_rl_navigation training.launch gazebo_gui:=true sim_speed:=1.0
```

## Speed Optimization

### 1. Simulation Speed Multiplier

The `real_time_factor` controls how fast the simulation runs:

```bash
# Real-time (1x speed)
roslaunch auv_rl_navigation training.launch sim_speed:=1.0

# 2x faster (recommended for RTX 3050)
roslaunch auv_rl_navigation training.launch sim_speed:=2.0

# 4x faster (if your hardware can handle it)
roslaunch auv_rl_navigation training.launch sim_speed:=4.0

# 10x faster (aggressive, may become unstable)
roslaunch auv_rl_navigation training.launch sim_speed:=10.0
```

**How it works:**
- `sim_speed:=2.0` means simulation runs 2x faster than real-time
- 500K timesteps at 2x speed = ~4-6 hours (instead of 8-12 hours)
- Higher speeds save training time but may cause physics instability

### 2. Performance vs Stability Trade-off

| Speed | Stability | Use Case |
|-------|-----------|----------|
| 1.0x  | Very Stable | Debugging, final evaluation |
| 2.0x  | Stable | **Recommended for training** |
| 4.0x  | Mostly Stable | Fast training, simple physics |
| 10.0x | May be Unstable | Very simple scenarios only |

**For RTX 3050: Use 2.0x-3.0x for best balance**

### 3. Headless Mode Benefits

Running without GUI saves significant resources:

**With GUI (gazebo_gui:=true):**
- GPU: ~500MB VRAM for rendering
- CPU: ~20-30% for GUI updates
- FPS: Limited to screen refresh

**Headless (gazebo_gui:=false):**
- GPU: ~50MB VRAM (10x less!)
- CPU: ~5-10% overhead
- FPS: Limited only by physics engine
- **Training speed: 2-3x faster!**

### 4. Recommended Settings for RTX 3050

**For your hardware, use these optimal settings:**

```bash
roslaunch auv_rl_navigation training.launch \
  gazebo_gui:=false \
  sim_speed:=2.5 \
  config:=rl_params_rtx3050.yaml \
  rviz:=false
```

**This configuration:**
- ✅ Runs headless (saves 500MB VRAM)
- ✅ 2.5x simulation speed (trains in ~3-5 hours for 500K steps)
- ✅ Optimized batch sizes for 4GB VRAM
- ✅ No RViz overhead

### 5. Monitor Performance

Check simulation performance in real-time:

```bash
# Monitor real-time factor (should match your target)
rostopic echo /clock

# Check Gazebo stats
gz stats

# Monitor CPU/GPU usage
htop                    # CPU
nvidia-smi -l 1        # GPU
```

**Good performance indicators:**
- Real-time factor ≈ target speed (2.0 if you set sim_speed:=2.0)
- CPU usage < 80%
- GPU VRAM < 3.5GB (leave headroom for neural network)

### 6. Expected Training Times (RTX 3050)

| Configuration | 500K Steps | 1M Steps |
|---------------|------------|----------|
| GUI + 1.0x speed | 12-14 hours | 24-28 hours |
| Headless + 1.0x | 8-10 hours | 16-20 hours |
| Headless + 2.0x | **4-5 hours** | **8-10 hours** ⭐ |
| Headless + 3.0x | 3-4 hours | 6-8 hours |

**Recommended: Headless + 2.0x speed for stability and speed**

## Summary

**Best practice for RTX 3050:**
1. ✅ Always use headless mode (`gazebo_gui:=false`)
2. ✅ Set simulation speed to 2.0-2.5x (`sim_speed:=2.5`)
3. ✅ Use optimized config (`config:=rl_params_rtx3050.yaml`)
4. ✅ Disable RViz during training (`rviz:=false`)
5. ✅ Monitor with TensorBoard instead of live visualization

**This setup will train 500K steps in ~4-5 hours instead of 12+ hours!**
