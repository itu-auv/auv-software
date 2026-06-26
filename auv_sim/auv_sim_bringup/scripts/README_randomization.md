# Randomized simulation (v1)

Runs the AUV sim for many episodes, each with a different **reproducible** random
configuration, to surface code bugs that the "always-nominal" sim hides.

## Idea

The sim hides a bug whenever it sits at a *special* value (zero, identity,
aligned, equal, default, in-range) where buggy code happens to look correct.
Randomization destroys those coincidences. v1 randomizes only things that
exercise **code paths**, not physics fidelity:

| Knob | Bug class it exposes |
|------|----------------------|
| Spawn x/y/z/roll/pitch/yaw (asymmetric) | frame/transform, "always level", "0 yaw", depth-range assumptions |
| Spawn placement constraint (≥4 m from every object, ≤8 m from one) | keeps the robot in the active lane, never on top of an object |
| z-origin depth offset | hard-coded depth thresholds (the 2 m teleop limit, pool_depth caps) — robot stays physically shallow but odom reports deep |
| IMU≠DVL≠pressure update rates | rate-plumbing bugs (e.g. the fixed spawn_robot.launch DVL/pressure→IMU wiring) |

`--no-randomize` reproduces today's exact behaviour.

## Usage

```bash
# 5 randomized episodes, gate+slalom mission, 3 min each, headless
rosrun auv_sim_bringup episode_runner.py \
    --episodes 5 --world pool \
    --mission "auv_bringup robosub.launch" \
    --episode-timeout 180

# Reproduce a single failing episode exactly (seed is logged per episode)
rosrun auv_sim_bringup episode_runner.py --episodes 1 --seed 1737045123

# Baseline: today's behaviour, no randomization
rosrun auv_sim_bringup episode_runner.py --episodes 5 --no-randomize
```

Each run writes to `~/auv_random_runs/<timestamp>/`:
- `summary.jsonl` — one line per episode (seed, spawn, status).
- `ep_XXXX_seed_N/params.json` — the full resolved config (replay with `--seed N`).
- `ep_XXXX_seed_N/gazebo.log`, `mission.log`.

> v1 logs episodes by timeout only. Automatic success/failure detection is v2.

**Reproducibility caveat:** the seed deterministically fixes the *configuration*
(spawn pose, sensor rates, depth offset) — verified by the unit tests. It does
**not** make the in-sim rollout bit-identical: Gazebo physics and the
sensor-noise plugins use their own RNG, which v1 does not seed. A replay
reproduces the same starting conditions, not necessarily the same trajectory.

## Config

- `config/randomization/randomization.yaml` — distributions / ranges per knob.
- `config/randomization/objects_<world>.yaml` — object positions + pool bounds +
  nominal spawn for the placement sampler. Add one file per world.

`scripts/randomization.py` is pure Python; run it directly to self-check the
sampler and the placement constraints without ROS:

```bash
python3 scripts/randomization.py pool
```
