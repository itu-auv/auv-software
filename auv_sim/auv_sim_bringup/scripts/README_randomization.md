# Randomized simulation

Runs the AUV sim for many **seed-reproducible** randomized episodes to surface
code bugs that the always-nominal sim hides, recording **one small rosbag** for
the whole run.

## Idea

The sim hides a bug whenever it sits at a *special* value (zero, identity,
aligned, equal, default, in-range) where buggy code happens to look correct.
Randomization destroys those coincidences. We randomize only things that
exercise **code paths**, not physics fidelity:

| Category | Knob | Bug class it exposes |
|----------|------|----------------------|
| `spawn`  | x/y/z (placement-constrained) | frame/transform, depth-range, "always here" assumptions |
| `yaw`    | heading | "0 yaw" / frame-alignment bugs (e.g. missing transforms) |
| `depth`  | z-origin offset | hard-coded depth thresholds (the 2 m teleop limit) — robot stays physically shallow, odom reports deep |
| `rates`  | IMU≠DVL≠pressure update rates | rate-plumbing bugs |

`spawn` keeps the robot ≥4 m from every object and ≤8 m from at least one
(stays in the active lane). The depth offset is applied **sim-side** (in
`auv_sim_bridge`); the autonomy is never edited.

## Setup

Needs the normal sim stack (ROS noetic, Gazebo, uuv_simulator) plus
`topic_tools` + `image_transport` (declared in `package.xml`; in ros-desktop-full
already). For **camera recording on a headless machine** also install xvfb:

```bash
sudo apt install xvfb mesa-utils libgl1-mesa-dri   # headless camera rendering
```

If xvfb is absent the runner prints a note and continues without it: with a real
display the cameras still render; truly headless they are silent but the sim and
all non-camera logging still work. (The docker image already has xvfb.)

## Run it

Easiest — the terminal UI (pick tasks/order, episodes, seed, toggles, then run):

```bash
rosrun auv_sim_bringup randomization_tui.py
```

Or the runner directly:

```bash
rosrun auv_sim_bringup episode_runner.py \
    --episodes 10 --randomize spawn,yaw,depth,rates \
    --mission "auv_smach start.launch test_mode:=true test_states:=init,gate,slalom" \
    --episode-timeout 180

# reproduce one episode (its seed is logged); randomize nothing == today's sim
rosrun auv_sim_bringup episode_runner.py --episodes 1 --seed 1737045123
rosrun auv_sim_bringup episode_runner.py --episodes 5 --randomize none
rosrun auv_sim_bringup episode_runner.py --episodes 5 --no-randomize  # same thing
```

`--randomize` takes `all`, `none`, or a comma list of `spawn,yaw,depth,rates`;
`--no-randomize` is kept as an alias for `--randomize none`.

## How it works

- **Persistent roscore**: roscore and one `rosbag record` live for the whole
  run; only Gazebo + the per-episode nodes are relaunched each episode. This is
  deliberate — if smach `INITIALIZE` truly resets state, episode N+1 must be
  independent of N. **Cross-episode leakage on the shared roscore is itself a
  bug** (an autonomy init/reset bug). So the mission must start with `init` (the
  TUI forces this).
- **One bag** (`<run>/run.bag`, lz4). Started before `use_sim_time`, so it is
  wall-clock monotonic across episodes even though each episode's sim `/clock`
  restarts. Segment it by the latched `/randomization/episode` marker (index +
  seed). The bag drops raw images / pointclouds / per-link ground truth, and
  keeps the pool/bottom/torpedo cameras compressed + throttled (~5 Hz).
- **Cameras need rendering**: headless Gazebo is wrapped in `xvfb-run` (needs
  `xvfb`+`mesa` — already in `docker/Dockerfile.auv`). `--no-xvfb` disables it.

Each run writes `~/auv_random_runs/<timestamp>/`:
`run.bag`, `summary.jsonl` (one line/episode), `run_metadata.json`,
`ep_XXXX_seed_N/{params.json,gazebo.log,mission.log}`.

`summary.jsonl` also includes a `smach` object when a mission is launched:
whether the mission was still running at timeout or exited early, the return
code, the last transition seen in `mission.log`, and any failed state/error
line. The episode status itself is still timeout-based in v1.

**Reproducibility caveat:** the seed fixes the *configuration* (verified by the
unit tests), not the in-sim rollout — Gazebo physics and the sensor-noise
plugins use their own unseeded RNG.

## Config

- `config/randomization/randomization.yaml` — distributions per category, the
  `bag` section (exclude regex, cameras, rate). `roll`/`pitch` are disabled (the
  AUV self-rights, so an initial tilt is ineffective).
- `config/randomization/objects_<world>.yaml` — object positions + bounds +
  nominal spawn for the placement sampler. One file per world.

`scripts/randomization.py` is pure Python; `python3 randomization.py pool`
self-checks the sampler. `pytest test_randomization.py` runs the unit tests.
