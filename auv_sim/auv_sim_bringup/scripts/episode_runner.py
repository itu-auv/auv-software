#!/usr/bin/env python3
"""Run the AUV simulation for N randomized episodes, logging every seed.

Relaunch-per-episode model: each episode is a full roslaunch lifecycle (fresh
roscore + Gazebo), so there is zero state leakage between runs. For every
episode we:

  1. derive a per-episode seed from the base seed (reproducible),
  2. sample a resolved parameter set (randomization.py),
  3. launch start_gazebo.launch headless with those values,
  4. optionally launch a mission (e.g. auv_bringup robosub.launch),
  5. let it run until --episode-timeout (v1: pure timeout; success/fail = v2),
  6. tear everything down and log seed + resolved params + result.

A failing episode is replayed with:
    episode_runner.py --episodes 1 --seed <that_seed>

Reproducibility caveat: the seed deterministically fixes the *configuration*
(spawn pose, rates, depth offset). It does NOT make the in-sim rollout
bit-identical -- Gazebo physics and the sensor-noise plugins use their own RNG,
which we do not seed in v1. So a replay reproduces the same starting conditions,
not necessarily the exact same trajectory.

`--no-randomize` reproduces today's exact behaviour for every episode.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

import randomization as randmod

GAZEBO_LAUNCH = ("auv_sim_bringup", "start_gazebo.launch")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; per-episode seed = base + index. Default: time-based.")
    ap.add_argument("--world", default="pool")
    ap.add_argument("--namespace", default="taluy")
    ap.add_argument("--randomize", dest="randomize", action="store_true", default=True)
    ap.add_argument("--no-randomize", dest="randomize", action="store_false",
                    help="reproduce today's exact behaviour every episode")
    ap.add_argument("--mission", default=None,
                    help='roslaunch spec to run after Gazebo is up, e.g. '
                         '"auv_bringup robosub.launch" (args appended verbatim)')
    ap.add_argument("--episode-timeout", type=float, default=120.0,
                    help="seconds to let each episode run before teardown")
    ap.add_argument("--startup-timeout", type=float, default=90.0,
                    help="seconds to wait for Gazebo/bringup to come up")
    ap.add_argument("--gui", action="store_true", help="show the Gazebo/RViz GUI")
    ap.add_argument("--output-dir", default=None,
                    help="default: ~/auv_random_runs/<timestamp>")
    return ap.parse_args()


def launch_args_from_params(p, world, namespace, gui):
    """Map a resolved parameter dict to start_gazebo.launch key:=value args."""
    return [
        "world:={}".format(world),
        "namespace:={}".format(namespace),
        "use_gui:={}".format("true" if gui else "false"),
        "use_taluy_gui:={}".format("true" if gui else "false"),
        "x:={:.4f}".format(p["x"]),
        "y:={:.4f}".format(p["y"]),
        "z:={:.4f}".format(p["z"]),
        "roll:={:.4f}".format(p["roll"]),
        "pitch:={:.4f}".format(p["pitch"]),
        "yaw:={:.4f}".format(p["yaw"]),
        "imu_update_rate:={}".format(p["imu_rate"]),
        "dvl_update_rate:={}".format(p["dvl_rate"]),
        "pressure_update_rate:={}".format(p["pressure_rate"]),
        "depth_origin_offset:={:.4f}".format(p["depth_origin_offset"]),
    ]


def wait_for_ready(namespace, timeout):
    """Block until the odometry topic shows up (bringup is alive) or timeout."""
    topic = "/{}/odometry".format(namespace)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(["rostopic", "list"],
                                          stderr=subprocess.DEVNULL).decode()
            if topic in out:
                return True
        except subprocess.CalledProcessError:
            pass  # roscore not up yet
        time.sleep(2.0)
    return False


def popen_group(cmd, logfile):
    """Start a process in its own process group so we can kill the whole tree."""
    return subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def kill_group(proc):
    """Kill the whole process group. SIGINT first (let roslaunch stop its
    nodes), then ALWAYS SIGKILL the group -- otherwise nodes whose roslaunch
    exits before they finish dying get orphaned (observed with the mission's
    state machine)."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return  # already gone
    try:
        os.killpg(pgid, signal.SIGINT)
    except OSError:
        pass
    try:
        proc.wait(timeout=12)
    except Exception:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)  # nuke any straggler still in the group
    except OSError:
        pass


def hard_cleanup():
    """Belt-and-suspenders: make sure nothing from the episode survives.

    roslaunch does not reliably keep its nodes in our process group, so killpg
    alone leaves orphaned nodes (e.g. the mission state machine). Every node
    roslaunch spawns carries a `__name:=` arg, which this script's own process
    does NOT -- so pkill on that pattern reaps all ROS nodes mission-agnostically
    without killing the runner.

    NOTE: this is container-wide (pkill by pattern). Run the batch on a
    dedicated sim machine/container -- it also kills unrelated ROS processes."""
    patterns = ["__name:=", "gzserver", "gzclient", "roslaunch", "rosmaster",
                "roscore"]
    for pat in patterns:
        subprocess.call(["pkill", "-9", "-f", pat],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)


def run_episode(idx, seed, params, args, ep_dir):
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "params.json"), "w") as f:
        json.dump({"index": idx, "seed": seed, "params": params}, f, indent=2)

    gz_log = open(os.path.join(ep_dir, "gazebo.log"), "w")
    mission_proc = None
    result = {"index": idx, "seed": seed, "status": "unknown",
              "mode": params["meta"]["mode"],
              "spawn": [round(params["x"], 2), round(params["y"], 2), round(params["z"], 2)],
              "yaw": round(params["yaw"], 3),
              "depth_origin_offset": round(params["depth_origin_offset"], 2),
              "rates": [params["imu_rate"], params["dvl_rate"], params["pressure_rate"]]}

    gz_cmd = ["roslaunch", GAZEBO_LAUNCH[0], GAZEBO_LAUNCH[1]] + \
        launch_args_from_params(params, args.world, args.namespace, args.gui)
    gz_proc = popen_group(gz_cmd, gz_log)

    try:
        if not wait_for_ready(args.namespace, args.startup_timeout):
            result["status"] = "startup_timeout"
            return result

        if args.mission:
            mlog = open(os.path.join(ep_dir, "mission.log"), "w")
            mission_proc = popen_group(["roslaunch"] + args.mission.split(), mlog)

        # v1: just let it run. v2 will watch for a terminal success/fail signal.
        time.sleep(args.episode_timeout)
        result["status"] = "completed"
    finally:
        kill_group(mission_proc)
        kill_group(gz_proc)
        gz_log.close()
        hard_cleanup()
    return result


def main():
    args = parse_args()
    base_seed = args.seed if args.seed is not None else int(time.time())
    out_dir = args.output_dir or os.path.join(
        os.path.expanduser("~"), "auv_random_runs",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    try:
        rand_cfg, world_cfg = randmod.load_config(args.world)
    except FileNotFoundError as e:
        sys.exit("[runner] missing randomization config for world '{}': {}".format(
            args.world, e))
    randomizer = randmod.Randomizer(rand_cfg, world_cfg)

    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"base_seed": base_seed, "episodes": args.episodes,
                   "randomize": args.randomize, "world": args.world,
                   "namespace": args.namespace, "mission": args.mission,
                   "argv": sys.argv[1:]}, f, indent=2)

    summary_path = os.path.join(out_dir, "summary.jsonl")
    print("[runner] output: {}".format(out_dir))
    print("[runner] base_seed={} episodes={} randomize={} mission={}".format(
        base_seed, args.episodes, args.randomize, args.mission))

    for idx in range(args.episodes):
        seed = base_seed + idx
        params = randomizer.sample(seed) if args.randomize else randomizer.nominal()
        ep_dir = os.path.join(out_dir, "ep_{:04d}_seed_{}".format(idx, seed))
        print("\n[runner] === episode {}/{}  seed={} ===".format(
            idx + 1, args.episodes, seed))
        try:
            result = run_episode(idx, seed, params, args, ep_dir)
        except KeyboardInterrupt:
            print("\n[runner] interrupted; cleaning up.")
            hard_cleanup()
            break
        except Exception as e:  # keep the batch alive on a single bad episode
            result = {"index": idx, "seed": seed, "status": "error", "error": str(e)}
            hard_cleanup()
        with open(summary_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        print("[runner] episode {} -> {}".format(idx, result["status"]))

    print("\n[runner] done. summary: {}".format(summary_path))


if __name__ == "__main__":
    sys.exit(main())
