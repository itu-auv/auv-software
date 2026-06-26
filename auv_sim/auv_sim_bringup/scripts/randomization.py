#!/usr/bin/env python3
"""Deterministic domain-randomization sampler for the AUV simulation.

Pure Python (no ROS) so it can be unit-tested on its own. Given a seed it
returns a fully resolved set of episode parameters; the same seed always yields
the same parameters, which is what makes a failing random episode reproducible.

See config/randomization/randomization.yaml for the meaning of every knob.
"""

import math
import os
import random

import yaml

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "config", "randomization")

# Randomization categories the TUI/runner can toggle independently.
#   spawn : x, y, z (+ placement constraint)   yaw : heading
#   depth : z-origin depth offset              rates : IMU/DVL/pressure rates
CATEGORIES = ("spawn", "yaw", "depth", "rates")


def load_config(world="pool", cfg_dir=_CFG_DIR):
    """Load the (shared) randomization spec and the per-world object data."""
    with open(os.path.join(cfg_dir, "randomization.yaml")) as f:
        rand_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "objects_{}.yaml".format(world))) as f:
        world_cfg = yaml.safe_load(f)
    return rand_cfg, world_cfg


class Randomizer:
    """Maps a seed -> a resolved dict of episode parameters."""

    def __init__(self, rand_cfg, world_cfg):
        self.cfg = rand_cfg
        self.world = world_cfg

    # -- public API --------------------------------------------------------

    def nominal(self):
        """Today's exact behaviour: zero offsets, hard-coded spawn pose."""
        n = self.world["nominal"]
        pool_depth = self.cfg["pool_depth_nominal"]
        return _resolved(
            x=n["x"], y=n["y"], z=n["z"],
            roll=0.0, pitch=0.0, yaw=n["yaw"],
            depth_origin_offset=0.0, pool_depth=pool_depth,
            imu_rate=20, dvl_rate=20, pressure_rate=20,
            meta={"mode": "nominal", "world": self.world["world"]},
        )

    def sample(self, seed, enabled=None):
        """Draw one reproducible episode configuration for the given seed.

        `enabled` is the set of randomization categories to apply (subset of
        CATEGORIES); disabled categories fall back to their nominal value. To
        keep a seed reproducible regardless of which toggles are on, every value
        is ALWAYS drawn (in a fixed order) and only then selected -- so toggling
        a category never shifts the RNG stream for the others."""
        if enabled is None:
            enabled = set(CATEGORIES)
        rng = random.Random(seed)
        s = self.cfg["spawn"]
        n = self.world["nominal"]

        # always draw, fixed order
        rx, ry, attempts, placed = self._sample_xy(rng)
        rz = rng.uniform(*self.world["spawn_z"])
        rroll = rng.uniform(s["roll"]["min"], s["roll"]["max"])
        rpitch = rng.uniform(s["pitch"]["min"], s["pitch"]["max"])
        ryaw = rng.uniform(s["yaw"]["min"], s["yaw"]["max"])
        d = self.cfg["depth_origin_offset"]
        rdepth = rng.uniform(d["min"], d["max"]) if rng.random() < d["probability"] else 0.0
        ur = self.cfg["update_rates"]
        rimu = rng.choice(ur["imu"]["values"])
        rdvl = rng.randint(ur["dvl"]["min"], ur["dvl"]["max"])
        rpress = rng.randint(ur["pressure"]["min"], ur["pressure"]["max"])

        # select per enabled category
        spawn_on = "spawn" in enabled
        x = rx if spawn_on else n["x"]
        y = ry if spawn_on else n["y"]
        z = rz if spawn_on else n["z"]
        roll = rroll if spawn_on else 0.0
        pitch = rpitch if spawn_on else 0.0
        yaw = ryaw if "yaw" in enabled else n["yaw"]
        depth_off = rdepth if "depth" in enabled else 0.0
        imu_rate = rimu if "rates" in enabled else 20
        dvl_rate = rdvl if "rates" in enabled else 20
        pressure_rate = rpress if "rates" in enabled else 20

        return _resolved(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            depth_origin_offset=depth_off,
            pool_depth=self.cfg["pool_depth_nominal"] + depth_off,
            imu_rate=imu_rate, dvl_rate=dvl_rate, pressure_rate=pressure_rate,
            meta={
                "mode": "random" if enabled else "nominal",
                "world": self.world["world"], "seed": seed,
                "enabled": sorted(enabled),
                "xy_attempts": attempts, "xy_placed": placed if spawn_on else None,
            },
        )

    # -- internals ---------------------------------------------------------

    def _sample_xy(self, rng):
        """Rejection-sample (x, y): >= exclusion_radius from EVERY object and
        <= max_nearest from at least one. Falls back to nominal xy on failure."""
        b = self.world["bounds"]
        objs = self.world["objects"]
        p = self.cfg["placement"]
        excl, maxn, budget = p["exclusion_radius"], p["max_nearest"], p["max_attempts"]

        for attempt in range(1, budget + 1):
            x = rng.uniform(*b["x"])
            y = rng.uniform(*b["y"])
            nearest = min(math.hypot(x - o["x"], y - o["y"]) for o in objs)
            if excl <= nearest <= maxn:
                return x, y, attempt, True

        n = self.world["nominal"]
        return n["x"], n["y"], budget, False


def _resolved(**kw):
    return kw


if __name__ == "__main__":
    # Tiny self-check: prints a few sampled configs and verifies constraints.
    import json
    import sys

    world = sys.argv[1] if len(sys.argv) > 1 else "pool"
    rand_cfg, world_cfg = load_config(world)
    r = Randomizer(rand_cfg, world_cfg)
    excl = rand_cfg["placement"]["exclusion_radius"]
    maxn = rand_cfg["placement"]["max_nearest"]

    print("NOMINAL:", json.dumps(r.nominal()))
    for seed in range(5):
        cfg = r.sample(seed)
        nearest = min(math.hypot(cfg["x"] - o["x"], cfg["y"] - o["y"])
                      for o in world_cfg["objects"])
        ok = excl - 1e-9 <= nearest <= maxn + 1e-9 or not cfg["meta"]["xy_placed"]
        print("seed={} nearest={:.2f} ok={}  {}".format(
            seed, nearest, ok, json.dumps(cfg)))
