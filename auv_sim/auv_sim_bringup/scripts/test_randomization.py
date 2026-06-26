#!/usr/bin/env python3
"""Unit tests for the domain-randomization sampler (pure Python, no ROS).

Run:  pytest test_randomization.py            (or)   python3 -m pytest -q
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import randomization as R  # noqa: E402

WORLD = "pool"


@pytest.fixture(scope="module")
def randr():
    rand_cfg, world_cfg = R.load_config(WORLD)
    return R.Randomizer(rand_cfg, world_cfg), rand_cfg, world_cfg


# -- reproducibility -------------------------------------------------------

def test_same_seed_is_deterministic(randr):
    r, _, _ = randr
    assert r.sample(12345) == r.sample(12345)


def test_different_seeds_differ(randr):
    r, _, _ = randr
    xs = {round(r.sample(s)["x"], 6) for s in range(20)}
    assert len(xs) > 15  # overwhelmingly distinct


# -- nominal == today's behaviour -----------------------------------------

def test_nominal_matches_hardcoded(randr):
    r, _, world = randr
    n = r.nominal()
    assert (n["x"], n["y"], n["z"], n["yaw"]) == (
        world["nominal"]["x"], world["nominal"]["y"],
        world["nominal"]["z"], world["nominal"]["yaw"])
    assert n["roll"] == n["pitch"] == 0.0
    assert n["depth_origin_offset"] == 0.0
    assert n["pool_depth"] == 2.134
    assert (n["imu_rate"], n["dvl_rate"], n["pressure_rate"]) == (20, 20, 20)


# -- placement constraint (the 4 m / 8 m circles) -------------------------

def test_placement_nearest_in_band(randr):
    r, cfg, world = randr
    excl = cfg["placement"]["exclusion_radius"]
    maxn = cfg["placement"]["max_nearest"]
    objs = world["objects"]
    placed = 0
    for seed in range(500):
        c = r.sample(seed)
        if not c["meta"]["xy_placed"]:
            continue
        placed += 1
        nearest = min(math.hypot(c["x"] - o["x"], c["y"] - o["y"]) for o in objs)
        assert excl - 1e-9 <= nearest <= maxn + 1e-9, \
            "seed {} nearest {:.3f} outside [{}, {}]".format(seed, nearest, excl, maxn)
    assert placed >= 490, "too many fallbacks: only {}/500 placed".format(placed)


def test_never_inside_any_exclusion_circle(randr):
    r, cfg, world = randr
    excl = cfg["placement"]["exclusion_radius"]
    for seed in range(500):
        c = r.sample(seed)
        if not c["meta"]["xy_placed"]:
            continue
        for o in world["objects"]:
            assert math.hypot(c["x"] - o["x"], c["y"] - o["y"]) >= excl - 1e-9


def test_spawn_within_bounds(randr):
    r, _, world = randr
    bx, by = world["bounds"]["x"], world["bounds"]["y"]
    for seed in range(200):
        c = r.sample(seed)
        if not c["meta"]["xy_placed"]:
            continue
        assert bx[0] <= c["x"] <= bx[1]
        assert by[0] <= c["y"] <= by[1]


# -- value ranges ----------------------------------------------------------

def test_attitude_and_depth_ranges(randr):
    r, _, world = randr
    zlo, zhi = world["spawn_z"]
    for seed in range(200):
        c = r.sample(seed)
        # roll/pitch disabled (self-righting AUV); yaw is full-circle
        assert c["roll"] == 0.0
        assert c["pitch"] == 0.0
        assert -math.pi <= c["yaw"] <= math.pi
        assert zlo <= c["z"] <= zhi


def test_depth_origin_offset_consistency(randr):
    r, cfg, _ = randr
    d = cfg["depth_origin_offset"]
    saw_zero = saw_big = False
    for seed in range(200):
        c = r.sample(seed)
        off = c["depth_origin_offset"]
        if off == 0.0:
            saw_zero = True
        else:
            saw_big = True
            assert d["min"] <= off <= d["max"]
        # pool_depth must always widen by exactly the offset
        assert abs(c["pool_depth"] - (cfg["pool_depth_nominal"] + off)) < 1e-9
    assert saw_zero and saw_big, "offset should sometimes fire and sometimes not"


def test_update_rates(randr):
    r, _, _ = randr
    saw_diff = False
    for seed in range(200):
        c = r.sample(seed)
        assert c["imu_rate"] == 20
        assert 5 <= c["dvl_rate"] <= 20
        assert 5 <= c["pressure_rate"] <= 20
        if len({c["imu_rate"], c["dvl_rate"], c["pressure_rate"]}) == 3:
            saw_diff = True
    assert saw_diff, "rates should differ across seeds (exposes rate-plumbing bug)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
