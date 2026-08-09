#!/usr/bin/env python3
"""Shared loader for the per-object ViTPose YAML configs.

One YAML per object under auv_vision/config/vitpose/<object>.yaml with three
sections — `model:` (vitpose_inference), `detection:` (vitpose_detection_node),
`process:` (vitpose_process_node). Schema: auv_vision/VITPOSE_PLAN.md §6.
"""

import glob
import os

import rospkg
import yaml

_rospack = rospkg.RosPack()


def config_dir() -> str:
    return os.path.join(_rospack.get_path("auv_vision"), "config", "vitpose")


def resolve_config_path(name_or_path: str) -> str:
    """Accept an object name ("gate") or an absolute/relative YAML path."""
    if name_or_path.endswith((".yaml", ".yml")) or os.path.sep in name_or_path:
        path = os.path.abspath(os.path.expanduser(name_or_path))
    else:
        path = os.path.join(config_dir(), f"{name_or_path}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"vitpose config not found: {path}")
    return path


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Bare filenames resolve against auv_detection/models/ (house style)."""
    if os.path.isabs(checkpoint):
        return checkpoint
    return os.path.join(_rospack.get_path("auv_detection"), "models", checkpoint)


def load_object_config(name_or_path: str) -> dict:
    path = resolve_config_path(name_or_path)
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    for section in ("object", "model", "detection", "process"):
        if section not in config:
            raise ValueError(f"{path}: missing required section '{section}'")
    config["_path"] = path
    return config


def all_object_configs() -> dict:
    """{object name: config} for every YAML in the config dir."""
    configs = {}
    for path in sorted(glob.glob(os.path.join(config_dir(), "*.yaml"))):
        config = load_object_config(path)
        name = config["object"]
        if name in configs:
            raise ValueError(
                f"duplicate vitpose object '{name}' "
                f"({configs[name]['_path']} vs {path})"
            )
        configs[name] = config
    return configs


def model_kwargs(config: dict) -> dict:
    """Translate a config's `model:` section into VitposeModel kwargs."""
    model = config["model"]
    kwargs = dict(
        device=model.get("device", "cuda"),
        input_size=model.get("input_size"),
        decode=model.get("decode"),
        flip_tta=bool(model.get("flip_tta", False)),
        flip_pairs=model.get("flip_pairs"),
        mask_threshold=model.get("mask_threshold"),
    )
    return kwargs
