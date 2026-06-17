#!/usr/bin/env python3
"""Shared YAML <-> SegParams IO for the offline tuning tools.

The on-disk format is a flat ``{param: value}`` dict using the cfg/dynparam
schema (combine_mode as an int), so a file written here can be pushed directly
to the live node with::

    rosrun dynamic_reconfigure dynparam load /taluy/opencv_seg_publisher best_params.yaml
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipe_segmentation import params_from_dict, params_to_dict  # noqa: E402


def load_params(path):
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    return params_from_dict(d)


def dump_params(params, path, report=None):
    d = params_to_dict(params)
    with open(path, "w") as f:
        if report:
            f.write("# %s\n" % report)
        yaml.safe_dump(d, f, default_flow_style=False, sort_keys=True)
    return d
