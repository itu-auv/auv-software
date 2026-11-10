#!/usr/bin/env python3
"""Wrap plain rosparam YAML files as ROS 2 parameter files (/**: ros__parameters:). usage: wrapyaml.py files..."""
import sys
for path in sys.argv[1:]:
    src = open(path).read()
    if "ros__parameters" in src:
        continue
    body = "".join(("    " + line if line.strip() else line) for line in src.splitlines(True))
    open(path, "w").write("/**:\n  ros__parameters:\n" + body)
    print("wrapped", path)
