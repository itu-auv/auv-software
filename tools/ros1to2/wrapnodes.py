#!/usr/bin/env python3
"""A rosparam YAML whose top-level keys are node names -> ROS 2 parameter file (/**/<node>: ros__parameters:).
usage: wrapnodes.py files..."""
import re, sys

for path in sys.argv[1:]:
    out = []
    for line in open(path).read().splitlines(True):
        m = re.match(r"^([A-Za-z_]\w*):\s*(#.*)?$", line.rstrip("\n"))
        if m:
            out.append(
                "/**/%s:%s\n  ros__parameters:\n"
                % (m.group(1), " " + m.group(2) if m.group(2) else "")
            )
        elif line.strip() and not line.startswith("#"):
            out.append("  " + line)
        else:
            out.append(line)
    open(path, "w").write("".join(out))
    print("node-wrapped", path)
