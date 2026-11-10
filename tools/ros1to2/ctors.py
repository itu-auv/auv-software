#!/usr/bin/env python3
"""ROS 2 message constructors take keywords only. Rewrite Point(a, b, c) / Quaternion(*q) etc.
usage: ctors.py files...   (also imported by ros1to2.py)"""
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ros1to2 import rewrite_calls  # noqa: E402

FIELDS = {
    "Point": "x y z", "Point32": "x y z", "Vector3": "x y z", "Quaternion": "x y z w",
    "Pose": "position orientation", "Transform": "translation rotation", "Twist": "linear angular",
    "Wrench": "force torque", "ColorRGBA": "r g b a", "PoseStamped": "header pose",
    "PointStamped": "header point", "Vector3Stamped": "header vector", "TransformStamped": "header child_frame_id transform",
    "Float32": "data", "Float64": "data", "Bool": "data", "String": "data", "Int32": "data", "Int8": "data",
    "UInt8": "data", "UInt16": "data", "Int16": "data", "UInt32": "data", "Int64": "data", "Empty": "",
}


def fix_ctors(src):
    notes = []

    def fix(m, args, kwargs):
        name = m.group(1)
        if not args:
            return None
        fields = FIELDS[name].split()
        if len(args) == 1 and args[0].startswith("*") and not args[0].startswith("**"):
            expr = args[0][1:].strip()
            if re.fullmatch(r"[\w.]+", expr):
                parts = ["%s=%s[%d]" % (f, expr, i) for i, f in enumerate(fields)]
            else:
                return "%s(**dict(zip(%r, %s)))" % (name, tuple(fields), expr)
        elif any(a.startswith("*") for a in args) or len(args) > len(fields):
            notes.append("ctor %s(%s): fix by hand" % (name, ", ".join(args)[:60]))
            return None
        else:
            parts = ["%s=%s" % (f, a) for f, a in zip(fields, args)]
        parts += ["%s=%s" % kv for kv in kwargs.items()]
        return "%s(%s)" % (name, ", ".join(parts))

    for _ in range(6):
        notes.clear()
        new = rewrite_calls(src, r"(?<![\w.])(%s)\(" % "|".join(FIELDS), fix)
        if new == src:
            break
        src = new
    return src, notes


if __name__ == "__main__":
    for path in sys.argv[1:]:
        new, notes = fix_ctors(open(path).read())
        if new != open(path).read():
            open(path, "w").write(new)
            print("fixed ctors in", path)
        for n in notes:
            print("  ", path, n)
