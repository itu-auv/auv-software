#!/usr/bin/env python3
"""Find names in Node subclasses that collide with rclpy.node.Node members. Run inside the container.
usage: nodelint.py paths..."""
import ast, os, sys
from rclpy.node import Node
RESERVED = set(dir(Node))
def check(path):
    try: tree = ast.parse(open(path).read())
    except SyntaxError as e: print("SYNTAX", path, e); return
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not any(getattr(b, "id", getattr(b, "attr", "")) == "Node" for b in cls.bases): continue
        for n in ast.walk(cls):
            if isinstance(n, ast.FunctionDef) and n.name in RESERVED and not n.name.startswith("__") and n in cls.body:
                print(f"{path}:{n.lineno}: method {cls.name}.{n.name} overrides Node.{n.name}")
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for a in ast.walk(t):
                        if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "self" and a.attr in RESERVED:
                            print(f"{path}:{n.lineno}: attribute self.{a.attr} collides with Node.{a.attr}")
for root in sys.argv[1:]:
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"): check(os.path.join(d, f))
    if root.endswith(".py"): check(root)
