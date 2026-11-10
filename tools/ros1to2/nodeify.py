#!/usr/bin/env python3
"""Second pass for node scripts already run through ros1to2.py:
 - class with rospy.init_node(...) in __init__ -> Node subclass
 - service callbacks get the extra `response` parameter
 - `if __name__ == "__main__":` block -> main() using ros.init()/ros.spin()
 - import fix-ups
usage: nodeify.py files...
"""
import re
import sys


def fix_imports(src):
    want = []
    helpers = [h for h in ("ServiceUnavailable", "get_param", "has_param", "latched_qos", "service_client", "wait_for_service", "get_node", "wait_for_message", "ROSException", "Rate", "on_shutdown", "init_node") if re.search(r"(?<![\w.])%s\b" % h, re.sub(r"from auv_common_lib\.ros import (\([^)]*\)|[^\n]+)\n", "", src))]
    src = re.sub(r"^from auv_common_lib\.ros import (\([^)]*\)|[^\n]+)\n", "", src, flags=re.M)
    if re.search(r"(?<![\w.])ros\.(init|spin|coerce|set_node)\(", src) and not re.search(r"^from auv_common_lib import ros$", src, flags=re.M):
        want.append("from auv_common_lib import ros")
    if helpers:
        want.append("from auv_common_lib.ros import %s" % ", ".join(helpers))
    if re.search(r"\(Node\)", src) and "from rclpy.node import Node" not in src:
        want.append("from rclpy.node import Node")
    for name, line in (("Duration", "from rclpy.duration import Duration"), ("Time", "from rclpy.time import Time")):
        if re.search(r"(?<![\w.])%s\(" % name, src) and line not in src:
            want.append(line)
    if "ROSInterruptException" not in re.sub(r"^from rclpy.exceptions import ROSInterruptException\n", "", src, flags=re.M):
        src = re.sub(r"^from rclpy.exceptions import ROSInterruptException\n", "", src, flags=re.M)
    if want:
        if re.search(r"^import rclpy$", src, flags=re.M):
            src = re.sub(r"^import rclpy$", "import rclpy\n" + "\n".join(want), src, count=1, flags=re.M)
        else:
            m = re.search(r"^(import|from) ", src, flags=re.M)
            src = src[: m.start()] + "\n".join(want) + "\n" + src[m.start() :]
    return src


def nodeify(src):
    notes = []
    # class -> Node
    for m in list(re.finditer(r'^(\s*)rospy\.init_node\(\s*("[^"]+"|\'[^\']+\')[^)]*\)\n', src, flags=re.M)):
        before = src[: m.start()]
        cm = list(re.finditer(r"^class (\w+)(\([^)]*\))?:", before, flags=re.M))
        if not cm or m.group(1) == "    " or m.group(1) == "":
            continue
        c = cm[-1]
        if c.group(2) not in (None, "()", "(object)"):
            notes.append("class %s has a base already; init_node left" % c.group(1))
            continue
        src = src[: c.start()] + "class %s(Node):" % c.group(1) + src[c.end() : m.start()] + "%ssuper().__init__(%s)\n" % (m.group(1), m.group(2)) + src[m.end() :]
        break
    # service callbacks
    for m in re.finditer(r"create_service\(\s*[\w.]+,\s*[^,]+,\s*(?:self\.)?(\w+)\s*[,)]", src):
        cb = m.group(1)
        d = re.search(r"def %s\(\s*(self,\s*)?(\w+)\s*(:\s*[\w.]+)?\s*\)" % cb, src)
        if d:
            src = src[: d.end() - 1] + ", response" + src[d.end() - 1 :]
        elif not re.search(r"def %s\([^)]*response" % cb, src):
            notes.append("service callback %s: fix signature by hand" % cb)
    # timer callbacks take no event argument
    for m in re.finditer(r"create_timer\(\s*[^,]+,\s*(?:self\.)?(\w+)\s*[,)]", src):
        d = re.search(r"def %s\(\s*self,\s*_?\w+(=None)?\s*\)" % m.group(1), src)
        if d:
            src = src[: d.start()] + "def %s(self)" % m.group(1) + src[d.end() :]
    # main block
    m = re.search(r'^if __name__ == "__main__":\n(.*)\Z', src, flags=re.M | re.S)
    if m:
        body = m.group(1)
        inst = re.search(r"(\w+)\s*=\s*(\w+)\(\)", body)
        call = re.search(r"\b(\w+)\.(spin|run|start|execute|main_loop|loop)\(\)", body)
        cls = inst.group(2) if inst else None
        if not cls:
            bare = re.search(r"^\s+(\w+)\(\)(?:\.(\w+)\(\))?\s*$", body, flags=re.M)
            if bare:
                cls = bare.group(1)
                method = bare.group(2)
                call = None
        if cls and re.search(r"^class %s\(Node\)" % cls, src, flags=re.M):
            if call and inst and call.group(1) == inst.group(1):
                method = call.group(2)
            elif call is not None:
                method = None
            loop = ", node.%s" % method if method else ""
            new = 'def main():\n    ros.init()\n    node = %s()\n    ros.spin(node%s)\n\n\nif __name__ == "__main__":\n    main()\n' % (cls, loop)
            src = src[: m.start()] + new
        else:
            notes.append("main block: convert by hand")
    # an existing main() that still starts the node the rospy way
    m = re.search(r"^def main\(\):\n((?:    [^\n]*\n|\n)+)", src, flags=re.M)
    if m and "ros.init()" not in m.group(1):
        body = m.group(1)
        classes = [c for c in re.findall(r"\b([A-Z]\w+)\(\)", body) if re.search(r"^class %s\(Node\)" % c, src, flags=re.M)]
        if classes:
            call = re.search(r"\b\w+\.(spin|run|start|execute|main_loop|loop)\(\)", re.sub(r"rclpy\.spin\([^)]*\)", "", body))
            loop = ", node.%s" % call.group(1) if call else ""
            new = "def main():\n    ros.init()\n    node = %s()\n    ros.spin(node%s)\n\n\n" % (classes[0], loop)
            src = src[: m.start()] + new + src[m.end() :].lstrip("\n")
            notes[:] = [n for n in notes if "main block" not in n]
    # a loop method that only spins is not a loop
    m = re.search(r"ros\.spin\(node, node\.(\w+)\)", src)
    if m:
        body = re.search(r"\n    def %s\(self\)( -> None)?:\n(        \"\"\"[^\n]*\"\"\"\n)?        rclpy\.spin\(self\)\n(\n\n?)?" % m.group(1), src)
        if body:
            src = src[: body.start()] + "\n" + src[body.end() :]
            src = src.replace(m.group(0), "ros.spin(node)")
            src = re.sub(r"\n{4,}def main", "\n\n\ndef main", src)
    return fix_imports(src), notes


if __name__ == "__main__":
    for path in sys.argv[1:]:
        new, notes = nodeify(open(path).read())
        open(path, "w").write(new)
        left = [l for l in new.split("\n") if "rospy" in l and not l.lstrip().startswith("#")]
        print("== %s" % path)
        for n in notes + ["rospy: " + l.strip()[:120] for l in left]:
            print("  " + n)
