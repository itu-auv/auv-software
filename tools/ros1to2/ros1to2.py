#!/usr/bin/env python3
"""Mechanical rospy -> rclpy rewrites. Does the boring 80%; prints what needs a human.

usage: ros1to2.py [--node EXPR] file.py [file.py ...]
  --node EXPR   expression that yields the rclpy node at the call sites (default: self)
"""
import re
import sys

LOG = {"info": "info", "warn": "warning", "err": "error", "debug": "debug", "fatal": "fatal"}


def split_args(s):
    """Split a call's argument text on top-level commas."""
    out, depth, cur, quote = [], 0, "", None
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            cur += c
            if c == "\\":
                cur += s[i + 1]
                i += 1
            elif s.startswith(quote, i):
                cur += quote[1:]
                i += len(quote) - 1
                quote = None
        elif c in "\"'":
            quote = s[i : i + 3] if s[i : i + 3] in ('"""', "'''") else c
            cur += quote
            i += len(quote) - 1
        elif c in "([{":
            depth += 1
            cur += c
        elif c in ")]}":
            depth -= 1
            cur += c
        elif c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur)
    merged = []
    for a in out:  # a lambda's own parameter commas are not argument separators
        if merged and merged[-1].lstrip().startswith("lambda") and ":" not in merged[-1]:
            merged[-1] += "," + a
        else:
            merged.append(a)
    return merged


def find_call(src, start):
    """src[start] is '('; return index of the matching ')'."""
    depth, i, quote = 0, start, None
    while i < len(src):
        c = src[i]
        if quote:
            if c == "\\":
                i += 1
            elif src.startswith(quote, i):
                i += len(quote) - 1
                quote = None
        elif c == "#":
            i = src.find("\n", i)
            if i < 0:
                break
            continue
        elif c in "\"'":
            quote = src[i : i + 3] if src[i : i + 3] in ('"""', "'''") else c
            i += len(quote) - 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced call at %d" % start)


def rewrite_calls(src, pattern, fn):
    """For each regex match ending right before '(', replace the whole call with fn(match, args, kwargs)."""
    pos, out = 0, []
    rx = re.compile(pattern)
    while True:
        m = rx.search(src, pos)
        if not m:
            break
        op = m.end() - 1
        line_start = src.rfind("\n", 0, m.start()) + 1
        if src[line_start : m.start()].lstrip().startswith("#"):
            out.append(src[pos : m.end()])
            pos = m.end()
            continue
        cl = find_call(src, op)
        raw = split_args(src[op + 1 : cl])
        args, kwargs = [], {}
        for a in raw:
            km = re.match(r"\s*([A-Za-z_]\w*)\s*=(?!=)(.*)", a, re.S)
            if km:
                kwargs[km.group(1)] = km.group(2).strip()
            else:
                args.append(a.strip())
        rep = fn(m, args, kwargs)
        if rep is None:
            out.append(src[pos : m.end()])
            pos = m.end()
            continue
        out.append(src[pos : m.start()] + rep)
        pos = cl + 1
    out.append(src[pos:])
    return "".join(out)


def convert(src, node):
    notes = []

    # --- service / action type names
    srv = set()
    for m in re.finditer(r"from\s+[\w.]+\.srv\s+import\s+(\([^)]*\)|[^\n]+)", src):
        for name in re.findall(r"[A-Za-z_]\w*", m.group(1)):
            if not name.endswith(("Request", "Response")) and name != "as":
                srv.add(name)

    def fix_srv_import(m):
        head, body = m.group(1), m.group(2)
        paren = body.startswith("(")
        names = [n for n in re.findall(r"[A-Za-z_]\w*", body) if not (n.endswith(("Request", "Response")) and n[: -7 if n.endswith("Request") else -8] in srv)]
        names = list(dict.fromkeys(names))
        if paren and len(names) > 3:
            return head + "(\n" + "".join("    %s,\n" % n for n in names) + ")"
        return head + ", ".join(names)

    src = re.sub(r"(from\s+[\w.]+\.srv\s+import\s+)(\([^)]*\)|[^\n]+)", fix_srv_import, src)
    for s in sorted(srv, key=len, reverse=True):
        src = re.sub(r"\b%sRequest\b" % s, s + ".Request", src)
        src = re.sub(r"\b%sResponse\b" % s, s + ".Response", src)

    # --- tf.transformations
    src = re.sub(r"import tf\.transformations as (\w+)", r"import tf_transformations as \1", src)
    src = re.sub(r"from tf\.transformations import", "from tf_transformations import", src)
    src = re.sub(r"^import tf\.transformations$", "import tf_transformations", src, flags=re.M)
    src = re.sub(r"^from tf import transformations as (\w+)$", r"import tf_transformations as \1", src, flags=re.M)
    src = re.sub(r"^from tf import transformations$", "import tf_transformations as transformations", src, flags=re.M)
    src = re.sub(r"\btf\.transformations\.", "tf_transformations.", src)
    src = re.sub(r"\btf_conversions\.transformations\.", "tf_transformations.", src)
    src = re.sub(r"^import tf_conversions$", "import tf_transformations", src, flags=re.M)

    # --- logging
    def log(m, args, kwargs):
        level, kind = LOG[m.group(1)], m.group(2)
        extra = ""
        if kind == "_throttle":
            extra = ", throttle_duration_sec=%s" % args.pop(0)
        elif kind == "_once":
            extra = ", once=True"
        if len(args) > 1:
            msg = "%s %% (%s)" % (args[0], ", ".join(args[1:]) + ("," if len(args) == 2 else ""))
        else:
            msg = args[0] if args else '""'
        return "%s.get_logger().%s(%s%s)" % (node, level, msg, extra)

    src = rewrite_calls(src, r"rospy\.log(info|warn|err|debug|fatal)(_throttle|_once)?\(", log)

    # --- params
    src = re.sub(r"rospy\.get_param\(", "get_param(%s, " % node, src)
    src = re.sub(r"rospy\.has_param\(", "has_param(%s, " % node, src)

    # --- pub / sub / srv
    def qos(kwargs, default="10"):
        depth = kwargs.get("queue_size", default)
        if depth in ("None",):
            depth = default
        if kwargs.get("latch") == "True":
            return "latched_qos(%s)" % depth if depth != "1" else "latched_qos()"
        return depth

    def pub(m, args, kwargs):
        extra = set(kwargs) - {"queue_size", "latch", "tcp_nodelay"}
        if extra or len(args) != 2:
            notes.append("Publisher with unusual args: %s %s" % (args, kwargs))
        return "%s.create_publisher(%s, %s, %s)" % (node, args[1], args[0], qos(kwargs))

    src = rewrite_calls(src, r"rospy\.Publisher\(", pub)

    def sub(m, args, kwargs):
        extra = set(kwargs) - {"queue_size", "tcp_nodelay", "buff_size", "callback"}
        cb = args[2] if len(args) > 2 else kwargs.get("callback")
        if extra or len(args) > 3 or cb is None:
            notes.append("Subscriber with unusual args: %s %s" % (args, kwargs))
        return "%s.create_subscription(%s, %s, %s, %s)" % (node, args[1], args[0], cb, qos(kwargs))

    src = rewrite_calls(src, r"rospy\.Subscriber\(", sub)
    src = rewrite_calls(src, r"rospy\.Service\(", lambda m, a, k: "%s.create_service(%s, %s, %s)" % (node, a[1], a[0], a[2]))
    src = rewrite_calls(src, r"rospy\.ServiceProxy\(", lambda m, a, k: "service_client(%s, %s, %s)" % (node, a[1], a[0]))
    src = rewrite_calls(
        src,
        r"rospy\.Timer\(",
        lambda m, a, k: "%s.create_timer(%s, %s)" % (node, re.sub(r"^rospy\.Duration\((.*)\)$", r"\1", a[0]), a[1]),
    )

    def mf_sub(m, args, kwargs):
        q = kwargs.get("queue_size")
        return "message_filters.Subscriber(%s, %s, %s%s)" % (node, args[1], args[0], ", qos_profile=%s" % q if q else "")

    src = rewrite_calls(src, r"message_filters\.Subscriber\(", mf_sub)

    # --- time
    src = re.sub(r"rospy\.Time\.now\(\)", "%s.get_clock().now()" % node, src)
    src = re.sub(r"rospy\.get_rostime\(\)", "%s.get_clock().now()" % node, src)
    src = re.sub(r"rospy\.get_time\(\)", "(%s.get_clock().now().nanoseconds / 1e9)" % node, src)
    src = re.sub(r"rospy\.Time\(0\)|rospy\.Time\(\)", "Time()", src)
    src = rewrite_calls(src, r"rospy\.Duration\(", lambda m, a, k: "Duration(seconds=%s)" % a[0] if len(a) == 1 and not k else None)
    src = rewrite_calls(src, r"rospy\.Duration\.from_sec\(", lambda m, a, k: "Duration(seconds=%s)" % a[0])
    src = re.sub(r"(stamp\s*=\s*[^\n#]*get_clock\(\)\.now\(\))(?!\.)", r"\1.to_msg()", src)
    src = re.sub(r"\.to_sec\(\)", ".nanoseconds / 1e9", src)
    src = rewrite_calls(src, r"rospy\.Rate\(", lambda m, a, k: "Rate(%s, %s)" % (node, a[0]))
    src = rewrite_calls(src, r"rospy\.sleep\(", lambda m, a, k: "%s.get_clock().sleep_for(%s)" % (node, a[0] if "Duration" in a[0] else "Duration(seconds=%s)" % a[0]))
    src = re.sub(r"not rospy\.is_shutdown\(\)", "rclpy.ok()", src)
    src = re.sub(r"rospy\.is_shutdown\(\)", "not rclpy.ok()", src)
    src = re.sub(r"rospy\.spin\(\)", "rclpy.spin(%s)" % node, src)
    src = re.sub(r"rospy\.ROSInterruptException", "ROSInterruptException", src)
    src = rewrite_calls(src, r"rospy\.signal_shutdown\(", lambda m, a, k: "rclpy.try_shutdown()")
    src = re.sub(r"smach_ros\.ServiceState\.__init__\(\s*self,", "smach_ros.ServiceState.__init__(\n            self,\n            %s," % node, src)
    src = re.sub(r"((?:self\.)?\w+)\.unregister\(\)", r"%s.destroy_subscription(\1)" % node, src)
    src = re.sub(r"rospy\.ServiceException", "Exception", src)
    src = re.sub(r"rospy\.(exceptions\.)?ROSException", "ROSException", src)
    src = re.sub(r"rospy\.wait_for_message\(", "wait_for_message(%s, " % node, src)
    src = re.sub(r"rospy\.on_shutdown\(", "on_shutdown(", src)

    # private names are "~/name" in ROS 2
    src = re.sub(r'((?:create_publisher|create_subscription|create_service|service_client)\([^)\n]*?)"~(\w)', r'\1"~/\2', src)

    # --- tf2
    src = re.sub(r"tf2_ros\.TransformListener\(([^(),]+)\)", r"tf2_ros.TransformListener(\1, %s)" % node, src)
    src = re.sub(r"tf2_ros\.(Static)?TransformBroadcaster\(\)", r"tf2_ros.\1TransformBroadcaster(%s)" % node, src)
    src = re.sub(r"\.get_num_connections\(\)", ".get_subscription_count()", src)

    if re.search(r"^import tf$", src, flags=re.M) and not re.search(r"\btf\.(?!transformations)", src):
        rep = "" if re.search(r"^import tf_transformations", src, flags=re.M) or "tf_transformations." not in src else "import tf_transformations"
        src = re.sub(r"^import tf\n", rep + "\n" if rep else "", src, count=1, flags=re.M)
    src = re.sub(r"\bdo_transform_pose\(", "do_transform_pose_stamped(", src)

    # --- message field renames
    src = re.sub(r"\b(\w*(?:calib|info|cam)\w*)\.(K|D|R|P)\b(?!\()", lambda m: "%s.%s" % (m.group(1), m.group(2).lower()), src)
    src = re.sub(r"\bbbox\.center\.(x|y)\b", r"bbox.center.position.\1", src)

    # --- imports
    imports = ["import rclpy"]
    if re.search(r"\(Node\)|\bNode\(", src):
        imports.append("from rclpy.node import Node")
    if re.search(r"\bDuration\(", src) and "from rclpy.duration import Duration" not in src:
        imports.append("from rclpy.duration import Duration")
    if re.search(r"\bTime\(", src) and "from rclpy.time import Time" not in src:
        imports.append("from rclpy.time import Time")
    if "ROSInterruptException" in src:
        imports.append("from rclpy.exceptions import ROSInterruptException")
    helpers = [h for h in ("get_param", "has_param", "latched_qos", "service_client", "get_node", "wait_for_message", "ROSException", "Rate", "on_shutdown") if re.search(r"\b%s\(" % h, src)]
    if helpers:
        imports.append("from auv_common_lib.ros import %s" % ", ".join(helpers))
    if re.search(r"^import rospy$", src, flags=re.M):
        src = re.sub(r"^import rospy$", "\n".join(imports), src, count=1, flags=re.M)
    elif "rospy" in src or helpers:
        notes.append("no plain 'import rospy' line; add imports by hand: %s" % imports)

    # --- what is left for a human
    for i, line in enumerate(src.split("\n"), 1):
        why = []
        if line.lstrip().startswith("#"):
            continue
        if "rospy" in line:
            why.append("rospy left")
        if re.search(r"create_service\(", line):
            why.append("service cb signature (req, resp)")
        if re.search(r"service_client\(", line):
            why.append("client: call sites")
        if ".nanoseconds / 1e9" in line and "get_clock" not in line:
            why.append("to_sec: msg stamp or rclpy Time?")
        if re.search(r"\bimport tf$|\btf\.Transform|actionlib|dynamic_reconfigure|rospkg|rosnode|roslaunch", line):
            why.append("ROS 1 lib")
        if re.search(r"\.results\[|\.results\b.*\.(id|score)\b|\.(id|score)\b", line) and "hypothesis" not in line:
            why.append("vision_msgs: results[i].hypothesis.class_id (str) / .score ?")
        if re.search(r"\.[KDRP]\[", line):
            why.append("CameraInfo field? (k d r p)")
        if "do_transform_pose_stamped(" in line:
            why.append("do_transform_pose: PoseStamped (ok) or Pose (use do_transform_pose)?")
        if re.search(r"create_timer\(", line):
            why.append("timer cb takes no event")
        if re.search(r"\b(Point|Vector3|Quaternion|ColorRGBA|Header|Pose|Twist|Transform|PoseStamped|PointStamped|Float32|Float64|Bool|String|Int32|UInt16)\((?!\))[^=)]*\)", line):
            why.append("positional msg ctor?")
        if why:
            notes.append("L%d: %s | %s" % (i, "; ".join(why), line.strip()[:110]))
    return src, notes


def main():
    argv = sys.argv[1:]
    node = "self"
    if argv and argv[0] == "--node":
        node = argv[1]
        argv = argv[2:]
    for path in argv:
        src = open(path).read()
        new, notes = convert(src, node)
        open(path, "w").write(new)
        print("== %s (%d notes)" % (path, len(notes)))
        for n in notes:
            print("  " + n)


if __name__ == "__main__":
    main()
