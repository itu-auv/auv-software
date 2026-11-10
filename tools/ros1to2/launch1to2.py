#!/usr/bin/env python3
"""roslaunch XML -> ROS 2 launch XML, in place (file names stay *.launch; ros2 launch reads them as XML).
Prints what needs a human. usage: launch1to2.py files..."""
import glob
import os
import re
import sys

import yaml

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PKG_DIR = {}
for px in glob.glob(os.path.join(REPO, "**", "package.xml"), recursive=True):
    m = re.search(r"<name>([^<]+)</name>", open(px).read())
    if m:
        PKG_DIR[m.group(1)] = os.path.dirname(px)


def convert(src):
    notes = []
    src = re.sub(r"\$\(arg ([^)]+)\)", r"$(var \1)", src)
    src = re.sub(r"\$\(find ([^)]+)\)", r"$(find-pkg-share \1)", src)
    src = re.sub(r"\$\(optenv ", "$(env ", src)

    # eval: arg('x') -> '$(var x)'
    def ev(m):
        expr = re.sub(r"arg\(\s*'(\w+)'\s*\)", r"'$(var \1)'", m.group(1))
        return "$(eval &quot;%s&quot;)" % expr.replace('"', "'")

    src = re.sub(r"\$\(eval ((?:[^()]|\([^()]*\))*)\)", ev, src)

    # rosparam
    def rosparam_load(m):
        f = re.search(r'file="([^"]+)"', m.group(0)).group(1)
        if re.search(r'\bns="[^"]+"', m.group(0)):
            notes.append("rosparam load with ns: %s" % m.group(0)[:100])
        return '<param from="%s" />' % f

    src = re.sub(r"<rosparam\b[^>]*command=\"load\"[^>]*/>", rosparam_load, src)
    src = re.sub(r"<rosparam\b[^>]*file=\"[^\"]+\"[^>]*/>", rosparam_load, src)

    def rosparam_inline(m):
        name, body = m.group(1), m.group(2)
        try:
            value = yaml.safe_load(body)
        except Exception:
            value = None
        if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
            return '<param name="%s" value="[%s]" />' % (name, ", ".join(str(v) for v in value))
        if isinstance(value, (str, int, float, bool)):
            return '<param name="%s" value="%s" />' % (name, value)
        notes.append("inline rosparam %s: convert by hand" % name)
        return m.group(0)

    src = re.sub(r'<rosparam\s+param="([^"]+)"\s*>(.*?)</rosparam>', rosparam_inline, src, flags=re.S)

    # groups with ns
    def group(m):
        indent, attrs = m.group(1), m.group(2)
        ns = re.search(r'\bns="([^"]+)"', attrs).group(1)
        rest = re.sub(r'\s*\bns="[^"]+"', "", attrs)
        return '%s<group%s>\n%s  <push_ros_namespace namespace="%s" />' % (indent, rest, indent, ns)

    src = re.sub(r'^([ \t]*)<group((?:\s+[\w-]+="[^"]*")*\s+ns="[^"]+"(?:\s+[\w-]+="[^"]*")*)\s*>', group, src, flags=re.M)

    # nodes
    def node(m):
        tag = m.group(0)
        tag = re.sub(r'\btype="', 'exec="', tag)
        tag = re.sub(r'\bns="', 'namespace="', tag)
        tag = re.sub(r'\s+clear_params="[^"]*"', "", tag)
        if re.search(r'\brequired="true"', tag):
            tag = re.sub(r'\s+required="true"', "", tag)
            notes.append("required=true dropped: %s" % tag[:80])
        if 'pkg="nodelet"' in tag or 'exec="nodelet"' in tag:
            notes.append("nodelet: %s" % tag[:100])
        if 'pkg="rviz"' in tag:
            tag = tag.replace('pkg="rviz"', 'pkg="rviz2"').replace('exec="rviz"', 'exec="rviz2"')
        return tag

    src = re.sub(r"<node\b[^>]*>", node, src, flags=re.S)
    src = re.sub(r"<env\b", "<set_env", src)
    src = re.sub(r'(<arg\b[^>]*?)\bdoc="', r'\1description="', src)

    # top-level <arg name value> -> <let>; includes keep <arg>
    out, depth = [], 0
    for line in src.split("\n"):
        if re.search(r"<include\b", line) and not re.search(r"/>\s*$", line):
            depth += 1
        if depth == 0 and re.search(r"<arg\b[^>]*\bvalue=", line) and "default=" not in line:
            line = re.sub(r"<arg\b", "<let", line)
        if "</include>" in line:
            depth -= 1
        out.append(line)
    src = "\n".join(out)

    # include targets: prefer an existing .launch.py / .launch.xml in the target package
    def include(m):
        pkg, rel = m.group(1), m.group(2)
        d = PKG_DIR.get(pkg)
        if d and not os.path.exists(os.path.join(d, rel)):
            for alt in (rel + ".py", rel + ".xml", rel.replace(".launch.xml", ".launch.py"), rel.replace(".launch.xml", ".launch")):
                if os.path.exists(os.path.join(d, alt)):
                    return "$(find-pkg-share %s)/%s" % (pkg, alt)
            notes.append("include target missing: %s/%s" % (pkg, rel))
        elif not d:
            notes.append("include from external package: %s/%s" % (pkg, rel))
        return m.group(0)

    src = re.sub(r"\$\(find-pkg-share ([\w-]+)\)/([\w/.-]+\.launch(?:\.xml|\.py)?)", include, src)

    for i, line in enumerate(src.split("\n"), 1):
        if re.search(r"<rosparam|<machine|<test\b|textfile=|command=|\$\(anon|\$\(dirname|launch-prefix|<param[^>]*\btype=\"(double|yaml)\"", line):
            notes.append("L%d: %s" % (i, line.strip()[:120]))
    src = re.sub(r'(<param\b[^>]*\btype=")double"', r'\1float"', src)
    return src, notes


if __name__ == "__main__":
    for path in sys.argv[1:]:
        new, notes = convert(open(path).read())
        open(path, "w").write(new)
        print("== %s" % path)
        for n in notes:
            print("  " + n)
