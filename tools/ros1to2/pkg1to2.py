#!/usr/bin/env python3
"""catkin -> ament_cmake packaging for packages that only install scripts/launch/config (or nothing: metapackages).
usage: pkg1to2.py package_dir..."""
import glob, os, re, subprocess, sys

DEP = {
    "rospy": ["rclpy"],
    "roscpp": ["rclcpp"],
    "tf": ["tf2_ros", "tf_transformations"],
    "tf_conversions": ["tf_transformations"],
    "actionlib": [],
    "dynamic_reconfigure": [],
    "message_generation": [],
    "message_runtime": [],
    "rviz": ["rviz2"],
    "nodelet": [],
    "rosserial_python": [],
    "rostopic": [],
    "roslaunch": [],
    "rosbag": ["rosbag2"],
    "smach_ros": ["smach_ros"],
}

for d in sys.argv[1:]:
    d = d.rstrip("/")
    px = os.path.join(d, "package.xml")
    src = open(px).read()
    name = re.search(r"<name>([^<]+)</name>", src).group(1)
    src = src.replace(
        "<buildtool_depend>catkin</buildtool_depend>",
        "<buildtool_depend>ament_cmake</buildtool_depend>",
    )

    def dep(m):
        tag, pkg = m.group(1), m.group(2)
        if pkg not in DEP:
            return m.group(0)
        return (
            "\n".join(
                "%s<%s>%s</%s>"
                % (
                    m.group(0)[: len(m.group(0)) - len(m.group(0).lstrip())],
                    tag,
                    p,
                    tag,
                )
                for p in DEP[pkg]
            )
            or "<!-- drop -->"
        )

    src = re.sub(
        r"[ \t]*<(\w*depend)>([\w-]+)</\1>",
        lambda m: dep(m) if m.group(2) in DEP else m.group(0),
        src,
    )
    src = re.sub(r"\n<!-- drop -->", "", src)
    uses_common = (
        subprocess.run(
            ["grep", "-rlq", "auv_common_lib", d, "--include=*.py"]
        ).returncode
        == 0
    )
    if (
        uses_common
        and "<exec_depend>auv_common_lib</exec_depend>" not in src
        and "<depend>auv_common_lib</depend>" not in src
        and name != "auv_common_lib"
    ):
        src = re.sub(
            r"(\n[ \t]*)(<export)",
            r"\1<exec_depend>auv_common_lib</exec_depend>\1\2",
            src,
            count=1,
        )
    src = re.sub(
        r"<export\s*/>|<export>\s*</export>",
        "<export>\n    <build_type>ament_cmake</build_type>\n  </export>",
        src,
    )
    if "<build_type>" not in src:
        src = src.replace(
            "<export>", "<export>\n    <build_type>ament_cmake</build_type>", 1
        )
    src = re.sub(r"\s*<metapackage\s*/>", "", src)
    open(px, "w").write(src)

    old = open(os.path.join(d, "CMakeLists.txt")).read()
    scripts = re.findall(r"(scripts/[\w./-]+\.py)", old) or sorted(
        os.path.relpath(p, d) for p in glob.glob(os.path.join(d, "scripts", "*.py"))
    )
    scripts = list(dict.fromkeys(scripts))
    dirs = [
        x
        for x in ("launch", "config", "rviz", "urdf", "meshes", "models", "worlds")
        if os.path.isdir(os.path.join(d, x))
    ]
    out = [
        "cmake_minimum_required(VERSION 3.10.2)",
        "project(%s)" % name,
        "",
        "find_package(ament_cmake REQUIRED)",
        "",
    ]
    if scripts:
        out += (
            ["install(PROGRAMS"]
            + ["  " + s for s in scripts]
            + ["  DESTINATION lib/${PROJECT_NAME}", ")"]
        )
    if dirs:
        out += [
            "install(DIRECTORY %s DESTINATION share/${PROJECT_NAME})" % " ".join(dirs)
        ]
    out += ["", "ament_package()", ""]
    open(os.path.join(d, "CMakeLists.txt"), "w").write("\n".join(out))
    print("packaged", name, "scripts:", len(scripts), "dirs:", dirs)
