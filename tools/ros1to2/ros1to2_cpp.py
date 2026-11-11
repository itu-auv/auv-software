#!/usr/bin/env python3
"""Mechanical roscpp -> rclcpp rewrites for C++ sources/headers. Prints leftovers.

usage: ros1to2_cpp.py [--node EXPR] files...     EXPR: pointer-like node expression (default nh_)
"""
import glob
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MSG_PKGS = "geometry_msgs sensor_msgs std_msgs nav_msgs vision_msgs visualization_msgs auv_msgs ultralytics_ros stereo_msgs diagnostic_msgs".split()
SRVS = {"std_srvs": {"Trigger", "SetBool", "Empty"}, "auv_msgs": set()}
for f in glob.glob(os.path.join(REPO, "auv_msgs", "srv", "*.srv")):
    SRVS["auv_msgs"].add(os.path.basename(f)[:-4])


def snake(name):
    return re.sub(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def kind(pkg, name):
    return "srv" if name in SRVS.get(pkg, ()) else "msg"


def convert(src, node):
    log = "%s->get_logger()" % node

    def inc(m):
        pkg, name = m.group(2), m.group(3)
        if name in (
            "image_encodings",
            "point_cloud2_iterator",
            "point_cloud_conversion",
            "distortion_models",
            "fill_image",
        ):
            return "#include %s%s/%s.hpp%s" % (m.group(1), pkg, name, m.group(4))
        return "#include %s%s/%s/%s.hpp%s" % (
            m.group(1),
            pkg,
            kind(pkg, name),
            snake(name),
            m.group(4),
        )

    pk = "|".join(MSG_PKGS + ["std_srvs"])
    src = re.sub(r'#include\s+([<"])(%s)/(\w+)\.h([>"])' % pk, inc, src)
    src = re.sub(
        r"\b(%s)::(?!msg::|srv::)([A-Z]\w*)" % pk,
        lambda m: "%s::%s::%s" % (m.group(1), kind(m.group(1), m.group(2)), m.group(2)),
        src,
    )
    src = re.sub(r"::ConstPtr\b", "::ConstSharedPtr", src)
    src = re.sub(
        r"(?<!Shared)(?<!Unique)(?<!Const)\b(\w+::(?:msg|srv)::\w+)::Ptr\b",
        r"\1::SharedPtr",
        src,
    )

    for a, b in [
        (r'#include\s+[<"]ros/ros\.h[>"]', '#include "rclcpp/rclcpp.hpp"'),
        (
            r"tf2_geometry_msgs/tf2_geometry_msgs\.h\b",
            "tf2_geometry_msgs/tf2_geometry_msgs.hpp",
        ),
        (
            r"tf2_sensor_msgs/tf2_sensor_msgs\.h\b",
            "tf2_sensor_msgs/tf2_sensor_msgs.hpp",
        ),
        (r"tf2_eigen/tf2_eigen\.h\b", "tf2_eigen/tf2_eigen.hpp"),
        (r"cv_bridge/cv_bridge\.h\b", "cv_bridge/cv_bridge.hpp"),
        (
            r"image_transport/image_transport\.h\b",
            "image_transport/image_transport.hpp",
        ),
        (r"message_filters/subscriber\.h\b", "message_filters/subscriber.h"),
        (r"pcl_ros/transforms\.h\b", "pcl_ros/transforms.hpp"),
        (r"pcl_ros/point_cloud\.h\b", "pcl_conversions/pcl_conversions.h"),
        (
            r"camera_info_manager/camera_info_manager\.h\b",
            "camera_info_manager/camera_info_manager.hpp",
        ),
    ]:
        src = re.sub(a, b, src)

    # logging
    lv = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARN": "WARN",
        "ERROR": "ERROR",
        "FATAL": "FATAL",
    }
    for k, v in lv.items():
        src = re.sub(
            r"\bROS_%s_STREAM_THROTTLE\(\s*([^,]+),\s*" % k,
            lambda m: "RCLCPP_%s_STREAM_THROTTLE(%s, *%s->get_clock(), static_cast<int>((%s) * 1000), "
            % (v, log, node, m.group(1).strip()),
            src,
        )
        src = re.sub(
            r"\bROS_%s_THROTTLE\(\s*([^,]+),\s*" % k,
            lambda m: "RCLCPP_%s_THROTTLE(%s, *%s->get_clock(), static_cast<int>((%s) * 1000), "
            % (v, log, node, m.group(1).strip()),
            src,
        )
        src = re.sub(
            r"\bROS_%s_STREAM_ONCE\(" % k, "RCLCPP_%s_STREAM_ONCE(%s, " % (v, log), src
        )
        src = re.sub(r"\bROS_%s_ONCE\(" % k, "RCLCPP_%s_ONCE(%s, " % (v, log), src)
        src = re.sub(r"\bROS_%s_STREAM\(" % k, "RCLCPP_%s_STREAM(%s, " % (v, log), src)
        src = re.sub(r"\bROS_%s\(" % k, "RCLCPP_%s(%s, " % (v, log), src)

    # time
    src = re.sub(r"\bros::Time::now\(\)", "%s->now()" % node, src)
    src = re.sub(r"\bros::Time\(0\)", "rclcpp::Time(0, 0, RCL_ROS_TIME)", src)
    src = re.sub(
        r"\bros::Duration\(([^()]+)\)", r"rclcpp::Duration::from_seconds(\1)", src
    )
    src = re.sub(r"\bros::Duration\b", "rclcpp::Duration", src)
    src = re.sub(r"\bros::Time\b", "rclcpp::Time", src)
    src = re.sub(r"\.toSec\(\)", ".seconds()", src)
    src = re.sub(r"\bros::ok\(\)", "rclcpp::ok()", src)
    src = re.sub(r"\bros::Publisher\b", "rclcpp::PublisherBase::SharedPtr", src)
    src = re.sub(r"\bros::Subscriber\b", "rclcpp::SubscriptionBase::SharedPtr", src)
    src = re.sub(r"\bros::ServiceServer\b", "rclcpp::ServiceBase::SharedPtr", src)
    src = re.sub(r"\bros::NodeHandle\b", "rclcpp::Node::SharedPtr", src)

    notes = []
    for i, line in enumerate(src.split("\n"), 1):
        if re.search(
            r"\bros::|ROS_|boost::|nodelet|\.param<|getParam|advertise|\.subscribe\(|dynamic_reconfigure|PublisherBase|SubscriptionBase|ServiceBase|XmlRpc",
            line,
        ) and not line.strip().startswith("//"):
            notes.append("L%d: %s" % (i, line.strip()[:120]))
    return src, notes


def main():
    argv = sys.argv[1:]
    node = "nh_"
    if argv and argv[0] == "--node":
        node, argv = argv[1], argv[2:]
    for path in argv:
        new, notes = convert(open(path).read(), node)
        open(path, "w").write(new)
        print("== %s (%d notes)" % (path, len(notes)))
        for n in notes:
            print("  " + n)


if __name__ == "__main__":
    main()
