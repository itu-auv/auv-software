#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET


LANE_CAMERA_Y = {"a": 15.5, "b": 3.3, "c": -9.0, "d": -21.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lane", choices=LANE_CAMERA_Y, required=True)
    args = parser.parse_args()

    tree = ET.parse(args.source)
    camera_pose = tree.getroot().find(".//gui/camera[@name='user_camera']/pose")
    if camera_pose is None:
        raise RuntimeError("user_camera pose is missing from semi-final pool world")

    pose = camera_pose.text.split()
    pose[1] = str(LANE_CAMERA_Y[args.lane])
    camera_pose.text = " ".join(pose)
    tree.write(args.output, encoding="unicode", xml_declaration=True)

    # The command output is stored in a diagnostic ROS parameter.
    print(args.output)


if __name__ == "__main__":
    main()
