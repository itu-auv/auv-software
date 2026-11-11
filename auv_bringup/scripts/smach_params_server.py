#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from auv_common_lib import ros
from auv_common_lib.reconfigure import Server
from rclpy.node import Node

# name, default, description, min, max
# D
# torpedo searchforprop -
# bin searchforprop +
# octagon searchforprop -
# A
# torpedo searchforprop +
# bin searchforprop -
# octagon searchforprop +
SmachParametersConfig = [
    (
        "selected_role",
        "survey_repair",
        "Selected role for mission: survey_repair or search_rescue",
    ),
    ("torpedo_map", "fire", "Selected torpedo map"),
    (
        "torpedo_priority",
        "da3",
        "Preferred source for torpedo closest frame: realsense or da3",
    ),
    ("wall_reference_yaw", 0.0, "Yaw angle of the wall reference frame", -3.14, 3.14),
    ("gate_exit_angle", 0.0, "Gate exit angle in degrees", -180.0, 180.0),
    ("bin_exit_angle", 0.0, "Bin exit angle in degrees", -180.0, 180.0),
    ("torpedo_exit_angle", 0.0, "Torpedo exit angle in degrees", -180.0, 180.0),
]


class SmachParametersServer(Node):
    def __init__(self):
        super().__init__("smach_parameters_server")
        self.srv = Server(self, SmachParametersConfig, self.callback)

    def callback(self, config, level):
        self.get_logger().info(
            "Reconfigure Request: %s, %s, %s"
            % (
                config["selected_role"],
                config["wall_reference_yaw"],
                config["torpedo_priority"],
            )
        )
        return config


def main():
    ros.init()
    ros.spin(SmachParametersServer())


if __name__ == "__main__":
    main()
