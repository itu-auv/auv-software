#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from dynamic_reconfigure.server import Server
from auv_bringup.cfg import SmachParametersConfig


def callback(config, level):
    rospy.loginfo(
        "Reconfigure Request: %s, %s, %s",
        config["selected_role"],
        config["wall_reference_yaw"],
        config["slalom_direction"],
    )
    return config


if __name__ == "__main__":
    rospy.init_node("smach_parameters_server", anonymous=False)

    srv = Server(SmachParametersConfig, callback)
    rospy.spin()
