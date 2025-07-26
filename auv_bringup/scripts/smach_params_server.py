#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from dynamic_reconfigure.server import Server
from auv_bringup.cfg import SmachParametersConfig


def callback(config, level):
    rospy.loginfo(
        """Reconfigure Request: {selected_animal}, {wall_reference_yaw}, {slalom_direction}""".format(
            **config
        )
    )
    return config


if __name__ == "__main__":
    rospy.init_node("smach_parameters_server", anonymous=False)

    srv = Server(SmachParametersConfig, callback)
    rospy.spin()
