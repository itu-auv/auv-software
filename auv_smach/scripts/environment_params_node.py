#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from auv_smach.cfg import WorldConfig


def callback(config, level):
    rospy.set_param("/env/lane_forward_angle", config.lane_forward_angle)
    return config


if __name__ == "__main__":
    rospy.init_node("auv_smach", anonymous=False)

    srv = Server(WorldConfig, callback)
    rospy.spin()
