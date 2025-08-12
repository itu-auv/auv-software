#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from dynamic_reconfigure.server import Server
from auv_bringup.cfg import SmachParametersConfig


def callback(config, level):
    if config.pool_selection == "pool_a":
        config.gate_exit_angle = config.pool_a_gate_exit_angle
        config.slalom_exit_angle = config.pool_a_slalom_exit_angle
        config.bin_exit_angle = config.pool_a_bin_exit_angle
        config.torpedo_exit_angle = config.pool_a_torpedo_exit_angle
        rospy.loginfo(
            """Reconfigure Request: pool_selection={pool_selection}, selected_animal={selected_animal}, wall_reference_yaw={wall_reference_yaw}, slalom_direction={slalom_direction}, Pool A Angles - Gate: {pool_a_gate_exit_angle}, Slalom: {pool_a_slalom_exit_angle}, Bin: {pool_a_bin_exit_angle}, Torpedo: {pool_a_torpedo_exit_angle}""".format(
                **config
            )
        )
    elif config.pool_selection == "pool_d":
        config.gate_exit_angle = config.pool_d_gate_exit_angle
        config.slalom_exit_angle = config.pool_d_slalom_exit_angle
        config.bin_exit_angle = config.pool_d_bin_exit_angle
        config.torpedo_exit_angle = config.pool_d_torpedo_exit_angle
        rospy.loginfo(
            """Reconfigure Request: pool_selection={pool_selection}, selected_animal={selected_animal}, wall_reference_yaw={wall_reference_yaw}, slalom_direction={slalom_direction}, Pool D Angles - Gate: {pool_d_gate_exit_angle}, Slalom: {pool_d_slalom_exit_angle}, Bin: {pool_d_bin_exit_angle}, Torpedo: {pool_d_torpedo_exit_angle}""".format(
                **config
            )
        )
    else:
        rospy.loginfo(
            """Reconfigure Request: pool_selection={pool_selection}, selected_animal={selected_animal}, wall_reference_yaw={wall_reference_yaw}, slalom_direction={slalom_direction}, Active angles - Gate: {gate_exit_angle}, Slalom: {slalom_exit_angle}, Bin: {bin_exit_angle}, Torpedo: {torpedo_exit_angle}""".format(
                **config
            )
        )

    return config


if __name__ == "__main__":
    rospy.init_node("smach_parameters_server", anonymous=False)

    srv = Server(SmachParametersConfig, callback)
    rospy.spin()
