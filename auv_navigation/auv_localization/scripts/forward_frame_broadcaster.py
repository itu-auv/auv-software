#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf_conversions
from geometry_msgs.msg import Quaternion, TransformStamped
from dynamic_reconfigure.client import Client as DynamicReconfigureClient
import tf2_ros
from auv_bringup.cfg import SmachParametersConfig


class ForwardFrameBroadcaster:
    """
    applies the wall_reference_yaw parameter as a yaw offset to odom frame, and broadcasts
    a new TF frame named 'forward_frame'.
    """

    def __init__(self):
        """Initializes the ForwardFrameBroadcaster node."""
        rospy.init_node("forward_frame_broadcaster_node")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.wall_reference_yaw = 0.0
        self.transform = TransformStamped()

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.forward_frame = rospy.get_param("~forward_frame", "forward_frame")
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)

        try:
            self.smach_params_client = DynamicReconfigureClient(
                "smach_parameters_server",
                timeout=5,
                config_callback=self.smach_params_callback,
            )
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Failed to connect to dynamic reconfigure server: {e}")
            self.smach_params_client = None

    def smach_params_callback(self, config):
        if "wall_reference_yaw" in config:
            rospy.loginfo_once(
                f"Received initial wall_reference_yaw: {config['wall_reference_yaw']}"
            )
            self.wall_reference_yaw = config["wall_reference_yaw"]

    def spin(self):
        rospy.loginfo(
            "Forward Frame Broadcaster node started. Waiting for parameters..."
        )
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            quat = tf_conversions.transformations.quaternion_from_euler(
                0, 0, self.wall_reference_yaw
            )

            self.transform.header.stamp = rospy.Time.now()
            self.transform.header.frame_id = self.odom_frame
            self.transform.child_frame_id = self.forward_frame
            self.transform.transform.translation.x = 0.0
            self.transform.transform.translation.y = 0.0
            self.transform.transform.translation.z = 0.0
            self.transform.transform.rotation = Quaternion(*quat)

            self.tf_broadcaster.sendTransform(self.transform)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ForwardFrameBroadcaster()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Forward Frame Broadcaster node shut down.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
