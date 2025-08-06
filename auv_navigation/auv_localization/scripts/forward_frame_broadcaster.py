#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Quaternion, TransformStamped
from dynamic_reconfigure.client import Client as DynamicReconfigureClient
from auv_bringup.cfg import SmachParametersConfig


class ForwardFrameBroadcaster:
    """
    applies the wall_reference_yaw parameter as a yaw offset to odom frame, and broadcasts
    a new TF frame named 'forward_frame'.
    """

    def __init__(self):
        """Initializes the ForwardFrameBroadcaster node."""
        rospy.init_node("forward_frame_broadcaster_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.wall_reference_yaw = 0.0

        # Dynamic reconfigure client
        try:
            self.smach_params_client = DynamicReconfigureClient(
                "smach_parameters_server",
                timeout=5,
                config_callback=self.smach_params_callback,
            )
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Failed to connect to dynamic reconfigure server: {e}")
            self.smach_params_client = None

        # Parameters
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.forward_frame = rospy.get_param("~forward_frame", "forward_frame")
        self.publish_rate = rospy.Rate(rospy.get_param("~publish_rate", 10.0))

    def smach_params_callback(self, config):
        if "wall_reference_yaw" in config:
            rospy.loginfo_once(
                f"Received initial wall_reference_yaw: {config['wall_reference_yaw']}"
            )
            self.wall_reference_yaw = config["wall_reference_yaw"]

    def broadcast_forward_frame(self):
        if not self.tf_buffer.can_transform(
            self.odom_frame, self.odom_frame, rospy.Time(0), rospy.Duration(0.1)
        ):
            return
        # The orientation is determined solely by the wall_reference_yaw
        quat = tf_conversions.transformations.quaternion_from_euler(
            0, 0, self.wall_reference_yaw
        )

        # Create and populate the TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = (
            rospy.Time.now()
        )  # Almost instant timestamp, not a problem to set to now
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.forward_frame
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation = Quaternion(*quat)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    def spin(self):
        rospy.loginfo("Forward Frame Broadcaster node started.")
        while not rospy.is_shutdown():
            self.broadcast_forward_frame()
            self.publish_rate.sleep()


if __name__ == "__main__":
    try:
        node = ForwardFrameBroadcaster()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Forward Frame Broadcaster node shut down.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
