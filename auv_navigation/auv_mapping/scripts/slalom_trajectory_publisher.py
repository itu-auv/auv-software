#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf
import numpy as np
import math
import yaml
import os
import rospkg

from geometry_msgs.msg import TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest

from dynamic_reconfigure.server import Server
from auv_mapping.cfg import SlalomTrajectoryConfig
from dynamic_reconfigure.client import Client


class SlalomTrajectoryPublisher(object):
    """
    This node publishes one TF frame for slalom entrance when triggered by a ROS 1 service.
    The frame calculations are based on the gate_exit frame and dynamic parameters.
    """

    def __init__(self):
        rospy.loginfo("Starting Slalom Trajectory Publisher node...")

        # Get config file path from parameter server
        try:
            rospack = rospkg.RosPack()
            default_path = os.path.join(
                rospack.get_path("auv_mapping"), "config", "slalom.yaml"
            )
            self.config_file = rospy.get_param("~config_file", default_path)
        except rospkg.common.ResourceNotFound:
            rospy.logerr(
                "auv_mapping package not found. Could not set default config file path."
            )
            self.config_file = ""

        self.parent_frame = "odom"
        self.gate_exit_frame = "gate_exit"

        # Setup dynamic reconfigure server.
        self.reconfigure_server = Server(
            SlalomTrajectoryConfig, self.reconfigure_callback
        )

        # Initialize TF broadcaster and listener
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Service to broadcast transforms
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.active = False
        self.pos_entrance = None
        self.faruk_y_offset = rospy.get_param("~faruk_y_offset", 0.0)
        self.faruk_x_offset = rospy.get_param("~faruk_x_offset", 0.0)

        # Create a service that will trigger the frame publishing
        self.srv = rospy.Service(
            "toggle_slalom_waypoints", SetBool, self.trigger_callback
        )

        # Create a timer to publish the frames at 4 Hz
        rospy.Timer(rospy.Duration(1.0 / 4.0), self.publish_loop)

        rospy.loginfo(
            "Slalom Trajectory Publisher is ready. Call the 'publish_slalom_waypoints' service."
        )

    def reconfigure_callback(self, config, level):
        """
        The callback function for dynamic reconfigure server.
        This function is called when a parameter is changed in the dynamic_reconfigure GUI.
        """
        self.faruk_y_offset = config.faruk_y_offset
        self.faruk_x_offset = config.faruk_x_offset

        # Save the updated parameters to the YAML file
        self.save_parameters()

        return config

    def trigger_callback(self, req):
        """
        The service callback that activates/deactivates the frame publishing loop.
        """
        self.active = req.data
        if self.active:
            rospy.loginfo("Activating slalom waypoint publishing...")
            return SetBoolResponse(
                success=True, message="Slalom waypoint publishing activated."
            )
        else:
            rospy.loginfo("Deactivating slalom waypoint publishing...")
            return SetBoolResponse(
                success=True, message="Slalom waypoint publishing deactivated."
            )

    def publish_loop(self, event):
        """
        The main loop that calculates and publishes the slalom waypoint frames.
        """
        if not self.active:
            return

        # Calculate and publish slalom_entrance
        try:
            t_gate_exit = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.gate_exit_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            gate_exit_pos = np.array(
                [
                    t_gate_exit.transform.translation.x,
                    t_gate_exit.transform.translation.y,
                    t_gate_exit.transform.translation.z,
                ]
            )
            gate_exit_q = [
                t_gate_exit.transform.rotation.x,
                t_gate_exit.transform.rotation.y,
                t_gate_exit.transform.rotation.z,
                t_gate_exit.transform.rotation.w,
            ]
            # Get the rotation matrix from the quaternion
            rotation_matrix = tf.transformations.quaternion_matrix(gate_exit_q)[:3, :3]

            # The y-axis in the gate_exit frame is (0, 1, 0)
            y_axis_in_gate_frame = np.array([0, 1, 0])
            # Transform the y-axis vector to the parent_frame (odom)
            y_axis_in_parent_frame = rotation_matrix.dot(y_axis_in_gate_frame)

            # The x-axis in the gate_exit frame is (1, 0, 0)
            x_axis_in_gate_frame = np.array([1, 0, 0])
            # Transform the x-axis vector to the parent_frame (odom)
            x_axis_in_parent_frame = rotation_matrix.dot(x_axis_in_gate_frame)

            # Calculate the new position by moving along the gate's y-axis and x-axis
            self.pos_entrance = (
                gate_exit_pos
                + y_axis_in_parent_frame * self.faruk_y_offset
                + x_axis_in_parent_frame * self.faruk_x_offset
            )
            self.send_transform(
                self.build_transform(
                    "pool_checkpoint",
                    self.parent_frame,
                    self.pos_entrance,
                    gate_exit_q,
                )
            )
        except Exception as e:
            rospy.logwarn_throttle(
                8,
                "Failed to get gate exit transform and publish slalom entrance: %s",
                e,
            )

    def build_transform(self, child_frame, parent_frame, pos, q):
        """Helper function to create and broadcast a TransformStamped message."""
        if pos is None or q is None:
            return None
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        return t

    def send_transform(self, transform: TransformStamped):
        if transform is None:
            return
        request = SetObjectTransformRequest()
        request.transform = transform
        response = self.set_object_transform_service.call(request)
        if not response.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {response.message}"
            )

    def save_parameters(self):
        """Save parameters to the YAML file."""
        params = {
            "faruk_y_offset": self.faruk_y_offset,
            "faruk_x_offset": self.faruk_x_offset,
        }
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(params, f, default_flow_style=False)
            rospy.loginfo(f"Parameters saved to {self.config_file}")
        except IOError as e:
            rospy.logerr(f"Failed to save parameters to {self.config_file}: {e}")


if __name__ == "__main__":
    try:
        rospy.init_node("slalom_trajectory_publisher", anonymous=True)
        node = SlalomTrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
