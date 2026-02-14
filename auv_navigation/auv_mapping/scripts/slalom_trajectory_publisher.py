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

from dynamic_reconfigure.server import Server
from auv_mapping.cfg import SlalomTrajectoryConfig
from dynamic_reconfigure.client import Client


class SlalomTrajectoryPublisher(object):
    """
    This node publishes four TF frames for slalom waypoints when triggered by a ROS 1 service.
    The frame calculations are based entirely on parameters loaded from the ROS parameter server.
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

        # Parameters will be loaded by the launch file.
        # We read them here using rospy.get_param.
        # The default values in the get_param calls serve as fallbacks.
        self.gate_dist = rospy.get_param("~gate_to_slalom_entrance_distance", 2.5)
        self.offset2 = rospy.get_param("~second_slalom_offset", 0.5)
        self.offset3 = rospy.get_param("~third_slalom_offset", 0.0)
        self.vertical_dist = rospy.get_param(
            "~vertical_distance_between_slalom_clusters", 2.0
        )
        self.slalom_entrance_backed_distance = rospy.get_param(
            "~slalom_entrance_backed_distance", 2.0
        )
        self.exit_distance = (
            0.5  # This parameter is not part of the dynamic reconfigure config
        )

        self.parent_frame = "odom"
        self.gate_exit_frame = "gate_exit"
        self.slalom_white_frame = "white_pipe_link"
        self.slalom_red_frame = "red_pipe_link"

        # Setup dynamic reconfigure server.
        # The server will automatically use the parameters from the ROS Parameter Server
        # for its initial values. The callback will be called only on subsequent changes.
        self.reconfigure_server = Server(
            SlalomTrajectoryConfig, self.reconfigure_callback
        )

        # Initialize TF broadcaster and listener
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Service to broadcast transforms
        self.set_object_transform_pub = rospy.Publisher(
            "set_object_transform", TransformStamped, queue_size=10
        )
        self.active = False
        self.q_orientation = None
        self.pos_entrance, self.pos_wp1, self.pos_wp2, self.pos_wp3, self.pos_exit = (
            None,
            None,
            None,
            None,
            None,
        )
        self.navigation_mode = "left"  # Default value
        self.smach_params_client = Client(
            "smach_parameters_server",
            timeout=10,
            config_callback=self.smach_params_callback,
        )

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
        self.gate_dist = config.gate_to_slalom_entrance_distance
        self.offset2 = config.second_slalom_offset
        self.offset3 = config.third_slalom_offset
        self.vertical_dist = config.vertical_distance_between_slalom_clusters
        self.slalom_entrance_backed_distance = config.slalom_entrance_backed_distance

        # Save the updated parameters to the YAML file
        self.save_parameters()

        return config

    def smach_params_callback(self, config):
        """
        Callback for the smach parameters server.
        """
        if config is None:
            rospy.logwarn("Could not get parameters from smach_parameters_server")
            return
        self.navigation_mode = config.slalom_direction
        rospy.loginfo(f"Slalom navigation_mode updated to: {self.navigation_mode}")

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

        try:
            # --- First, get the pipe locations to establish the coordinate system ---
            t_white = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.slalom_white_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            pos_white = np.array(
                [
                    t_white.transform.translation.x,
                    t_white.transform.translation.y,
                    t_white.transform.translation.z,
                ]
            )
            t_red = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.slalom_red_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
            pos_red = np.array(
                [
                    t_red.transform.translation.x,
                    t_red.transform.translation.y,
                    t_red.transform.translation.z,
                ]
            )
            if self.navigation_mode == "right":
                slalom_parallel_vector = pos_white - pos_red
            elif self.navigation_mode == "left":
                slalom_parallel_vector = pos_red - pos_white
            else:
                rospy.logwarn("Invalid navigation mode. Defaulting to 'left'.")
                slalom_parallel_vector = pos_red - pos_white

            slalom_parallel_vector = slalom_parallel_vector / np.linalg.norm(
                slalom_parallel_vector
            )
            # Forward vector is the normal of the vector between pipes
            forward_vec = np.array(
                [-slalom_parallel_vector[1], slalom_parallel_vector[0], 0.0]
            )
            forward_vec = forward_vec / np.linalg.norm(forward_vec)

            # --- Frame Calculation and Publishing ---

            # Calculate slalom_waypoint_1 (midpoint of the pipes)
            self.pos_wp1 = (pos_white + pos_red) / 2.0

            # Ensure the forward vector points away from the odom frame origin
            if np.dot(forward_vec, self.pos_wp1) < 0:
                forward_vec = -forward_vec

            # Angle for orientation
            slalom_forward_angle = math.atan2(forward_vec[1], forward_vec[0])
            self.q_orientation = tf.transformations.quaternion_from_euler(
                0.0, 0.0, slalom_forward_angle
            )

            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_1",
                    self.parent_frame,
                    self.pos_wp1,
                    self.q_orientation,
                )
            )

            # Calculate intermediate frame orientations
            self.pos_wp2 = (
                self.pos_wp1
                + forward_vec * self.vertical_dist
                + slalom_parallel_vector * self.offset2
            )
            self.pos_wp3 = (
                self.pos_wp1
                + forward_vec * (self.vertical_dist + self.vertical_dist)
                + slalom_parallel_vector * self.offset3
            )

            dir_vec1 = self.pos_wp2 - self.pos_wp1
            dir_vec1 = dir_vec1 / np.linalg.norm(dir_vec1)
            angle1 = math.atan2(dir_vec1[1], dir_vec1[0])
            q1 = tf.transformations.quaternion_from_euler(0, 0, angle1)

            dir_vec2 = self.pos_wp3 - self.pos_wp2
            dir_vec2 = dir_vec2 / np.linalg.norm(dir_vec2)
            angle2 = math.atan2(dir_vec2[1], dir_vec2[0])
            q2 = tf.transformations.quaternion_from_euler(0, 0, angle2)

            # Publish intermediate frames
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_1_inter",
                    self.parent_frame,
                    self.pos_wp1,
                    q1,
                )
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_2_inter",
                    self.parent_frame,
                    self.pos_wp2,
                    q2,
                )
            )

            # Publish main waypoints with updated orientations
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_2",
                    self.parent_frame,
                    self.pos_wp2,
                    q1,
                )
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_3",
                    self.parent_frame,
                    self.pos_wp3,
                    q2,
                )
            )

            # Calculate and publish slalom_exit
            self.pos_exit = self.pos_wp3 + forward_vec * self.exit_distance
            self.send_transform(
                self.build_transform(
                    "slalom_exit",
                    self.parent_frame,
                    self.pos_exit,
                    self.q_orientation,
                )
            )

            # Broadcast the debug frame
            self.send_transform(
                self.build_transform(
                    "slalom_debug_frame",
                    self.parent_frame,
                    np.array([0.5, 0.0, 0.0]),
                    self.q_orientation,
                )
            )
        except Exception as e:
            rospy.logwarn_throttle(
                8, "Failed to get pipe locations and publish slalom waypoints: %s", e
            )

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
            # Calculate the new position by moving along the gate's y-axis
            self.pos_entrance = gate_exit_pos + y_axis_in_parent_frame * self.gate_dist
            self.send_transform(
                self.build_transform(
                    "slalom_entrance",
                    self.parent_frame,
                    self.pos_entrance,
                    gate_exit_q,
                )
            )

            # The x-axis in the gate_exit frame is (1, 0, 0)
            x_axis_in_gate_frame = np.array([1, 0, 0])
            # Transform the x-axis vector to the parent_frame (odom)
            x_axis_in_parent_frame = rotation_matrix.dot(x_axis_in_gate_frame)
            # Calculate the new position for the backed frame
            pos_entrance_backed = (
                self.pos_entrance
                - x_axis_in_parent_frame * self.slalom_entrance_backed_distance
            )
            self.send_transform(
                self.build_transform(
                    "slalom_entrance_backed",
                    self.parent_frame,
                    pos_entrance_backed,
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
            rospy.logwarn_throttle(5, "Transform is None")
            return
        self.set_object_transform_pub.publish(transform)

    def save_parameters(self):
        """Save parameters to the YAML file."""
        params = {
            "gate_to_slalom_entrance_distance": self.gate_dist,
            "second_slalom_offset": self.offset2,
            "third_slalom_offset": self.offset3,
            "vertical_distance_between_slalom_clusters": self.vertical_dist,
            "slalom_entrance_backed_distance": self.slalom_entrance_backed_distance,
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
