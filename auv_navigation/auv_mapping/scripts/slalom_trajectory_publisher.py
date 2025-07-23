#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf
import numpy as np
import math

from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest

from dynamic_reconfigure.server import Server
from auv_mapping.cfg import SlalomTrajectoryConfig


class SlalomTrajectoryPublisher(object):
    """
    This node publishes four TF frames for slalom waypoints when triggered by a ROS 1 service.
    The frame calculations are based entirely on parameters loaded from the ROS parameter server.
    """

    def __init__(self):
        rospy.loginfo("Starting Slalom Trajectory Publisher node...")

        # Initialize parameters with default values
        self.gate_dist = 2.0
        self.offset2 = 1.0
        self.exit_distance = 0.8
        self.offset3 = -1.0
        self.vertical_dist = 1.5
        self.slalom_entrance_backed_distance = 1.0
        self.parent_frame = "odom"
        self.gate_exit_frame = "gate_exit"
        self.slalom_white_frame = "white_pipe_link"
        self.slalom_red_frame = "red_pipe_link"

        # Setup dynamic reconfigure server
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
        self.q_orientation = None
        self.pos_entrance, self.pos_wp1, self.pos_wp2, self.pos_wp3, self.pos_exit = (
            None,
            None,
            None,
            None,
            None,
        )

        # Create a service that will trigger the frame publishing
        self.srv = rospy.Service(
            "publish_slalom_waypoints", Trigger, self.trigger_callback
        )

        # Create a timer to publish the frames at 4 Hz
        rospy.Timer(rospy.Duration(1.0 / 4.0), self.publish_loop)

        rospy.loginfo(
            "Slalom Trajectory Publisher is ready. Call the 'publish_slalom_waypoints' service."
        )

    def reconfigure_callback(self, config, level):
        """
        The callback function for dynamic reconfigure server.
        """
        rospy.loginfo("Updating slalom trajectory parameters...")
        self.gate_dist = config.gate_to_slalom_entrance_distance
        self.offset2 = config.second_slalom_offset
        self.offset3 = config.third_slalom_offset
        self.vertical_dist = config.vertical_distance_between_slalom_clusters
        self.slalom_entrance_backed_distance = config.slalom_entrance_backed_distance
        return config

    def trigger_callback(self, req):
        """
        The service callback that activates the frame publishing loop.
        """
        rospy.loginfo("Activating slalom waypoint publishing...")
        self.active = True
        return TriggerResponse(
            success=True, message="Slalom waypoint publishing activated."
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
                rospy.Duration(1.0),
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
                rospy.Duration(1.0),
            )
            pos_red = np.array(
                [
                    t_red.transform.translation.x,
                    t_red.transform.translation.y,
                    t_red.transform.translation.z,
                ]
            )
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

            # Calculate and publish slalom_waypoint_2
            self.pos_wp2 = (
                self.pos_wp1
                + forward_vec * self.vertical_dist
                + slalom_parallel_vector * self.offset2
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_2",
                    self.parent_frame,
                    self.pos_wp2,
                    self.q_orientation,
                )
            )

            # Calculate and publish slalom_waypoint_3
            self.pos_wp3 = (
                self.pos_wp1
                + forward_vec * (self.vertical_dist + self.vertical_dist)
                + slalom_parallel_vector * self.offset3
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_3",
                    self.parent_frame,
                    self.pos_wp3,
                    self.q_orientation,
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
                rospy.Duration(1.0),
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
        self.tf_broadcaster.sendTransform(t)
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


if __name__ == "__main__":
    try:
        rospy.init_node("slalom_trajectory_publisher", anonymous=True)
        node = SlalomTrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
