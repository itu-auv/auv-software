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
        self.offset3 = -1.0
        self.vertical_dist = 1.5
        self.lane_angle_rad = 0.0
        self.parent_frame = "odom"
        self.gate_exit_frame = "gate_exit"
        self.slalom_white_frame = "slalom_white_link"
        self.slalom_red_frame = "slalom_red_link"

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
        self.lane_angle_rad = config.forward_lane_angle_rad
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

        # --- Coordinate System Setup ---
        forward_vec = np.array(
            [math.cos(self.lane_angle_rad), math.sin(self.lane_angle_rad), 0.0]
        )
        pool_parallel_vec = np.array(
            [-math.sin(self.lane_angle_rad), math.cos(self.lane_angle_rad), 0.0]
        )
        self.q_orientation = tf.transformations.quaternion_from_euler(
            0.0, 0.0, self.lane_angle_rad
        )

        try:
            # Get the initial `gate_exit` frame's position
            t_gate_exit = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.gate_exit_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            gate_exit_pos = np.array(
                [
                    t_gate_exit.transform.translation.x,
                    t_gate_exit.transform.translation.y,
                    t_gate_exit.transform.translation.z,
                ]
            )

            # Calculate slalom_entrance
            self.pos_entrance = gate_exit_pos + pool_parallel_vec * self.gate_dist

            # Calculate slalom_waypoint_1
            t_white = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.slalom_white_frame,
                rospy.Time(0),
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
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            pos_red = np.array(
                [
                    t_red.transform.translation.x,
                    t_red.transform.translation.y,
                    t_red.transform.translation.z,
                ]
            )

            self.pos_wp1 = (pos_white + pos_red) / 2.0

            # Calculate slalom_waypoint_2
            self.pos_wp2 = (
                self.pos_wp1
                + forward_vec * self.vertical_dist
                + pool_parallel_vec * self.offset2
            )

            # Calculate slalom_waypoint_3
            self.pos_wp3 = (
                self.pos_wp2
                + forward_vec * self.vertical_dist
                + pool_parallel_vec * self.offset3
            )

            # Calculate slalom_exit
            self.pos_exit = self.pos_wp3 + forward_vec * 0.5

            # Broadcast all frames
            self.send_transform(
                self.build_transform(
                    "slalom_entrance",
                    self.parent_frame,
                    self.pos_entrance,
                    self.q_orientation,
                )
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_1",
                    self.parent_frame,
                    self.pos_wp1,
                    self.q_orientation,
                )
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_2",
                    self.parent_frame,
                    self.pos_wp2,
                    self.q_orientation,
                )
            )
            self.send_transform(
                self.build_transform(
                    "slalom_waypoint_3",
                    self.parent_frame,
                    self.pos_wp3,
                    self.q_orientation,
                )
            )
            self.send_transform(
                self.build_transform(
                    "slalom_exit",
                    self.parent_frame,
                    self.pos_exit,
                    self.q_orientation,
                )
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr("Could not perform transform or calculation: %s", e)

    def build_transform(self, child_frame, parent_frame, pos, q):
        """Helper function to create and broadcast a TransformStamped message."""
        if pos is None or q is None:
            return
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

    def send_transform(self, transform: TransformStamped):
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
