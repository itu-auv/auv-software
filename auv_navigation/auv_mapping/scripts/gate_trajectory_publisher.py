#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional
import math
import rospy
import threading
import tf2_ros
import tf_conversions
import geometry_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from nav_msgs.msg import Odometry
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client as DynamicReconfigureClient
from auv_mapping.cfg import GateTrajectoryConfig
from auv_bringup.cfg import SmachParametersConfig


class TransformServiceNode:
    def __init__(self):
        self.is_enabled = False
        rospy.init_node("create_gate_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Dynamic reconfigure server for gate parameters
        self.gate_frame_1 = "gate_shark_link"
        self.gate_frame_2 = "gate_sawfish_link"
        self.target_gate_frame = "gate_shark_link"
        self.entrance_offset = 1.0
        self.exit_offset = 1.0
        self.z_offset = 0.5
        self.parallel_shift_offset = 0.20
        self.rescuer_distance = 1.0
        self.wall_reference_yaw = 0.0
        self.reconfigure_server = Server(
            GateTrajectoryConfig, self.reconfigure_callback
        )

        # Client for auv_smach parameters
        if SmachParametersConfig is not None:
            self.smach_params_client = DynamicReconfigureClient(
                "smach_parameters_server",
                timeout=10,
                config_callback=self.smach_params_callback,
            )
        else:
            rospy.logerr("Smach dynamic reconfigure client not started.")

        # Service to broadcast transforms
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.gate_passage_frame = "gate_passage"

        # Verify that we have valid frame names
        if not all([self.gate_frame_1, self.gate_frame_2]):
            rospy.logerr("Missing required gate frame parameters")
            rospy.signal_shutdown("Missing required parameters")

        self.set_enable_service = rospy.Service(
            "toggle_gate_trajectory", SetBool, self.handle_enable_service
        )

        # --- Parameters for fallback (single-frame) mode
        self.fallback_entrance_offset = rospy.get_param(
            "~fallback_entrance_offset", 1.0
        )
        self.fallback_exit_offset = rospy.get_param("~fallback_exit_offset", 1.0)
        self.MIN_GATE_SEPARATION = rospy.get_param("~min_gate_separation", 0.2)
        self.MAX_GATE_SEPARATION = rospy.get_param("~max_gate_separation", 2.5)
        self.MIN_GATE_SEPARATION_THRESHOLD = 0.3

        self.odom_sub = rospy.Subscriber("odometry", Odometry, self.odom_callback)
        self.latest_odom = None

    def smach_params_callback(self, config):
        """Callback for receiving parameters from the auv_smach node."""
        rospy.loginfo("Received smach parameters update: %s", config)
        if "wall_reference_yaw" in config:
            self.wall_reference_yaw = config["wall_reference_yaw"]
        if "selected_animal" in config:
            if config["selected_animal"] == "shark":
                self.target_gate_frame = "gate_shark_link"
            elif config["selected_animal"] == "sawfish":
                self.target_gate_frame = "gate_sawfish_link"

    def odom_callback(self, msg):
        self.latest_odom = msg

    def create_trajectory_frames(self) -> None:
        """
        Creates a gate_passage frame at the target gate frame's position,
        oriented from the robot towards the gate.
        """
        try:
            # Get target gate frame transform
            target_transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.target_gate_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            # Get robot transform
            robot_transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(
                5.0, f"Could not get transforms for gate passage: {e}"
            )
            return

        gate_pos = target_transform.transform.translation
        robot_pos = robot_transform.transform.translation

        # Vector from robot to gate
        dx = gate_pos.x - robot_pos.x
        dy = gate_pos.y - robot_pos.y

        # Orientation from robot to gate
        yaw = math.atan2(dy, dx)
        quat = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)

        # Create pose for gate_passage
        passage_pose = Pose()
        passage_pose.position = gate_pos
        passage_pose.orientation = Quaternion(*quat)

        # Publish transform
        transform = self.build_transform_message(self.gate_passage_frame, passage_pose)
        self.send_transform(transform)

    def build_transform_message(
        self,
        child_frame_id: str,
        pose: Pose,
    ) -> TransformStamped:
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation
        return transform

    def send_transform(self, transform: TransformStamped):
        request = SetObjectTransformRequest()
        request.transform = transform
        try:
            response = self.set_object_transform_service.call(request)
            if not response.success:
                rospy.logerr(
                    f"Failed to set transform for {transform.child_frame_id}: {response.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def handle_enable_service(self, request: SetBool):
        self.is_enabled = request.data
        message = f"Gate trajectory transform publishing is set to: {self.is_enabled}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.create_trajectory_frames()

            rate.sleep()

    def reconfigure_callback(self, config, level):
        self.entrance_offset = config.entrance_offset
        self.exit_offset = config.exit_offset
        self.z_offset = config.z_offset
        self.parallel_shift_offset = config.parallel_shift_offset
        self.rescuer_distance = config.rescuer_distance
        return config


if __name__ == "__main__":
    try:
        node = TransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
