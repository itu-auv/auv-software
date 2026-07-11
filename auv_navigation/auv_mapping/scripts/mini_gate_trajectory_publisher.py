#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
from typing import Optional, Tuple

import rospy
import tf2_ros
import tf_conversions
from auv_mapping.cfg import GateTrajectoryConfig
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from dynamic_reconfigure.client import Client as DynamicReconfigureClient
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse


ROLE_TO_GATE_FRAME = {
    "survey_repair": "gate_survey_repair_link",
    "search_rescue": "gate_search_rescue_link",
}


class MiniGateTrajectoryPublisher:
    def __init__(self):
        self.is_enabled = False
        self.publish_lock = threading.Lock()

        rospy.init_node("mini_gate_trajectory_publisher")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_base_frame = rospy.get_param(
            "~robot_base_frame", "taluy_mini/base_link"
        )
        self.gate_frame_1 = rospy.get_param("~gate_frame_1", "gate_survey_repair_link")
        self.gate_frame_2 = rospy.get_param("~gate_frame_2", "gate_search_rescue_link")
        self.target_gate_frame = self.gate_frame_1

        self.entrance_frame = rospy.get_param("~entrance_frame", "mini_gate_entrance")
        self.exit_frame = rospy.get_param("~exit_frame", "mini_gate_exit")

        self.entrance_offset = 1.0
        self.exit_offset = 1.0
        self.z_offset = 0.5
        self.min_gate_separation_threshold = rospy.get_param(
            "~min_gate_separation_threshold", 0.3
        )
        self.gate_lookup_timeout = rospy.get_param("~gate_lookup_timeout", 0.1)

        self.reconfigure_server = Server(
            GateTrajectoryConfig, self.reconfigure_callback
        )
        self.smach_params_client = DynamicReconfigureClient(
            "smach_parameters_server",
            timeout=10,
            config_callback=self.smach_params_callback,
        )

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.set_enable_service = rospy.Service(
            "toggle_mini_gate_trajectory", SetBool, self.handle_enable_service
        )

    def smach_params_callback(self, config):
        if "selected_role" in config:
            self.set_target_gate_frame(config["selected_role"])

    def reconfigure_callback(self, config, level):
        self.entrance_offset = config.entrance_offset
        self.exit_offset = config.exit_offset
        self.z_offset = config.z_offset
        if self.is_enabled and hasattr(self, "set_object_transform_service"):
            self.publish_current_trajectory()
        return config

    def set_target_gate_frame(self, selected_role):
        target_gate_frame = ROLE_TO_GATE_FRAME.get(selected_role)
        if target_gate_frame is None:
            rospy.logwarn(
                "Unknown selected role '%s'. Keeping target gate frame: %s",
                selected_role,
                self.target_gate_frame,
            )
            return

        self.target_gate_frame = target_gate_frame

    def handle_enable_service(self, request: SetBool) -> SetBoolResponse:
        self.is_enabled = request.data
        if self.is_enabled:
            self.publish_current_trajectory()

        message = (
            "Mini gate single-frame trajectory publishing is set to: "
            f"{self.is_enabled}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def create_trajectory_frames(self) -> None:
        target_transform = self.lookup_selected_gate_transform()
        if target_transform is None:
            rospy.logwarn(
                "Mini gate trajectory requested, but no gate frame is visible."
            )
            return

        poses = self.compute_single_frame_trajectory(target_transform)
        if poses is None:
            return

        entrance_pose, exit_pose = poses

        self.publish_pose(self.entrance_frame, entrance_pose)
        self.publish_pose(self.exit_frame, exit_pose)

    def lookup_gate_transform(self, frame: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                frame,
                rospy.Time(0),
                rospy.Duration(self.gate_lookup_timeout),
            )
        except tf2_ros.TransformException:
            return None

    def lookup_selected_gate_transform(self) -> Optional[TransformStamped]:
        target_transform = self.lookup_gate_transform(self.target_gate_frame)
        if target_transform is not None:
            return target_transform

        fallback_frame = self.get_fallback_gate_frame()
        fallback_transform = self.lookup_gate_transform(fallback_frame)
        if fallback_transform is not None:
            rospy.logwarn_throttle(
                5.0,
                "Target gate frame '%s' is not visible. Using '%s'.",
                self.target_gate_frame,
                fallback_frame,
            )

        return fallback_transform

    def get_fallback_gate_frame(self) -> str:
        if self.target_gate_frame == self.gate_frame_1:
            return self.gate_frame_2
        return self.gate_frame_1

    def compute_single_frame_trajectory(
        self, gate_transform: TransformStamped
    ) -> Optional[Tuple[Pose, Pose]]:
        robot_transform = self.lookup_robot_transform()
        if robot_transform is None:
            return None

        gate_pos = gate_transform.transform.translation
        robot_pos = robot_transform.transform.translation
        return self.compute_entrance_exit_from_position(gate_pos, robot_pos)

    def lookup_robot_transform(self) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(4.0),
            )
        except tf2_ros.TransformException as e:
            rospy.logwarn(
                "Mini gate trajectory failed: could not get robot transform '%s': %s",
                self.robot_base_frame,
                e,
            )
            return None

    def compute_entrance_exit_from_position(
        self,
        gate_pos: Point,
        robot_pos: Point,
    ) -> Optional[Tuple[Pose, Pose]]:
        dx = gate_pos.x - robot_pos.x
        dy = gate_pos.y - robot_pos.y
        length = math.sqrt(dx**2 + dy**2)
        if length < self.min_gate_separation_threshold:
            rospy.logwarn(
                "Robot is too close to the mini gate for trajectory calculation."
            )
            return None

        unit_dx = dx / length
        unit_dy = dy / length
        common_yaw = math.atan2(dy, dx)
        common_quat = tf_conversions.transformations.quaternion_from_euler(
            0, 0, common_yaw
        )

        entrance_pose = Pose(
            position=Point(
                gate_pos.x - unit_dx * self.entrance_offset,
                gate_pos.y - unit_dy * self.entrance_offset,
                gate_pos.z - self.z_offset,
            ),
            orientation=Quaternion(*common_quat),
        )
        exit_pose = Pose(
            position=Point(
                gate_pos.x + unit_dx * self.exit_offset,
                gate_pos.y + unit_dy * self.exit_offset,
                gate_pos.z - self.z_offset,
            ),
            orientation=Quaternion(*common_quat),
        )

        return entrance_pose, exit_pose

    def publish_pose(self, child_frame_id: str, pose: Pose) -> None:
        self.send_transform(self.build_transform_message(child_frame_id, pose))

    def build_transform_message(
        self,
        child_frame_id: str,
        pose: Pose,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation
        return transform

    def send_transform(self, transform: TransformStamped) -> None:
        request = SetObjectTransformRequest()
        request.transform = transform
        try:
            response = self.set_object_transform_service.call(request)
            if not response.success:
                rospy.logerr(
                    "Failed to set transform for %s: %s",
                    transform.child_frame_id,
                    response.message,
                )
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def publish_current_trajectory(self) -> None:
        with self.publish_lock:
            self.create_trajectory_frames()

    def spin(self) -> None:
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.publish_current_trajectory()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = MiniGateTrajectoryPublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
