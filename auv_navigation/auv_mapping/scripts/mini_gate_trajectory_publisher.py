#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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
        self.relative_gate_pose = None

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
        self.middle_frame = rospy.get_param("~middle_frame", "mini_gate_middle_part")
        self.center_entrance_frame = rospy.get_param(
            "~center_entrance_frame", "mini_gate_center_entrance"
        )

        self.entrance_offset = 1.0
        self.exit_offset = 1.0
        self.z_offset = 0.5
        self.min_gate_separation_threshold = rospy.get_param(
            "~min_gate_separation_threshold", 0.3
        )

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
        if request.data:
            self.capture_relative_gate_pose()
        else:
            self.relative_gate_pose = None

        message = (
            "Mini gate single-frame trajectory publishing is set to: "
            f"{self.is_enabled}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def create_trajectory_frames(self) -> None:
        t_gate1 = self.lookup_gate_transform(self.gate_frame_1)
        t_gate2 = self.lookup_gate_transform(self.gate_frame_2)

        target_transform = self.select_single_frame_transform(t_gate1, t_gate2)
        if target_transform is None:
            rospy.logwarn(
                "Mini gate trajectory requested, but no gate frame is visible."
            )
            return

        poses = self.compute_single_frame_trajectory(target_transform)
        if poses is None:
            return

        entrance_pose, exit_pose = poses
        target_position = target_transform.transform.translation

        self.publish_pose(
            self.middle_frame,
            Pose(
                position=Point(target_position.x, target_position.y, target_position.z),
                orientation=entrance_pose.orientation,
            ),
        )
        self.publish_pose(self.entrance_frame, entrance_pose)
        self.publish_pose(self.exit_frame, exit_pose)

        relative_gate_pose = self.compute_relative_gate_pose(t_gate1, t_gate2)
        if relative_gate_pose is not None:
            self.publish_pose(self.center_entrance_frame, relative_gate_pose)

    def lookup_gate_transform(self, frame: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
        except tf2_ros.TransformException:
            return None

    def select_single_frame_transform(
        self,
        t_gate1: Optional[TransformStamped],
        t_gate2: Optional[TransformStamped],
    ) -> Optional[TransformStamped]:
        if self.target_gate_frame == self.gate_frame_1 and t_gate1 is not None:
            return t_gate1

        if self.target_gate_frame == self.gate_frame_2 and t_gate2 is not None:
            return t_gate2

        fallback_transform = t_gate1 if t_gate1 is not None else t_gate2
        if fallback_transform is not None:
            fallback_frame = (
                self.gate_frame_1
                if fallback_transform is t_gate1
                else self.gate_frame_2
            )
            rospy.logwarn(
                "Target gate frame '%s' is not visible. Using '%s'.",
                self.target_gate_frame,
                fallback_frame,
            )

        return fallback_transform

    def compute_single_frame_trajectory(
        self, gate_transform: TransformStamped
    ) -> Optional[Tuple[Pose, Pose]]:
        robot_transform = self.lookup_robot_transform()
        if robot_transform is None:
            return None

        gate_pos = gate_transform.transform.translation
        robot_pos = robot_transform.transform.translation
        return self.compute_entrance_exit_from_position(gate_pos, robot_pos)

    def capture_relative_gate_pose(self) -> bool:
        t_gate1 = self.lookup_gate_transform(self.gate_frame_1)
        t_gate2 = self.lookup_gate_transform(self.gate_frame_2)
        robot_transform = self.lookup_robot_transform()
        if t_gate1 is None or t_gate2 is None or robot_transform is None:
            rospy.logwarn(
                "Mini gate relative frame was not captured yet. Waiting for both gate frames and robot TF."
            )
            return False

        geometry = self.compute_gate_pair_geometry(t_gate1, t_gate2)
        if geometry is None:
            return False

        center, unit_gate, unit_normal = geometry
        robot_pos = robot_transform.transform.translation
        offset_x = robot_pos.x - center.x
        offset_y = robot_pos.y - center.y
        self.relative_gate_pose = (
            offset_x * unit_gate[0] + offset_y * unit_gate[1],
            offset_x * unit_normal[0] + offset_y * unit_normal[1],
            robot_pos.z - center.z,
        )
        rospy.loginfo(
            "Captured mini gate relative frame offset: along=%.3f, normal=%.3f, z=%.3f",
            self.relative_gate_pose[0],
            self.relative_gate_pose[1],
            self.relative_gate_pose[2],
        )
        return True

    def compute_relative_gate_pose(
        self,
        t_gate1: Optional[TransformStamped],
        t_gate2: Optional[TransformStamped],
    ) -> Optional[Pose]:
        if t_gate1 is None or t_gate2 is None:
            return None

        if self.relative_gate_pose is None and not self.capture_relative_gate_pose():
            return None

        geometry = self.compute_gate_pair_geometry(t_gate1, t_gate2)
        if geometry is None:
            return None

        center, unit_gate, unit_normal = geometry
        along_offset, normal_offset, z_offset = self.relative_gate_pose
        frame_position = Point(
            center.x + along_offset * unit_gate[0] + normal_offset * unit_normal[0],
            center.y + along_offset * unit_gate[1] + normal_offset * unit_normal[1],
            center.z + z_offset,
        )
        yaw_to_center = math.atan2(
            center.y - frame_position.y,
            center.x - frame_position.x,
        )
        quat = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw_to_center)

        return Pose(position=frame_position, orientation=Quaternion(*quat))

    def compute_gate_pair_geometry(
        self,
        t_gate1: TransformStamped,
        t_gate2: TransformStamped,
    ) -> Optional[Tuple[Point, Tuple[float, float], Tuple[float, float]]]:
        p1 = t_gate1.transform.translation
        p2 = t_gate2.transform.translation
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx**2 + dy**2)
        if length < self.min_gate_separation_threshold:
            rospy.logwarn(
                "Mini gate links are too close for relative frame calculation."
            )
            return None

        center = Point(
            (p1.x + p2.x) / 2.0,
            (p1.y + p2.y) / 2.0,
            (p1.z + p2.z) / 2.0,
        )
        unit_gate = (dx / length, dy / length)
        unit_normal = (-unit_gate[1], unit_gate[0])
        return center, unit_gate, unit_normal

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

    def spin(self) -> None:
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.create_trajectory_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = MiniGateTrajectoryPublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
