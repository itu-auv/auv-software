#!/usr/bin/env python3

import numpy as np
import rclpy
from auv_common_lib.ros import get_param, service_client, Rate
from auv_common_lib import ros
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
import tf_transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool
from auv_msgs.srv import SetObjectTransform


class OctagonTransformServiceNode(Node):
    def __init__(self):
        self.enable = False
        super().__init__("create_octagon_frame_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.set_object_transform_service = service_client(
            self, SetObjectTransform, "set_object_transform"
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.robot_frame = get_param(self, "~robot_frame", "taluy/base_link")
        self.octagon_frame = "octagon_link_kde"
        self.middle_basket_frame = get_param(
            self, "~middle_basket_frame", "middle_basket"
        )
        self.redcross_basket_segment_frame = get_param(
            self, "~basket_redcross_segment_frame", "basket_redcross_segment_link"
        )
        self.warning_basket_segment_frame = get_param(
            self, "~basket_warning_segment_frame", "basket_warning_segment_link"
        )
        self.octagon_table_segment_frame = get_param(
            self, "~octagon_table_segment_frame", "octagon_table_segment_link"
        )

        self.octagon_closer_frame = get_param(
            self, "~octagon_closer_frame", "octagon_closer_link"
        )
        self.octagon_further_frame = get_param(
            self, "~octagon_further_frame", "octagon_further_link"
        )
        self.closer_distance = get_param(
            self, "~closer_distance", 2.0
        )  # distance from octagon to closer frame
        self.further_distance = get_param(
            self, "~further_distance", 2.0
        )  # distance from octagon to further frame
        self.search_distance = get_param(
            self, "~search_distance", 0.3
        )  # distance from octagon to search frames
        self.search_frames = {
            "octagon_search_forward": "forward",  # towards robot (back from octagon)
            "octagon_search_backward": "backward",  # away from robot
            "octagon_search_left": "left",  # perpendicular left
            "octagon_search_right": "right",  # perpendicular right
        }

        self.set_enable_service = self.create_service(
            SetBool, "set_transform_octagon_frame", self.handle_enable_service
        )

    def get_pose(self, transform: TransformStamped) -> Pose:
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform):
        req = SetObjectTransform.Request()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            self.get_logger().warning(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def lookup_pose(self, frame_name: str):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame, frame_name, Time(), Duration(seconds=0.5)
            )
            return self.get_pose(transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

    def build_pose_from_position_and_yaw(
        self, position: np.ndarray, yaw: float
    ) -> Pose:
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose

    def create_middle_basket_frame(self):
        redcross_pose = self.lookup_pose(self.redcross_basket_segment_frame)
        warning_pose = self.lookup_pose(self.warning_basket_segment_frame)

        if redcross_pose is not None and warning_pose is not None:
            redcross_pos = np.array(
                [
                    redcross_pose.position.x,
                    redcross_pose.position.y,
                    redcross_pose.position.z,
                ]
            )
            warning_pos = np.array(
                [
                    warning_pose.position.x,
                    warning_pose.position.y,
                    warning_pose.position.z,
                ]
            )
            red_to_warning = warning_pos[:2] - redcross_pos[:2]
            red_to_warning_norm = np.linalg.norm(red_to_warning)

            if red_to_warning_norm > 1e-6:
                midpoint = 0.5 * (redcross_pos + warning_pos)
                left_direction = (
                    np.array([-red_to_warning[1], red_to_warning[0]])
                    / red_to_warning_norm
                )
                yaw = np.arctan2(left_direction[1], left_direction[0])
                middle_basket_pose = self.build_pose_from_position_and_yaw(
                    midpoint, yaw
                )
                middle_basket_transform = self.build_transform_message(
                    self.middle_basket_frame, middle_basket_pose
                )
                self.send_transform(middle_basket_transform)
                return

            self.get_logger().warning(
                "basket_redcross_segment_link and basket_warning_segment_link overlap; "
                "falling back to octagon_table_segment_link for middle_basket.",
                throttle_duration_sec=5.0,
            )

        table_pose = self.lookup_pose(self.octagon_table_segment_frame)
        if table_pose is None:
            self.get_logger().warning(
                "middle_basket could not be created: octagon_table_segment_link not available yet.",
                throttle_duration_sec=5.0,
            )
            return

        middle_basket_transform = self.build_transform_message(
            self.middle_basket_frame, table_pose
        )
        self.send_transform(middle_basket_transform)

    def create_octagon_frame(self):
        if not self.enable:
            return

        try:
            transform_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, Time(), Duration(seconds=4.0)
            )
            transform_octagon = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.octagon_frame,
                self.get_clock().now(),
                Duration(seconds=4.0),
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warning(f"TF lookup failed: {e}")
            return

        robot_pose = self.get_pose(transform_robot)
        octagon_pose = self.get_pose(transform_octagon)

        robot_pos = np.array(
            [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
        )
        octagon_pos = np.array(
            [octagon_pose.position.x, octagon_pose.position.y, octagon_pose.position.z]
        )

        direction_vector_2d = octagon_pos[:2] - robot_pos[:2]
        total_distance_2d = np.linalg.norm(direction_vector_2d)

        if total_distance_2d == 0:
            self.get_logger().warning(
                "Robot and octagon are at the same XY position! Cannot create frame."
            )
            return

        direction_unit_2d = direction_vector_2d / total_distance_2d

        # Calculate yaw from the direction vector (robot to octagon)
        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)

        orientation = Pose().orientation
        orientation.x = q[0]
        orientation.y = q[1]
        orientation.z = q[2]
        orientation.w = q[3]

        # Calculate position for closer frame
        closer_pos_2d = octagon_pos[:2] - (direction_unit_2d * self.closer_distance)
        closer_pos = np.append(closer_pos_2d, robot_pos[2])
        further_pos_2d = octagon_pos[:2] + (direction_unit_2d * self.further_distance)
        further_pos = np.append(further_pos_2d, robot_pos[2])

        # Create closer frame
        closer_pose = Pose()
        closer_pose.position.x, closer_pose.position.y, closer_pose.position.z = (
            closer_pos
        )
        closer_pose.orientation = orientation
        further_pose = Pose()
        further_pose.position.x, further_pose.position.y, further_pose.position.z = (
            further_pos
        )
        further_pose.orientation = orientation

        # Send the transform
        closer_transform = self.build_transform_message(
            self.octagon_closer_frame, closer_pose
        )
        self.send_transform(closer_transform)
        further_transform = self.build_transform_message(
            self.octagon_further_frame, further_pose
        )
        self.send_transform(further_transform)

        self.create_search_frames(octagon_pos, direction_unit_2d, robot_pos[2])

    def create_search_frames(self, octagon_pos, direction_unit_2d, z_height):
        """
        Create 4 search frames in a + pattern around octagon.
        - forward: away from robot (further from octagon perspective)
        - backward: towards robot direction
        - left: perpendicular left
        - right: perpendicular right

        All frames have the same orientation as closer_frame (facing robot->octagon direction).
        """
        perpendicular_unit_2d = np.array([-direction_unit_2d[1], direction_unit_2d[0]])

        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)

        for frame_name, direction_type in self.search_frames.items():
            if direction_type == "forward":
                offset_2d = direction_unit_2d * self.search_distance
            elif direction_type == "backward":
                offset_2d = -direction_unit_2d * self.search_distance
            elif direction_type == "left":
                offset_2d = perpendicular_unit_2d * self.search_distance
            elif direction_type == "right":
                offset_2d = -perpendicular_unit_2d * self.search_distance
            else:
                continue

            frame_pos_2d = octagon_pos[:2] + offset_2d
            frame_pos = np.array([frame_pos_2d[0], frame_pos_2d[1], z_height])

            search_pose = Pose()
            search_pose.position.x = frame_pos[0]
            search_pose.position.y = frame_pos[1]
            search_pose.position.z = frame_pos[2]
            search_pose.orientation.x = q[0]
            search_pose.orientation.y = q[1]
            search_pose.orientation.z = q[2]
            search_pose.orientation.w = q[3]

            search_transform = self.build_transform_message(frame_name, search_pose)
            self.send_transform(search_transform)

    def handle_enable_service(self, req: SetBool, response):
        self.enable = req.data
        self.get_logger().info(
            f"Octagon frame publishing {'enabled' if self.enable else 'disabled'}"
        )
        return SetBool.Response(success=True)

    def run(self):
        rate = Rate(self, 10)  # 10 Hz
        while rclpy.ok():
            if self.enable:
                self.create_middle_basket_frame()
            self.create_octagon_frame()
            rate.sleep()


def main():
    ros.init()
    node = OctagonTransformServiceNode()
    ros.spin(node, node.run)


if __name__ == "__main__":
    main()
