#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional
import math
import rclpy
from auv_common_lib.ros import get_param, service_client
from auv_common_lib import ros
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import threading
import tf2_ros
import tf_transformations
import geometry_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, Trigger
from auv_msgs.srv import SetObjectTransform
from nav_msgs.msg import Odometry
from auv_common_lib.reconfigure import Server
from auv_common_lib.reconfigure import Client as DynamicReconfigureClient

# name, default, description, min, max
GateTrajectoryConfig = [
    (
        "selected_role",
        "survey_repair",
        "Selected gate role: survey_repair or search_rescue",
    ),
    ("entrance_offset", 1.5, "Entrance offset (m)", 0, 5),
    ("exit_offset", 0.5, "Exit offset (m)", 0, 10),
    ("z_offset", 0.5, "Z offset (m)", -2, 2),
    ("parallel_shift_offset", 0.0, "Parallel shift offset (m)", -1, 1),
    ("rescuer_distance", 1.0, "Rescuer distance (m)", 0, 5),
]


ROLE_TO_GATE_FRAME = {
    "survey_repair": "gate_survey_repair_link",
    "search_rescue": "gate_search_rescue_link",
}


class TransformServiceNode(Node):
    def __init__(self):
        self.is_enabled = False
        self.gate_angle = None
        self.publish_gate_angle_enabled = False
        super().__init__("create_gate_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Dynamic reconfigure server for gate parameters
        self.gate_frame_1 = "gate_survey_repair_link"
        self.gate_frame_2 = "gate_search_rescue_link"
        self.target_gate_frame = self.gate_frame_1
        self.entrance_offset = 1.0
        self.exit_offset = 1.0
        self.z_offset = 0.5
        self.parallel_shift_offset = 0.20
        self.rescuer_distance = 1.0
        self.wall_reference_yaw = 0.0
        self.reconfigure_server = Server(
            self, GateTrajectoryConfig, self.reconfigure_callback
        )

        # Client for auv_smach parameters
        try:
            self.smach_params_client = DynamicReconfigureClient(
                self,
                "smach_parameters_server",
                timeout=10,
                config_callback=self.smach_params_callback,
            )
        except TimeoutError:
            self.get_logger().error("Smach dynamic reconfigure client not started.")

        # Service to broadcast transforms
        self.set_object_transform_service = service_client(
            self, SetObjectTransform, "set_object_transform"
        )
        self.set_object_transform_service.wait_for_service()
        self.gate_angle_publisher = self.create_publisher(Float64, "gate_angle", 10)
        self.publish_gate_angle_service = self.create_service(
            Trigger, "publish_gate_angle", self.handle_publish_gate_angle
        )

        self.odom_frame = "odom"
        self.robot_base_frame = get_param(self, "~robot_base_frame", "taluy/base_link")
        self.entrance_frame = "gate_entrance"
        self.exit_frame = "gate_exit"
        self.middle_frame = "gate_middle_part"

        # Verify that we have valid frame names
        if not all([self.gate_frame_1, self.gate_frame_2]):
            self.get_logger().error("Missing required gate frame parameters")
            raise SystemExit("Missing required parameters")

        self.set_enable_service = self.create_service(
            SetBool, "toggle_gate_trajectory", self.handle_enable_service
        )

        # --- Parameters for fallback (single-frame) mode
        self.fallback_entrance_offset = get_param(
            self, "~fallback_entrance_offset", 1.0
        )
        self.fallback_exit_offset = get_param(self, "~fallback_exit_offset", 1.0)
        self.MIN_GATE_SEPARATION = get_param(self, "~min_gate_separation", 0.2)
        self.MAX_GATE_SEPARATION = get_param(self, "~max_gate_separation", 2.5)
        self.MIN_GATE_SEPARATION_THRESHOLD = 0.3

        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self.odom_callback, 10
        )
        self.latest_odom = None

        self.coin_flip_enabled = False
        self.coin_flip_rescuer_pose = None
        self.rescuer_frame_name = "coin_flip_rescuer"

        self.toggle_rescuer_service = self.create_service(
            SetBool, "toggle_coin_flip_rescuer", self.handle_toggle_rescuer_service
        )

    def smach_params_callback(self, config):
        """Callback for receiving parameters from the auv_smach node."""
        self.get_logger().info("Received smach parameters update: %s" % (config,))
        if "wall_reference_yaw" in config:
            self.wall_reference_yaw = config["wall_reference_yaw"]
        if "selected_role" in config:
            self.set_target_gate_frame(config["selected_role"])

    def set_target_gate_frame(self, selected_role):
        target_gate_frame = ROLE_TO_GATE_FRAME.get(selected_role)
        if target_gate_frame is None:
            self.get_logger().warning(
                "Unknown selected role '%s'. Keeping target gate frame: %s"
                % (selected_role, self.target_gate_frame)
            )
            return

        self.target_gate_frame = target_gate_frame

    def handle_toggle_rescuer_service(
        self, request: SetBool, response
    ) -> SetBool.Response:

        self.coin_flip_enabled = request.data
        message = f"Coin flip rescuer frame publishing set to: {self.coin_flip_enabled}"
        self.get_logger().info(message)

        if request.data:
            self._publish_rescuer_frame()

        if self.coin_flip_enabled and self.coin_flip_rescuer_pose is None:
            try:
                initial_transform = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.robot_base_frame,
                    Time(),
                    Duration(seconds=4.0),
                )

                initial_pos = initial_transform.transform.translation

                rescuer_pose = Pose()
                rescuer_pose.position.x = initial_pos.x + self.rescuer_distance
                rescuer_pose.position.y = initial_pos.y
                rescuer_pose.position.z = initial_pos.z

                q = tf_transformations.quaternion_from_euler(0, 0, 0)
                rescuer_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

                self.coin_flip_rescuer_pose = rescuer_pose
                return SetBool.Response(success=True, message=message)

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                error_message = f"Failed to get initial vehicle pose to calculate rescuer frame: {e}"
                self.get_logger().error(error_message)
                self.coin_flip_enabled = False
                return SetBool.Response(success=False, message=error_message)

        return SetBool.Response(success=True, message=message)

    def odom_callback(self, msg):
        self.latest_odom = msg

    def _publish_rescuer_frame(self):
        if not self.coin_flip_enabled:
            self.get_logger().debug("Coin flip rescuer frame publishing is disabled.")
            return
        if self.latest_odom is None:
            self.get_logger().warning(
                "Coin flip rescuer frame odometry message not received."
            )
            return

        pos = self.latest_odom.pose.pose.position
        reference_yaw = self.wall_reference_yaw
        dx = self.rescuer_distance * math.cos(reference_yaw)
        dy = self.rescuer_distance * math.sin(reference_yaw)
        rescuer_x = pos.x + dx
        rescuer_y = pos.y + dy
        rescuer_z = pos.z
        rescuer_quat = tf_transformations.quaternion_from_euler(0, 0, reference_yaw)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.rescuer_frame_name
        transform.transform.translation.x = rescuer_x
        transform.transform.translation.y = rescuer_y
        transform.transform.translation.z = rescuer_z
        transform.transform.rotation.x = rescuer_quat[0]
        transform.transform.rotation.y = rescuer_quat[1]
        transform.transform.rotation.z = rescuer_quat[2]
        transform.transform.rotation.w = rescuer_quat[3]
        self.send_transform(transform)

    def create_trajectory_frames(self) -> None:
        """
        Main logic to decide which method to use for creating gate frames.
        It tries to use two frames, but switches to a fallback method if needed.
        """
        # --- Try to get transforms for both gate frames
        try:
            t_gate1 = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.gate_frame_1,
                self.get_clock().now(),
                Duration(seconds=4.0),
            )
        except tf2_ros.TransformException:
            t_gate1 = None

        try:
            t_gate2 = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.gate_frame_2,
                self.get_clock().now(),
                Duration(seconds=4.0),
            )
        except tf2_ros.TransformException:
            t_gate2 = None

        poses = None
        # --- Decision logic: Check which mode to use
        if t_gate1 and t_gate2:
            # Both frames are visible
            p1 = t_gate1.transform.translation
            p2 = t_gate2.transform.translation
            distance = math.sqrt(
                (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
            )

            # --- Create gate_middle_part_link at the midpoint between the two gate role links
            middle_x = (p1.x + p2.x) / 2.0
            middle_y = (p1.y + p2.y) / 2.0
            middle_z = (p1.z + p2.z) / 2.0
            # Orientation: same as odom frame (no rotation)
            identity_quat = tf_transformations.quaternion_from_euler(0, 0, 0)
            middle_pose = Pose(
                position=Point(x=middle_x, y=middle_y, z=middle_z),
                orientation=Quaternion(
                    x=identity_quat[0],
                    y=identity_quat[1],
                    z=identity_quat[2],
                    w=identity_quat[3],
                ),
            )
            middle_transform = self.build_transform_message(
                self.middle_frame, middle_pose
            )
            self.send_transform(middle_transform)

            if self.MIN_GATE_SEPARATION < distance < self.MAX_GATE_SEPARATION:
                # Distance is valid, use standard dual-frame method
                self.get_logger().info(
                    "Using dual-frame mode for gate trajectory.", once=True
                )
                poses = self._compute_frames_dual_mode(
                    self.get_translation(t_gate1), self.get_translation(t_gate2)
                )
            else:
                # Distance is out of bounds, use fallback method with the selected target
                self.get_logger().warning(
                    f"Gate distance ({distance:.2f}m) is out of bounds. Using fallback mode.",
                    once=True,
                )
                target_transform = (
                    t_gate1 if self.target_gate_frame == self.gate_frame_1 else t_gate2
                )
                poses = self._compute_frames_fallback_mode(target_transform)

        elif t_gate1 or t_gate2:
            # Only one frame is visible, use fallback method
            self.get_logger().warning(
                "Only one gate frame is visible. Using fallback mode.", once=True
            )
            visible_transform = t_gate1 if t_gate1 else t_gate2
            # --- Place gate_middle_part at the visible frame's position
            p = visible_transform.transform.translation
            q = visible_transform.transform.rotation
            middle_pose = Pose(
                position=Point(x=p.x, y=p.y, z=p.z),
                orientation=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w),
            )
            middle_transform = self.build_transform_message(
                self.middle_frame, middle_pose
            )
            self.send_transform(middle_transform)
            poses = self._compute_frames_fallback_mode(visible_transform)

        else:
            # No frames are visible
            self.get_logger().warning(
                "Neither gate frame is visible. Cannot create trajectory."
            )
            return

        if poses:
            entrance_pose, exit_pose = poses
            # Create and send transforms
            entrance_transform = self.build_transform_message(
                self.entrance_frame, entrance_pose
            )
            exit_transform = self.build_transform_message(self.exit_frame, exit_pose)
            self.send_transform(entrance_transform)
            self.send_transform(exit_transform)

    def _compute_frames_fallback_mode(
        self, gate_transform: TransformStamped
    ) -> Optional[Tuple[Pose, Pose]]:
        """
        Fallback method: Creates entrance/exit frames on a line from the robot to a single gate frame.
        """
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                Time(),
                Duration(seconds=4.0),
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warning(
                f"Fallback mode failed: Could not get robot transform '{self.robot_base_frame}': {e}"
            )
            return None

        gate_pos = gate_transform.transform.translation
        robot_pos = robot_transform.transform.translation

        # Vector from robot to gate
        dx = gate_pos.x - robot_pos.x
        dy = gate_pos.y - robot_pos.y

        length = math.sqrt(dx**2 + dy**2)
        if length < self.MIN_GATE_SEPARATION_THRESHOLD:
            self.get_logger().warning(
                "Robot is too close to the gate frame for fallback calculation."
            )
            return None

        # Unit vector for the direction
        unit_dx = dx / length
        unit_dy = dy / length

        # Position entrance before the gate, exit after the gate
        entrance_position = Point(
            x=gate_pos.x - unit_dx * self.fallback_entrance_offset,
            y=gate_pos.y - unit_dy * self.fallback_entrance_offset,
            z=gate_pos.z - self.z_offset,  # Apply Z-offset
        )
        exit_position = Point(
            x=gate_pos.x + unit_dx * self.fallback_exit_offset,
            y=gate_pos.y + unit_dy * self.fallback_exit_offset,
            z=gate_pos.z - self.z_offset,  # Apply Z-offset
        )

        # Orientation should look along the line of approach
        common_yaw = math.atan2(dy, dx)
        common_quat = tf_transformations.quaternion_from_euler(0, 0, common_yaw)

        entrance_pose = Pose(
            position=entrance_position,
            orientation=Quaternion(
                x=common_quat[0], y=common_quat[1], z=common_quat[2], w=common_quat[3]
            ),
        )
        exit_pose = Pose(
            position=exit_position,
            orientation=Quaternion(
                x=common_quat[0], y=common_quat[1], z=common_quat[2], w=common_quat[3]
            ),
        )

        return entrance_pose, exit_pose

    def _compute_frames_dual_mode(
        self,
        gate_link_1_translation: Tuple[float, float, float],
        gate_link_2_translation: Tuple[float, float, float],
    ) -> Optional[Tuple[Pose, Pose]]:
        """
        Original method: Computes entrance/exit frames based on two gate posts.
        """
        # Assign selected and other gate link translations
        selected_gate_link_translation, other_gate_link_translation = (
            self.assign_selected_gate_translations(
                gate_link_1_translation, gate_link_2_translation
            )
        )

        # Calculate and store the gate angle
        dx = other_gate_link_translation[0] - selected_gate_link_translation[0]
        dy = other_gate_link_translation[1] - selected_gate_link_translation[1]
        self.gate_angle = math.atan2(dy, dx)

        # Compute the entrance and exit frames relative to selected gate frame
        try:
            entrance_pose, exit_pose = self.compute_entrance_and_exit(
                selected_gate_link_translation, other_gate_link_translation
            )
        except ValueError as e:
            self.get_logger().warning(f"Could not compute dual-mode frames: {e}")
            return None

        # Create TransformStamped messages for entrance and exit
        entrance_transform = self.build_transform_message(
            self.entrance_frame, entrance_pose
        )
        exit_transform = self.build_transform_message(self.exit_frame, exit_pose)

        # Apply parallel shift if offset is significant
        if abs(self.parallel_shift_offset) > 1e-6:
            other_gate_frame = (
                self.gate_frame_2
                if self.target_gate_frame == self.gate_frame_1
                else self.gate_frame_1
            )

            entrance_transform = self._shift_transform_parallel_to_gate_line(
                entrance_transform,
                self.target_gate_frame,
                other_gate_frame,
                self.parallel_shift_offset,
            )
            exit_transform = self._shift_transform_parallel_to_gate_line(
                exit_transform,
                self.target_gate_frame,
                other_gate_frame,
                self.parallel_shift_offset,
            )

        # Extract final poses from potentially shifted transforms
        final_entrance_pose = Pose(
            position=entrance_transform.transform.translation,
            orientation=entrance_transform.transform.rotation,
        )
        final_exit_pose = Pose(
            position=exit_transform.transform.translation,
            orientation=exit_transform.transform.rotation,
        )

        return final_entrance_pose, final_exit_pose

    def assign_selected_gate_translations(
        self,
        gate_link_1_translation: Tuple[float, float, float],
        gate_link_2_translation: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Determine selected and other gate link translations based on target frame."""
        if self.target_gate_frame == self.gate_frame_1:
            return gate_link_1_translation, gate_link_2_translation
        elif self.target_gate_frame == self.gate_frame_2:
            return gate_link_2_translation, gate_link_1_translation
        else:
            raise ValueError("Invalid selected gate frame")

    def compute_entrance_and_exit(
        self,
        selected_gate_link_translation: Tuple[float, float, float],
        other_gate_link_translation: Tuple[float, float, float],
    ) -> Tuple[Pose, Pose]:
        """Calculate entrance and exit poses based on gate frames."""

        dx = other_gate_link_translation[0] - selected_gate_link_translation[0]
        dy = other_gate_link_translation[1] - selected_gate_link_translation[1]
        length = math.sqrt(dx**2 + dy**2)

        if length < self.MIN_GATE_SEPARATION_THRESHOLD:
            raise ValueError("The gate links are too close to each other.")

        # Get robot's current position
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                Time(),
                Duration(seconds=4.0),
            )
            robot_pos = robot_transform.transform.translation
        except tf2_ros.TransformException as e:
            self.get_logger().warning(
                f"Could not get robot transform for entrance/exit calculation: {e}"
            )
            raise ValueError(
                "Failed to get robot position for entrance/exit calculation"
            )

        unit_perpendicular_x = -dy / length
        unit_perpendicular_y = dx / length
        robot_dx = robot_pos.x - selected_gate_link_translation[0]
        robot_dy = robot_pos.y - selected_gate_link_translation[1]

        # Dot product to determine which side of perpendicular line the robot is on
        dot_product = robot_dx * unit_perpendicular_x + robot_dy * unit_perpendicular_y

        entrance_direction = 1.0 if dot_product > 0 else -1.0
        exit_direction = -1.0 * entrance_direction

        # Calculate new positions relative to selected_translation
        entrance_position = (
            selected_gate_link_translation[0]
            + entrance_direction * unit_perpendicular_x * self.entrance_offset,
            selected_gate_link_translation[1]
            + entrance_direction * unit_perpendicular_y * self.entrance_offset,
            selected_gate_link_translation[2] - self.z_offset,
        )
        exit_position = (
            selected_gate_link_translation[0]
            + exit_direction * unit_perpendicular_x * self.exit_offset,
            selected_gate_link_translation[1]
            + exit_direction * unit_perpendicular_y * self.exit_offset,
            selected_gate_link_translation[2] - self.z_offset,
        )

        # Calculate orientations - entrance looks toward exit (perpendicular to gate line)
        # Both frames have same orientation, perpendicular to the gate line
        common_yaw = math.atan2(
            exit_position[1] - entrance_position[1],
            exit_position[0] - entrance_position[0],
        )
        common_quat = tf_transformations.quaternion_from_euler(0, 0, common_yaw)

        entrance_pose = Pose(
            position=Point(
                x=entrance_position[0], y=entrance_position[1], z=entrance_position[2]
            ),
            orientation=Quaternion(
                x=common_quat[0], y=common_quat[1], z=common_quat[2], w=common_quat[3]
            ),
        )
        exit_pose = Pose(
            position=Point(x=exit_position[0], y=exit_position[1], z=exit_position[2]),
            orientation=Quaternion(
                x=common_quat[0], y=common_quat[1], z=common_quat[2], w=common_quat[3]
            ),
        )

        return entrance_pose, exit_pose

    def _shift_transform_parallel_to_gate_line(
        self,
        transform_to_shift: TransformStamped,
        selected_gate_frame_name: str,
        other_gate_frame_name: str,
        parallel_offset: float,
    ) -> TransformStamped:
        """Shift transform parallel to the line connecting the two gate frames."""
        try:
            # Transform from selected to other, gives the direction vector in the selected frame's coordinate system
            transform_selected_to_other = self.tf_buffer.lookup_transform(
                selected_gate_frame_name,
                other_gate_frame_name,
                self.get_clock().now(),
                Duration(seconds=4.0),
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warning(f"Parallel shift failed due to TF Error: {e}")
            return transform_to_shift

        dx_sel = transform_selected_to_other.transform.translation.x
        dy_sel = transform_selected_to_other.transform.translation.y
        length = math.sqrt(dx_sel**2 + dy_sel**2)

        if length < self.MIN_GATE_SEPARATION_THRESHOLD:
            self.get_logger().warning(
                "Gate links too close for parallel shift. Skipping."
            )
            return transform_to_shift

        # Shift is applied in the odom frame, so we need the odom-frame direction vector
        # (calculated from the dual-mode pose computation)
        angle = self.gate_angle  # Angle of the gate line in the odom frame
        unit_dx_odom = math.cos(angle)
        unit_dy_odom = math.sin(angle)

        shift_x = unit_dx_odom * parallel_offset
        shift_y = unit_dy_odom * parallel_offset

        transform_to_shift.transform.translation.x += shift_x
        transform_to_shift.transform.translation.y += shift_y
        return transform_to_shift

    def get_translation(
        self, transform: TransformStamped
    ) -> Tuple[float, float, float]:
        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        )

    def build_transform_message(
        self,
        child_frame_id: str,
        pose: Pose,
    ) -> TransformStamped:
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation
        return transform

    def send_transform(self, transform: TransformStamped):
        request = SetObjectTransform.Request()
        request.transform = transform
        try:
            response = self.set_object_transform_service.call(request)
            if not response.success:
                self.get_logger().error(
                    f"Failed to set transform for {transform.child_frame_id}: {response.message}"
                )
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def handle_enable_service(self, request: SetBool, response):
        self.is_enabled = request.data
        message = f"Gate trajectory transform publishing is set to: {self.is_enabled}"
        self.get_logger().info(message)
        return SetBool.Response(success=True, message=message)

    def handle_publish_gate_angle(self, req, response):
        self.publish_gate_angle_enabled = True
        message = "Gate angle publishing enabled."
        self.get_logger().info(message)
        return Trigger.Response(success=True, message=message)

    def _publish_gate_angle_loop(self):
        rate = self.create_rate(10.0)
        while rclpy.ok():
            if self.publish_gate_angle_enabled and self.gate_angle is not None:
                self.gate_angle_publisher.publish(self.gate_angle)
            rate.sleep()

    def spin(self):
        gate_angle_thread = threading.Thread(target=self._publish_gate_angle_loop)
        gate_angle_thread.daemon = True
        gate_angle_thread.start()

        rate = self.create_rate(2.0)
        while rclpy.ok():
            if self.is_enabled:
                self.create_trajectory_frames()

            if self.coin_flip_enabled:
                self._publish_rescuer_frame()

            rate.sleep()

    def reconfigure_callback(self, config, level):
        if "selected_role" in config:
            self.set_target_gate_frame(config["selected_role"])
        self.entrance_offset = config.entrance_offset
        self.exit_offset = config.exit_offset
        self.z_offset = config.z_offset
        self.parallel_shift_offset = config.parallel_shift_offset
        self.rescuer_distance = config.rescuer_distance
        return config


def main():
    ros.init()
    node = TransformServiceNode()
    ros.spin(node, node.spin)


if __name__ == "__main__":
    main()
