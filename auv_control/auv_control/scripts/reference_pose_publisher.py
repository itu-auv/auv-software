#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from auv_common_lib import ros
from rclpy.time import Time
from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
from auv_control.control_clients import ControlClients, ControllerParameters, log, log_throttle, parameter, init_ros
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    PoseStamped,
    Twist,
    PoseWithCovarianceStamped,
    PointStamped,
    TransformStamped,
)
from std_srvs.srv import Trigger
TriggerResponse = Trigger.Response
from std_msgs.msg import Bool
from auv_msgs.srv import (
    SetDepth,
    AlignFrameController,
    SetObjectTransform,
)
from robot_localization.srv import SetPose
SetDepthRequest = SetDepth.Request
SetDepthResponse = SetDepth.Response
AlignFrameControllerResponse = AlignFrameController.Response
SetObjectTransformRequest = SetObjectTransform.Request
SetPoseRequest = SetPose.Request
from tf_transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_multiply,
    quaternion_inverse,
    quaternion_matrix,
)
from angles import normalize_angle, shortest_angular_distance
import numpy as np

from auv_common_lib.transform import is_transform_fresh
from threading import Lock
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from geometry_msgs.msg import Vector3Stamped


RIGHT_ANGLE_YAW_OFFSETS = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)


def normalize_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def closest_right_angle_yaw(source_yaw: float, reference_yaw: float) -> float:
    return min(
        (normalize_angle(source_yaw + offset) for offset in RIGHT_ANGLE_YAW_OFFSETS),
        key=lambda yaw: abs(shortest_angular_distance(yaw, reference_yaw)),
    )


def closest_half_turn_yaw(source_yaw: float, reference_yaw: float) -> float:
    return min(
        (normalize_angle(source_yaw + offset) for offset in (0.0, np.pi)),
        key=lambda yaw: abs(shortest_angular_distance(yaw, reference_yaw)),
    )


class ControlEnableHandler:
    def __init__(self, node, timeout_duration):
        self.node = node
        self.enable = False
        self.timeout_duration = Duration(seconds=timeout_duration)
        self.last_enable_time = self.node.get_clock().now()
        self.subscription = node.create_subscription(Bool, "enable", self._enable_callback, 10)

    def _enable_callback(self, msg):
        self.enable = msg.data
        if self.enable:
            self.last_enable_time = self.node.get_clock().now()

    def is_enabled(self) -> bool:
        current_time = self.node.get_clock().now()
        is_timeouted = (
            current_time - self.last_enable_time
        ).nanoseconds / 1e9 >= self.timeout_duration.nanoseconds / 1e9

        return self.enable and not is_timeouted


class ReferencePosePublisherNode(Node):
    def __init__(self):
        super().__init__("reference_pose_publisher_node")
        self._control_clients = ControlClients(self.get_namespace())
        for name, value in (("linear_kp", 1.0), ("linear_kd", 0.7),
                            ("angular_kp", 0.8), ("angular_kd", 0.3)):
            descriptor = ParameterDescriptor(floating_point_range=[
                FloatingPointRange(from_value=0.0, to_value=10.0, step=0.0)])
            self.declare_parameter(name, value, descriptor)
        self.tf_node = Node("reference_tf_listener", use_global_arguments=False)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)

        # subscribers
        self.odometry_sub = self.create_subscription(
            Odometry, "odometry", self.odometry_callback, 10
        )
        self.set_depth_service = self.create_service(
            SetDepth, "set_depth", lambda req, res: self.target_depth_handler(req)
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback, 10
        )

        # services
        self.reset_odometry_service = self.create_service(
            Trigger, "reset_odometry", lambda req, res: self.reset_odometry_handler(req)
        )
        self.align_frame_service = self.create_service(
            AlignFrameController, "align_frame/start", lambda req, res: self.handle_align_request(req)
        )
        self.cancel_control_service = self.create_service(
            Trigger, "align_frame/cancel", lambda req, res: self.handle_cancel_request(req)
        )
        self.reset_orientation_service = self.create_service(
            Trigger, "reset_command_orientation", lambda req, res: self.handle_reset_orientation_request(req)
        )
        self.sync_cmd_pose_service = self.create_service(
            Trigger, "sync_cmd_pose", lambda req, res: self.handle_sync_cmd_pose_request(req)
        )
        self.control_enable_handler = ControlEnableHandler(self, 1.0)
        self.set_pose_client = lambda req: self._control_clients.service(SetPose, self.resolve_service_name("set_pose"), req)

        # Service to broadcast cmd_pose as a frame
        self.set_object_transform_service = lambda req: self._control_clients.service(SetObjectTransform, self.resolve_service_name("set_object_transform"), req)

        # publishers
        self.cmd_pose_pub = self.create_publisher(PoseStamped, "cmd_pose", 10)
        self.enable_pub = self.create_publisher(Bool, "enable", 1)

        # target state
        self.target_frame_id = "odom"
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_depth = -0.4  # z
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_heading = 0.0  # yaw

        self.align_frame_active = False
        self.use_align_frame_depth = False
        self.align_frame_keep_orientation = False

        self.set_depth_velocity = None  # Track Z velocity set by set_depth

        self.last_cmd_time = self.get_clock().now()
        self.state_lock = Lock()
        self.latest_odometry = Odometry()

        self.set_pose_req = SetPoseRequest()
        self.set_pose_req.pose = PoseWithCovarianceStamped()
        self.set_pose_req.pose.header.stamp = self.get_clock().now().to_msg()
        self.set_pose_req.pose.header.frame_id = "odom"
        self.is_resetting = False

        # parameters
        self.namespace = parameter(self, "~namespace", "taluy")
        self.base_frame = self.namespace + "/base_link"
        self.update_rate = parameter(self, "~update_rate", 10.0)
        self.command_timeout = parameter(self, "~command_timeout", 0.1)
        self.max_xy_offset = parameter(self, "~max_xy_offset", 1.0)
        self.max_z_offset = parameter(self, "~max_z_offset", 0.2)
        self.max_z = parameter(self, "~max_z", 0.0)
        self.min_z = parameter(self, "~min_z", -2.0)
        self.max_yaw_offset = parameter(self, "~max_yaw_offset", np.pi / 18.0)
        self.tf_lookup_timeout = Duration(seconds=
            parameter(self, "~tf_lookup_timeout", 0.2)
        )
        self.tf_freshness_threshold = Duration(seconds=
            parameter(self, "~tf_freshness_threshold", 0.4)
        )

        self.killswitch_sub = self.create_subscription(
            Bool, "propulsion_board/status", self.killswitch_callback, 10
        )

        try:
            # Prefer a configurable server name; resolve relative names via ROS
            server_param = parameter(self,
                "~controller_reconfigure_server", "auv_control_node"
            )
            target_server = self.resolve_topic_name(server_param)
            self.reconfigure_client = ControllerParameters(
                self._control_clients, target_server, timeout=5
            )
            log(self, 'info', f"Connected to controller parameter services: {target_server}")

            # Capture current max velocity configuration as defaults to restore later
            current_cfg = self._read_controller_cfg()
            if current_cfg is not None:
                self.default_max_velocity = list(current_cfg["max_velocity"])
            else:
                log(self, 'warning',
                    "Failed to read initial controller configuration; using params/fallback"
                )
                self.default_max_velocity = self.reconfigure_client.get_parameter("max_velocity") or [1.0] * 6
        except Exception as e:
            log(self, 'warning', f"Failed to connect to controller parameter services: {e}")
            self.reconfigure_client = None
            self.default_max_velocity = [1.0] * 6

    def killswitch_callback(self, msg: Bool) -> None:
        if not msg.data:
            with self.state_lock:
                if self.align_frame_active:
                    self.cancel_align_controller()
                    log_throttle(self, 'info',
                        1.0, "Killswitch inactive; alignment deactivated"
                    )

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        with self.state_lock:
            if self.align_frame_active and self.use_align_frame_depth:
                return SetDepthResponse(
                    success=False,
                    message="Cannot set depth while align_frame is active and use_depth is true",
                )

            if req.frame_id == "" or req.frame_id is None:
                req.frame_id = "odom"

            depth = self.get_transformed_depth(
                "odom",
                req.frame_id,
                req.target_depth,
            )
            if depth is None:
                return SetDepthResponse(
                    success=False,
                    message="Failed to transform depth",
                )

            self.target_depth = depth

            # Update max velocity for Z axis
            if self.reconfigure_client:
                try:
                    max_velocity = list(self.reconfigure_client.get_parameter("max_velocity"))
                    if req.max_velocity > 0:
                        self.set_depth_velocity = req.max_velocity
                        max_velocity[2] = req.max_velocity
                    else:
                        # Restore default Z velocity
                        self.set_depth_velocity = None
                        max_velocity[2] = self.default_max_velocity[2]
                    self.reconfigure_client.update_configuration({"max_velocity": max_velocity})
                except Exception as e:
                    log(self, 'warning', f"Failed to update max velocity: {e}")

            return SetDepthResponse(
                success=True,
                message=f"Target depth set to {self.target_depth} in frame odom",
            )

    def handle_align_request(
        self, req: AlignFrameController
    ) -> AlignFrameControllerResponse:
        with self.state_lock:
            if self.align_frame_active:
                self.cancel_align_controller()

            self.align_frame_active = True

            t = self.tf_lookup(
                req.source_frame,
                self.base_frame,
                Time(),
                self.tf_lookup_timeout,
            )

            if t is None:
                self.cancel_align_controller()
                return AlignFrameControllerResponse(
                    success=False,
                    message="Failed to lookup transform",
                )

            self.target_x = t.transform.translation.x
            self.target_y = t.transform.translation.y

            self.use_align_frame_depth = req.use_depth
            if req.use_depth:
                self.target_depth = t.transform.translation.z

            self.align_frame_keep_orientation = req.keep_orientation
            if req.keep_orientation:
                source_in_target = self.tf_lookup(
                    req.target_frame,
                    req.source_frame,
                    Time(),
                    self.tf_lookup_timeout,
                )
                if source_in_target is None:
                    self.cancel_align_controller()
                    return AlignFrameControllerResponse(
                        success=False,
                        message="Failed to lookup source frame in target frame",
                    )

                offset_in_source = Vector3Stamped()
                offset_in_source.vector = t.transform.translation

                offset_in_target = do_transform_vector3(
                    offset_in_source, source_in_target
                )
                self.target_x = offset_in_target.vector.x
                self.target_y = offset_in_target.vector.y
            else:
                quaternion = [
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ]

                if req.angle_offset != 0.0:
                    q_rot = quaternion_from_euler(0, 0, req.angle_offset)
                    quaternion = quaternion_multiply(quaternion, q_rot)

                    # also rotate the position offset by angle_offset
                    rotation_tf = TransformStamped()
                    rotation_tf.transform.rotation.x = q_rot[0]
                    rotation_tf.transform.rotation.y = q_rot[1]
                    rotation_tf.transform.rotation.z = q_rot[2]
                    rotation_tf.transform.rotation.w = q_rot[3]

                    pos = PointStamped()
                    pos.point.x = self.target_x
                    pos.point.y = self.target_y
                    pos.point.z = 0.0

                    rotated_pos = do_transform_point(pos, rotation_tf)
                    self.target_x = rotated_pos.point.x
                    self.target_y = rotated_pos.point.y

                self.target_roll, self.target_pitch, self.target_heading = (
                    euler_from_quaternion(quaternion)
                )

                if req.closest_yaw:
                    base_in_source_quaternion = [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ]
                    t_base = self.tf_lookup(
                        req.target_frame,
                        self.base_frame,
                        Time(),
                        Duration(seconds=1.0),
                    )
                    if t_base is not None:
                        _, _, base_yaw_in_target = euler_from_quaternion(
                            [
                                t_base.transform.rotation.x,
                                t_base.transform.rotation.y,
                                t_base.transform.rotation.z,
                                t_base.transform.rotation.w,
                            ]
                        )
                        self.target_heading = closest_right_angle_yaw(
                            self.target_heading, base_yaw_in_target
                        )

                    desired_base_in_target = quaternion_from_euler(
                        self.target_roll, self.target_pitch, self.target_heading
                    )
                    desired_source_in_target = quaternion_multiply(
                        desired_base_in_target,
                        quaternion_inverse(base_in_source_quaternion),
                    )
                    rotation_matrix = quaternion_matrix(desired_source_in_target)[
                        :3, :3
                    ]
                    offset_in_target = rotation_matrix.dot(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                        ]
                    )
                    self.target_x, self.target_y = offset_in_target[:2]
                elif req.closest_yaw_180:
                    base_in_source_quaternion = [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ]
                    t_base = self.tf_lookup(
                        req.target_frame,
                        self.base_frame,
                        Time(),
                        Duration(seconds=1.0),
                    )
                    if t_base is not None:
                        _, _, base_yaw_in_target = euler_from_quaternion(
                            [
                                t_base.transform.rotation.x,
                                t_base.transform.rotation.y,
                                t_base.transform.rotation.z,
                                t_base.transform.rotation.w,
                            ]
                        )
                        self.target_heading = closest_half_turn_yaw(
                            self.target_heading, base_yaw_in_target
                        )

                    desired_base_in_target = quaternion_from_euler(
                        self.target_roll, self.target_pitch, self.target_heading
                    )
                    desired_source_in_target = quaternion_multiply(
                        desired_base_in_target,
                        quaternion_inverse(base_in_source_quaternion),
                    )
                    rotation_matrix = quaternion_matrix(desired_source_in_target)[
                        :3, :3
                    ]
                    offset_in_target = rotation_matrix.dot(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                        ]
                    )
                    self.target_x, self.target_y = offset_in_target[:2]

            self.target_frame_id = req.target_frame

            if self.reconfigure_client:
                linear_vel = (
                    req.max_linear_velocity
                    if req.max_linear_velocity > 0
                    else self.default_max_velocity[0]
                )
                angular_vel = (
                    req.max_angular_velocity
                    if req.max_angular_velocity > 0
                    else self.default_max_velocity[3]
                )
                self._update_controller_cfg(linear_vel, angular_vel, req.use_depth)

        log(self, 'info',
            f"Aligning {req.source_frame} to {req.target_frame} with angle offset {req.angle_offset}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req) -> TriggerResponse:
        with self.state_lock:
            if not self.align_frame_active:
                return TriggerResponse(
                    success=False, message="Alignment is not active."
                )

            self.cancel_align_controller()

        log(self, 'info', "Align frame control canceled")
        return TriggerResponse(success=True, message="Alignment deactivated")

    def handle_reset_orientation_request(self, req) -> TriggerResponse:
        if self.align_frame_active:
            return TriggerResponse(
                success=False,
                message="Cannot reset orientation while align_frame is active.",
            )

        with self.state_lock:
            self.target_roll = 0.0
            self.target_pitch = 0.0

        log(self, 'info', "Roll and pitch reset to zero")
        return TriggerResponse(success=True, message="Orientation reset successfully")

    def handle_sync_cmd_pose_request(self, req) -> TriggerResponse:
        """Sync cmd_pose to current robot position (odometry)."""
        with self.state_lock:
            self.set_target_to_odometry()
        log(self, 'info', "cmd_pose synced to current odometry")
        return TriggerResponse(success=True, message="cmd_pose synced to odometry")

    def cancel_align_controller(self):
        self.set_target_to_odometry()
        self.align_frame_active = False
        self.use_align_frame_depth = False
        self.align_frame_keep_orientation = False

        if self.reconfigure_client:
            self._restore_controller_cfg()

    # --- Helper methods for dynamic reconfigure handling ---
    def _read_controller_cfg(self):
        if not self.reconfigure_client:
            return None
        try:
            return self.reconfigure_client.get_configuration()
        except Exception as e:
            log(self, 'warning', f"Failed to read controller configuration: {e}")
            return None

    def _update_controller_cfg(
        self, linear_vel: float, angular_vel: float, use_depth: bool = True
    ):
        try:
            # Determine Z velocity: use linear_vel if use_depth, else preserve current or use default
            if use_depth:
                z_vel = linear_vel
            elif self.set_depth_velocity is not None:
                z_vel = self.set_depth_velocity
            else:
                z_vel = self.default_max_velocity[2]

            self.reconfigure_client.update_configuration(
                {
                    "max_velocity": [linear_vel, linear_vel, z_vel,
                                     angular_vel, angular_vel, angular_vel],
                }
            )
        except Exception as e:
            log(self, 'warning', f"Failed to update controller configuration: {e}")

    def _restore_controller_cfg(self):
        try:
            # Preserve Z velocity if it was set by set_depth
            z_vel = (
                self.set_depth_velocity
                if self.set_depth_velocity is not None
                else self.default_max_velocity[2]
            )
            self.reconfigure_client.update_configuration(
                {
                    "max_velocity": [*self.default_max_velocity[:2], z_vel,
                                     *self.default_max_velocity[3:]],
                }
            )
        except Exception as e:
            log(self, 'warning', f"Failed to reset controller configuration: {e}")

    def reset_odometry_handler(self, req):
        if self.is_resetting:
            return TriggerResponse(
                success=False, message="Odometry reset already in progress."
            )

        self.is_resetting = True
        log(self, 'debug', "Starting odometry reset.")

        try:
            self.set_pose_client(self.set_pose_req)
            log(self, 'debug', "Called set_pose service.")
        except Exception as e:
            log(self, 'error', f"Service call failed: {e}")
            self.is_resetting = False
            return TriggerResponse(success=False, message=f"Service call failed: {e}")

        log(self, 'debug', "Waiting for heading to settle.")
        self.get_clock().sleep_for(Duration(seconds=2.0))

        self.is_resetting = False
        log(self, 'debug', "Odometry reset finished.")
        return TriggerResponse(success=True, message="Odometry reset successfully.")

    def odometry_callback(self, msg):
        self.latest_odometry = msg

        if (
            self.control_enable_handler.is_enabled() and not self.is_resetting
        ) or self.align_frame_active:
            return

        self.set_target_to_odometry()

    def set_target_to_odometry(self):
        self.target_x = self.latest_odometry.pose.pose.position.x
        self.target_y = self.latest_odometry.pose.pose.position.y

        # If use_align_frame_depth=False, depth is already in odom frame
        if self.target_frame_id != "odom" and self.use_align_frame_depth:
            depth = self.get_transformed_depth(
                "odom",
                self.target_frame_id,
                self.target_depth,
            )
            if depth is None:
                return

            self.target_depth = depth

        self.target_frame_id = "odom"

        quaternion = [
            self.latest_odometry.pose.pose.orientation.x,
            self.latest_odometry.pose.pose.orientation.y,
            self.latest_odometry.pose.pose.orientation.z,
            self.latest_odometry.pose.pose.orientation.w,
        ]
        _, _, self.target_heading = euler_from_quaternion(quaternion)

        self.target_roll = 0.0
        self.target_pitch = 0.0

    def get_transformed_depth(
        self,
        target_frame: str,
        source_frame: str,
        target_depth: float,
        lookup_time: Time = None,
    ):
        if lookup_time is None:
            lookup_time = Time()

        source_to_odom = self.tf_lookup(
            "odom", source_frame, lookup_time, self.tf_lookup_timeout
        )
        if source_to_odom is None:
            return None

        target_to_odom = self.tf_lookup(
            "odom", target_frame, lookup_time, self.tf_lookup_timeout
        )
        if target_to_odom is None:
            return None

        depth_in_odom = source_to_odom.transform.translation.z + target_depth

        return depth_in_odom - target_to_odom.transform.translation.z

    def get_transformed_orientation(
        self,
        target_frame: str,
        source_frame: str,
        roll: float,
        pitch: float,
        yaw: float,
        lookup_time: Time = None,
    ):
        if lookup_time is None:
            lookup_time = Time()

        transform = self.tf_lookup(
            target_frame,
            source_frame,
            lookup_time,
            self.tf_lookup_timeout,
        )

        if transform is None:
            return None

        q_orientation = quaternion_from_euler(roll, pitch, yaw)

        q_transform = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]

        q_new = quaternion_multiply(q_transform, q_orientation)

        new_roll, new_pitch, new_yaw = euler_from_quaternion(q_new)
        return (new_roll, new_pitch, new_yaw)

    def cmd_vel_callback(self, msg: Twist):
        if (
            not self.control_enable_handler.is_enabled()
            or self.is_resetting
            or self.align_frame_active
        ):
            return

        with self.state_lock:
            now = self.get_clock().now()
            dt = (now - self.last_cmd_time).nanoseconds / 1e9
            dt = min(dt, self.command_timeout)

            q_current = quaternion_from_euler(
                self.target_roll, self.target_pitch, self.target_heading
            )

            rotation_matrix = quaternion_matrix(q_current)[:3, :3]

            linear_vel_body = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
            linear_vel_odom = rotation_matrix.dot(linear_vel_body)

            self.target_x += linear_vel_odom[0] * dt
            self.target_y += linear_vel_odom[1] * dt
            self.target_depth += linear_vel_odom[2] * dt

            odom_position = self.latest_odometry.pose.pose.position
            xy_offset = np.array(
                [
                    self.target_x - odom_position.x,
                    self.target_y - odom_position.y,
                ]
            )
            xy_distance = np.linalg.norm(xy_offset)
            if xy_distance > self.max_xy_offset and xy_distance > 1e-6:
                xy_offset *= self.max_xy_offset / xy_distance
                self.target_x = odom_position.x + xy_offset[0]
                self.target_y = odom_position.y + xy_offset[1]

            z_offset = self.target_depth - odom_position.z
            if abs(z_offset) > self.max_z_offset:
                self.target_depth = (
                    odom_position.z + np.sign(z_offset) * self.max_z_offset
                )

            self.target_depth = np.clip(self.target_depth, self.min_z, self.max_z)

            angle = np.linalg.norm([msg.angular.x, msg.angular.y, msg.angular.z]) * dt
            if angle > 1e-6:  # Avoid division by zero
                axis = np.array([msg.angular.x, msg.angular.y, msg.angular.z]) / (
                    angle / dt
                )
                axis = axis / np.linalg.norm(axis)
                q_delta = quaternion_from_euler(
                    axis[0] * angle, axis[1] * angle, axis[2] * angle
                )
            else:
                q_delta = [0.0, 0.0, 0.0, 1.0]

            # Multiply current orientation by delta rotation (cmd_pose frame rotation)
            q_new = quaternion_multiply(q_current, q_delta)

            # Extract new euler angles
            self.target_roll, self.target_pitch, self.target_heading = (
                euler_from_quaternion(q_new)
            )

            odom_quaternion = [
                self.latest_odometry.pose.pose.orientation.x,
                self.latest_odometry.pose.pose.orientation.y,
                self.latest_odometry.pose.pose.orientation.z,
                self.latest_odometry.pose.pose.orientation.w,
            ]
            _, _, odom_yaw = euler_from_quaternion(odom_quaternion)
            yaw_error = normalize_angle(self.target_heading - odom_yaw)
            yaw_error = np.clip(yaw_error, -self.max_yaw_offset, self.max_yaw_offset)
            self.target_heading = normalize_angle(odom_yaw + yaw_error)

            self.last_cmd_time = now

    def tf_lookup(
        self,
        target_frame: str,
        source_frame: str,
        time: Time,
        timeout: Duration,
    ):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time,
                timeout,
            )
            if time.nanoseconds == 0 and not is_transform_fresh(
                transform, self, self.tf_freshness_threshold
            ):
                age = abs((self.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds / 1e9)
                log_throttle(self, 'warning',
                    1.0,
                    f"TF lookup from {source_frame} to {target_frame} returned stale data "
                    f"({age:.3f}s > {self.tf_freshness_threshold.nanoseconds / 1e9:.3f}s)",
                )
                return None
            return transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            log_throttle(self, 'warning',
                1.0, f"TF lookup failed from {source_frame} to {target_frame}: {e}"
            )
            return None

    def publish_cmd_pose_frame(self, cmd_pose: PoseStamped):
        try:
            transform = TransformStamped()
            transform.header = cmd_pose.header
            transform.child_frame_id = "cmd_pose"
            transform.transform.translation.x = cmd_pose.pose.position.x
            transform.transform.translation.y = cmd_pose.pose.position.y
            transform.transform.translation.z = cmd_pose.pose.position.z
            transform.transform.rotation = cmd_pose.pose.orientation

            req = SetObjectTransformRequest()
            req.transform = transform
            self.set_object_transform_service(req)
        except Exception as e:
            log_throttle(self, 'warning', 5.0, f"Failed to publish cmd_pose frame: {e}")

    def control_loop(self):
        with self.state_lock:
            frame_id = self.target_frame_id
            tx = self.target_x
            ty = self.target_y
            tz = self.target_depth

            t_roll = self.target_roll
            t_pitch = self.target_pitch
            t_heading = self.target_heading

            use_align_depth = self.use_align_frame_depth
            align_keep_orient = self.align_frame_keep_orientation
            align_active = self.align_frame_active

        cmd_pose_stamped = PoseStamped()
        cmd_pose_stamped.header.frame_id = frame_id

        cmd_pose_stamp = self.get_clock().now()
        if frame_id != "odom":
            target_to_odom = self.tf_lookup(
                "odom",
                frame_id,
                Time(),
                self.tf_lookup_timeout,
            )
            if target_to_odom is None:
                log_throttle(self, 'warning',
                    1.0, f"Failed to lookup target frame '{frame_id}'"
                )
                return
            cmd_pose_stamp = Time.from_msg(target_to_odom.header.stamp)

        cmd_pose_stamped.header.stamp = cmd_pose_stamp.to_msg()

        cmd_pose_stamped.pose.position.x = tx
        cmd_pose_stamped.pose.position.y = ty

        if not use_align_depth:
            transformed_depth = self.get_transformed_depth(
                frame_id,
                "odom",
                tz,
                cmd_pose_stamp,
            )
            if transformed_depth is None:
                log_throttle(self, 'error', 1.0, "Failed to transform target depth")
                return
            cmd_pose_stamped.pose.position.z = transformed_depth
        else:
            cmd_pose_stamped.pose.position.z = tz

        if align_keep_orient:
            transformed_rpy = self.get_transformed_orientation(
                frame_id,
                "odom",
                t_roll,
                t_pitch,
                t_heading,
                cmd_pose_stamp,
            )
            if transformed_rpy is None:
                log_throttle(self, 'error', 1.0, "Failed to transform target orientation")
                return
            roll, pitch, yaw = transformed_rpy
        else:
            roll = t_roll
            pitch = t_pitch
            yaw = t_heading

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        cmd_pose_stamped.pose.orientation.x = quaternion[0]
        cmd_pose_stamped.pose.orientation.y = quaternion[1]
        cmd_pose_stamped.pose.orientation.z = quaternion[2]
        cmd_pose_stamped.pose.orientation.w = quaternion[3]

        self.cmd_pose_pub.publish(cmd_pose_stamped)

        # Publish cmd_pose as a TF frame
        self.publish_cmd_pose_frame(cmd_pose_stamped)

        if align_active:
            self.enable_pub.publish(Bool(data=True))

    def run(self):
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop)
        # Callbacks must run side by side as in ROS 1: reset_odometry sleeps in its
        # handler and relies on odometry callbacks arriving meanwhile.
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.spin()


def main():
    ros.init()
    node = ReferencePosePublisherNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.tf_listener.unregister()
        node.tf_node.destroy_node()
        node._control_clients.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
