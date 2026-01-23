#!/usr/bin/env python3

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    PoseStamped,
    Twist,
    PoseWithCovarianceStamped,
    PointStamped,
    TransformStamped,
)
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool
from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetDepthResponse,
    AlignFrameController,
    AlignFrameControllerResponse,
    SetObjectTransform,
    SetObjectTransformRequest,
)
from robot_localization.srv import SetPose, SetPoseRequest
from tf.transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_multiply,
    quaternion_matrix,
)
import numpy as np

import dynamic_reconfigure.client
from auv_common_lib.control.enable_state import ControlEnableHandler
from threading import Lock
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from geometry_msgs.msg import Vector3Stamped


class ReferencePosePublisherNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # subscribers
        self.odometry_sub = rospy.Subscriber(
            "odometry", Odometry, self.odometry_callback, tcp_nodelay=True
        )
        self.set_depth_service = rospy.Service(
            "set_depth", SetDepth, self.target_depth_handler
        )
        self.cmd_vel_sub = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
        )

        # services
        self.reset_odometry_service = rospy.Service(
            "reset_odometry", Trigger, self.reset_odometry_handler
        )
        self.align_frame_service = rospy.Service(
            "align_frame/start", AlignFrameController, self.handle_align_request
        )
        self.cancel_control_service = rospy.Service(
            "align_frame/cancel", Trigger, self.handle_cancel_request
        )
        self.reset_orientation_service = rospy.Service(
            "reset_command_orientation", Trigger, self.handle_reset_orientation_request
        )
        self.control_enable_handler = ControlEnableHandler(1.0)
        self.set_pose_client = rospy.ServiceProxy("set_pose", SetPose)

        # Service to broadcast cmd_pose as a frame
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        # publishers
        self.cmd_pose_pub = rospy.Publisher("cmd_pose", PoseStamped, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

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

        self.last_cmd_time = rospy.Time.now()
        self.state_lock = Lock()
        self.latest_odometry = Odometry()

        self.set_pose_req = SetPoseRequest()
        self.set_pose_req.pose = PoseWithCovarianceStamped()
        self.set_pose_req.pose.header.stamp = rospy.Time.now()
        self.set_pose_req.pose.header.frame_id = "odom"
        self.is_resetting = False

        # parameters
        self.namespace = rospy.get_param("~namespace", "taluy")
        self.base_frame = self.namespace + "/base_link"
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.command_timeout = rospy.get_param("~command_timeout", 0.1)

        self.killswitch_sub = rospy.Subscriber(
            "propulsion_board/status", Bool, self.killswitch_callback
        )

        try:
            # Prefer a configurable server name; resolve relative names via ROS
            server_param = rospy.get_param(
                "~controller_reconfigure_server", "auv_control_node"
            )
            target_server = rospy.resolve_name(server_param)
            self.reconfigure_client = dynamic_reconfigure.client.Client(
                target_server, timeout=5
            )
            rospy.loginfo(f"Connected to dynamic reconfigure server: {target_server}")

            # Capture current max velocity configuration as defaults to restore later
            current_cfg = self._read_controller_cfg()
            if current_cfg is not None:
                self.default_max_velocity = [
                    current_cfg.get("max_velocity_0", 1.0),
                    current_cfg.get("max_velocity_1", 1.0),
                    current_cfg.get("max_velocity_2", 1.0),
                    current_cfg.get("max_velocity_3", 1.0),
                    current_cfg.get("max_velocity_4", 1.0),
                    current_cfg.get("max_velocity_5", 1.0),
                ]
            else:
                rospy.logwarn(
                    "Failed to read initial controller configuration; using params/fallback"
                )
                self.default_max_velocity = rospy.get_param(
                    f"{target_server}/max_velocity", [1.0] * 6
                )
        except Exception as e:
            rospy.logwarn(f"Failed to connect to dynamic reconfigure server: {e}")
            self.reconfigure_client = None
            self.default_max_velocity = [1.0] * 6

    def killswitch_callback(self, msg: Bool) -> None:
        if not msg.data:
            with self.state_lock:
                if self.align_frame_active:
                    self.set_target_to_odometry()
                    self.align_frame_active = False
                    self.use_align_frame_depth = False
                    self.align_frame_keep_orientation = False
                    if self.reconfigure_client:
                        self._restore_controller_cfg()
            rospy.loginfo_throttle(1.0, "Killswitch inactive; alignment deactivated")

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
            return SetDepthResponse(
                success=True,
                message=f"Target depth set to {self.target_depth} in frame odom",
            )

    def handle_align_request(
        self, req: AlignFrameController
    ) -> AlignFrameControllerResponse:
        with self.state_lock:
            if self.align_frame_active:
                self.set_target_to_odometry()
                self.use_align_frame_depth = False
                self.align_frame_keep_orientation = False

            self.align_frame_active = True

            t = self.tf_lookup(
                req.source_frame, self.base_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            if t is None:
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
            if not req.keep_orientation:
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
                self._update_controller_cfg(linear_vel, angular_vel)

        rospy.loginfo(
            f"Aligning {req.source_frame} to {req.target_frame} with angle offset {req.angle_offset}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req) -> TriggerResponse:
        with self.state_lock:
            if not self.align_frame_active:
                return TriggerResponse(
                    success=False, message="Alignment is not active."
                )

            self.set_target_to_odometry()
            self.align_frame_active = False
            self.use_align_frame_depth = False
            self.align_frame_keep_orientation = False

            if self.reconfigure_client:
                self._restore_controller_cfg()

        rospy.loginfo("Align frame control canceled")
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

        rospy.loginfo("Roll and pitch reset to zero")
        return TriggerResponse(success=True, message="Orientation reset successfully")

    # --- Helper methods for dynamic reconfigure handling ---
    def _read_controller_cfg(self):
        if not self.reconfigure_client:
            return None
        try:
            return self.reconfigure_client.get_configuration()
        except Exception as e:
            rospy.logwarn(f"Failed to read controller configuration: {e}")
            return None

    def _update_controller_cfg(self, linear_vel: float, angular_vel: float):
        try:
            self.reconfigure_client.update_configuration(
                {
                    "max_velocity_0": linear_vel,
                    "max_velocity_1": linear_vel,
                    "max_velocity_2": linear_vel,
                    "max_velocity_3": angular_vel,
                    "max_velocity_4": angular_vel,
                    "max_velocity_5": angular_vel,
                }
            )
        except Exception as e:
            rospy.logwarn(f"Failed to update controller configuration: {e}")

    def _restore_controller_cfg(self):
        try:
            self.reconfigure_client.update_configuration(
                {
                    "max_velocity_0": self.default_max_velocity[0],
                    "max_velocity_1": self.default_max_velocity[1],
                    "max_velocity_2": self.default_max_velocity[2],
                    "max_velocity_3": self.default_max_velocity[3],
                    "max_velocity_4": self.default_max_velocity[4],
                    "max_velocity_5": self.default_max_velocity[5],
                }
            )
        except Exception as e:
            rospy.logwarn(f"Failed to reset controller configuration: {e}")

    def reset_odometry_handler(self, req):
        if self.is_resetting:
            return TriggerResponse(
                success=False, message="Odometry reset already in progress."
            )

        self.is_resetting = True
        rospy.logdebug("Starting odometry reset.")

        try:
            self.set_pose_client(self.set_pose_req)
            rospy.logdebug("Called set_pose service.")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"Service call failed: {e}")
            self.is_resetting = False
            return TriggerResponse(success=False, message=f"Service call failed: {e}")

        rospy.logdebug("Waiting for heading to settle.")
        rospy.sleep(2.0)

        self.is_resetting = False
        rospy.logdebug("Odometry reset finished.")
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
        self, target_frame: str, source_frame: str, target_depth: float
    ):
        transform = self.tf_lookup(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(1.0),
        )

        if transform is None:
            return None

        p = PointStamped()
        p.header.stamp = transform.header.stamp
        p.header.frame_id = source_frame
        p.point.x = 0.0
        p.point.y = 0.0
        p.point.z = target_depth

        p_in_target = do_transform_point(p, transform)
        return p_in_target.point.z

    def get_transformed_orientation(
        self,
        target_frame: str,
        source_frame: str,
        roll: float,
        pitch: float,
        yaw: float,
    ):
        transform = self.tf_lookup(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(1.0),
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

        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
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

        self.last_cmd_time = rospy.Time.now()

    def tf_lookup(
        self,
        target_frame: str,
        source_frame: str,
        time: rospy.Time,
        timeout: rospy.Duration,
    ):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time,
                timeout,
            )
            return transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"TF lookup failed from {source_frame} to {target_frame}: {e}")
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
            rospy.logwarn_throttle(5.0, f"Failed to publish cmd_pose frame: {e}")

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
        cmd_pose_stamped.header.stamp = rospy.Time.now()
        cmd_pose_stamped.header.frame_id = frame_id

        cmd_pose_stamped.pose.position.x = tx
        cmd_pose_stamped.pose.position.y = ty

        if not use_align_depth:
            transformed_depth = self.get_transformed_depth(frame_id, "odom", tz)
            if transformed_depth is None:
                rospy.logerr_throttle(1.0, "Failed to transform target depth")
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
            )
            if transformed_rpy is None:
                rospy.logerr_throttle(1.0, "Failed to transform target orientation")
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
        rate = rospy.Rate(self.update_rate)

        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("reference_pose_publisher_node")
        node = ReferencePosePublisherNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
