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
        self.gate_angle = None
        self.publish_gate_angle_enabled = False
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
        self.pool_x_offset = 1.0
        self.pool_y_offset = 0.0
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
        self.gate_angle_publisher = rospy.Publisher(
            "gate_angle", Float64, queue_size=10
        )
        self.publish_gate_angle_service = rospy.Service(
            "publish_gate_angle", Trigger, self.handle_publish_gate_angle
        )

        self.odom_frame = "odom"
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.entrance_frame = "gate_entrance"
        self.exit_frame = "gate_exit"
        self.middle_frame = "gate_middle_part"
        self.pool_frame = "gate_to_pool"

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

        self.coin_flip_enabled = False
        self.coin_flip_rescuer_pose = None
        self.rescuer_frame_name = "coin_flip_rescuer"

        self.toggle_rescuer_service = rospy.Service(
            "toggle_coin_flip_rescuer", SetBool, self.handle_toggle_rescuer_service
        )

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

    def handle_toggle_rescuer_service(self, request: SetBool) -> SetBoolResponse:

        self.coin_flip_enabled = request.data
        message = f"Coin flip rescuer frame publishing set to: {self.coin_flip_enabled}"
        rospy.loginfo(message)

        if request.data:
            self._publish_rescuer_frame()

        if self.coin_flip_enabled and self.coin_flip_rescuer_pose is None:
            try:
                initial_transform = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.robot_base_frame,
                    rospy.Time(0),
                    rospy.Duration(4.0),
                )

                initial_pos = initial_transform.transform.translation

                rescuer_pose = Pose()
                rescuer_pose.position.x = initial_pos.x + self.rescuer_distance
                rescuer_pose.position.y = initial_pos.y
                rescuer_pose.position.z = initial_pos.z

                q = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
                rescuer_pose.orientation = Quaternion(*q)

                self.coin_flip_rescuer_pose = rescuer_pose
                return SetBoolResponse(success=True, message=message)

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                error_message = f"Failed to get initial vehicle pose to calculate rescuer frame: {e}"
                rospy.logerr(error_message)
                self.coin_flip_enabled = False
                return SetBoolResponse(success=False, message=error_message)

        return SetBoolResponse(success=True, message=message)

    def odom_callback(self, msg):
        self.latest_odom = msg

    def _publish_rescuer_frame(self):
        if not self.coin_flip_enabled:
            rospy.logdebug("Coin flip rescuer frame publishing is disabled.")
            return
        if self.latest_odom is None:
            rospy.logwarn("Coin flip rescuer frame odometry message not received.")
            return

        pos = self.latest_odom.pose.pose.position
        reference_yaw = self.wall_reference_yaw
        dx = self.rescuer_distance * math.cos(reference_yaw)
        dy = self.rescuer_distance * math.sin(reference_yaw)
        rescuer_x = pos.x + dx
        rescuer_y = pos.y + dy
        rescuer_z = pos.z
        rescuer_quat = tf_conversions.transformations.quaternion_from_euler(
            0, 0, reference_yaw
        )
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
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
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
        except tf2_ros.TransformException:
            t_gate1 = None

        try:
            t_gate2 = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.gate_frame_2,
                rospy.Time.now(),
                rospy.Duration(4.0),
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

            # --- Create gate_middle_part_link at the midpoint between gate_shark_link and gate_sawfish_link
            middle_x = (p1.x + p2.x) / 2.0
            middle_y = (p1.y + p2.y) / 2.0
            middle_z = (p1.z + p2.z) / 2.0
            # Orientation: same as odom frame (no rotation)
            identity_quat = tf_conversions.transformations.quaternion_from_euler(
                0, 0, 0
            )
            middle_pose = Pose(
                position=Point(middle_x, middle_y, middle_z),
                orientation=Quaternion(*identity_quat),
            )
            middle_transform = self.build_transform_message(
                self.middle_frame, middle_pose
            )
            self.send_transform(middle_transform)

            if self.MIN_GATE_SEPARATION < distance < self.MAX_GATE_SEPARATION:
                # Distance is valid, use standard dual-frame method
                rospy.loginfo_once("Using dual-frame mode for gate trajectory.")
                poses = self._compute_frames_dual_mode(
                    self.get_translation(t_gate1), self.get_translation(t_gate2)
                )
            else:
                # Distance is out of bounds, use fallback method with the selected target
                rospy.logwarn_once(
                    f"Gate distance ({distance:.2f}m) is out of bounds. Using fallback mode."
                )
                target_transform = (
                    t_gate1 if self.target_gate_frame == self.gate_frame_1 else t_gate2
                )
                poses = self._compute_frames_fallback_mode(target_transform)

        elif t_gate1 or t_gate2:
            # Only one frame is visible, use fallback method
            rospy.logwarn_once("Only one gate frame is visible. Using fallback mode.")
            visible_transform = t_gate1 if t_gate1 else t_gate2
            # --- Place gate_middle_part at the visible frame's position
            p = visible_transform.transform.translation
            q = visible_transform.transform.rotation
            middle_pose = Pose(
                position=Point(p.x, p.y, p.z),
                orientation=Quaternion(q.x, q.y, q.z, q.w),
            )
            middle_transform = self.build_transform_message(
                self.middle_frame, middle_pose
            )
            self.send_transform(middle_transform)
            poses = self._compute_frames_fallback_mode(visible_transform)

        else:
            # No frames are visible
            rospy.logwarn("Neither gate frame is visible. Cannot create trajectory.")
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

            # Create and send the gate_to_pool frame relative to the exit frame
            pool_pose = Pose()
            pool_pose.position.x = exit_pose.position.x + self.pool_x_offset
            pool_pose.position.y = exit_pose.position.y + self.pool_y_offset
            pool_pose.position.z = exit_pose.position.z  # Keep the same Z
            pool_pose.orientation = exit_pose.orientation  # Keep the same orientation

            pool_transform = self.build_transform_message(self.pool_frame, pool_pose)
            self.send_transform(pool_transform)

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
                rospy.Time(0),
                rospy.Duration(4.0),
            )
        except tf2_ros.TransformException as e:
            rospy.logwarn(
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
            rospy.logwarn(
                "Robot is too close to the gate frame for fallback calculation."
            )
            return None

        # Unit vector for the direction
        unit_dx = dx / length
        unit_dy = dy / length

        # Position entrance before the gate, exit after the gate
        entrance_position = Point(
            gate_pos.x - unit_dx * self.fallback_entrance_offset,
            gate_pos.y - unit_dy * self.fallback_entrance_offset,
            gate_pos.z - self.z_offset,  # Apply Z-offset
        )
        exit_position = Point(
            gate_pos.x + unit_dx * self.fallback_exit_offset,
            gate_pos.y + unit_dy * self.fallback_exit_offset,
            gate_pos.z - self.z_offset,  # Apply Z-offset
        )

        # Orientation should look along the line of approach
        common_yaw = math.atan2(dy, dx)
        common_quat = tf_conversions.transformations.quaternion_from_euler(
            0, 0, common_yaw
        )

        entrance_pose = Pose(
            position=entrance_position, orientation=Quaternion(*common_quat)
        )
        exit_pose = Pose(position=exit_position, orientation=Quaternion(*common_quat))

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
            rospy.logwarn(f"Could not compute dual-mode frames: {e}")
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
                rospy.Time(0),
                rospy.Duration(4.0),
            )
            robot_pos = robot_transform.transform.translation
        except tf2_ros.TransformException as e:
            rospy.logwarn(
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
        common_quat = tf_conversions.transformations.quaternion_from_euler(
            0, 0, common_yaw
        )

        entrance_pose = Pose(
            position=Point(*entrance_position), orientation=Quaternion(*common_quat)
        )
        exit_pose = Pose(
            position=Point(*exit_position), orientation=Quaternion(*common_quat)
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
                rospy.Time.now(),
                rospy.Duration(4.0),
            )
        except tf2_ros.TransformException as e:
            rospy.logwarn(f"Parallel shift failed due to TF Error: {e}")
            return transform_to_shift

        dx_sel = transform_selected_to_other.transform.translation.x
        dy_sel = transform_selected_to_other.transform.translation.y
        length = math.sqrt(dx_sel**2 + dy_sel**2)

        if length < self.MIN_GATE_SEPARATION_THRESHOLD:
            rospy.logwarn("Gate links too close for parallel shift. Skipping.")
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

    def handle_publish_gate_angle(self, req):
        self.publish_gate_angle_enabled = True
        message = "Gate angle publishing enabled."
        rospy.loginfo(message)
        return TriggerResponse(success=True, message=message)

    def _publish_gate_angle_loop(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if self.publish_gate_angle_enabled and self.gate_angle is not None:
                self.gate_angle_publisher.publish(self.gate_angle)
            rate.sleep()

    def spin(self):
        gate_angle_thread = threading.Thread(target=self._publish_gate_angle_loop)
        gate_angle_thread.daemon = True
        gate_angle_thread.start()

        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.create_trajectory_frames()

            if self.coin_flip_enabled:
                self._publish_rescuer_frame()

            rate.sleep()

    def reconfigure_callback(self, config, level):
        self.entrance_offset = config.entrance_offset
        self.exit_offset = config.exit_offset
        self.z_offset = config.z_offset
        self.parallel_shift_offset = config.parallel_shift_offset
        self.rescuer_distance = config.rescuer_distance
        self.pool_x_offset = config.pool_x_offset
        self.pool_y_offset = config.pool_y_offset
        return config


if __name__ == "__main__":
    try:
        node = TransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
