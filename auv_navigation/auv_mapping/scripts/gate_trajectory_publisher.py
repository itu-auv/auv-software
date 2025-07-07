#!/usr/bin/env python3
from typing import Tuple
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


class TransformServiceNode:
    def __init__(self):
        self.is_enabled = False
        self.gate_angle = None
        self.publish_gate_angle_enabled = False
        rospy.init_node("create_gate_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
        self.entrance_frame = "gate_entrance"
        self.exit_frame = "gate_exit"

        # Gate frames
        self.gate_frame_1 = rospy.get_param("~gate_frame_1", "gate_blue_arrow_link")
        self.gate_frame_2 = rospy.get_param("~gate_frame_2", "gate_red_arrow_link")
        self.set_enable_service = rospy.Service(
            "toggle_gate_trajectory", SetBool, self.handle_enable_service
        )

        self.entrance_offset = rospy.get_param("~entrance_offset", 1.0)
        self.exit_offset = rospy.get_param("~exit_offset", 1.0)
        self.z_offset = rospy.get_param("~z_offset", 0.5)
        self.parallel_shift_offset = rospy.get_param("~parallel_shift_offset", 0.15)
        self.target_gate_frame = rospy.get_param(
            "~target_gate_frame", "gate_blue_arrow_link"
        )

        # Threshold for gate link separation
        self.MIN_GATE_SEPARATION_THRESHOLD = 0.15

    def assign_selected_gate_translations(
        self,
        gate_link_1_translation: Tuple[float, float, float],
        gate_link_2_translation: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Determines selected_gate_link and other_gate_link based on self.target_gate_frame.
        """
        if self.target_gate_frame == self.gate_frame_1:
            selected_gate_link_translation = gate_link_1_translation
            other_gate_link_translation = gate_link_2_translation
        elif self.target_gate_frame == self.gate_frame_2:
            selected_gate_link_translation = gate_link_2_translation
            other_gate_link_translation = gate_link_1_translation
        else:
            raise ValueError("Invalid selected gate frame")
        return selected_gate_link_translation, other_gate_link_translation

    def compute_entrance_and_exit(
        self,
        selected_gate_link_translation: Tuple[float, float, float],
        other_gate_link_translation: Tuple[float, float, float],
    ) -> Tuple[Pose, Pose]:
        """
        Compute entrance and exit transforms based on the selected gate frame.

        1. Define a perpendicular unit vector to the line between the two gate frames.
        2. Calculate the two shifted positions relative to selected_gate_link_translation:
            - entrance_position: offset in the negative perpendicular direction.
            - exit_position: offset in the positive perpendicular direction.
        3. Apply a vertical offset below the selected gate frame's z-value by z_offset.
        """

        dx = other_gate_link_translation[0] - selected_gate_link_translation[0]
        dy = other_gate_link_translation[1] - selected_gate_link_translation[1]

        length = math.sqrt(dx**2 + dy**2)
        if length < self.MIN_GATE_SEPARATION_THRESHOLD:
            raise ValueError(
                "The gate links are almost identical or at the same position"
            )

        # Perpendicular unit vector to the line connecting selected_gate_link and other_gate_link frames.
        unit_perpendicular_x = -dy / length
        unit_perpendicular_y = dx / length

        # Calculate new positions relative to selected_translation
        entrance_position = (
            -unit_perpendicular_x * self.entrance_offset,
            -unit_perpendicular_y * self.entrance_offset,
            selected_gate_link_translation[2]
            - self.z_offset,  # 0.5m below selected frame
        )

        exit_position = (
            unit_perpendicular_x * self.exit_offset,
            unit_perpendicular_y * self.exit_offset,
            selected_gate_link_translation[2]
            - self.z_offset,  # 0.5m below selected frame
        )

        # Calculate orientations so that the frames look toward the origin (0,0,0 in local frame)
        common_yaw = math.atan2(
            -entrance_position[1],  # Look toward (0,0,0)
            -entrance_position[0],
        )
        common_quat = tf_conversions.transformations.quaternion_from_euler(
            0, 0, common_yaw
        )

        entrance_quaternion = common_quat
        exit_quaternion = common_quat

        # Create the entrance and exit poses
        entrance_pose = Pose()
        entrance_pose.position = Point(*entrance_position)
        entrance_pose.orientation = Quaternion(*entrance_quaternion)

        exit_pose = Pose()
        exit_pose.position = Point(*exit_position)
        exit_pose.orientation = Quaternion(*exit_quaternion)

        return entrance_pose, exit_pose

    def _shift_transform_parallel_to_gate_line(
        self,
        transform_to_shift: TransformStamped,
        selected_gate_frame_name: str,
        other_gate_frame_name: str,
        parallel_offset: float,
        tf_buffer: tf2_ros.Buffer,
    ) -> TransformStamped:
        """
        Shifts the given transform parallel to the line connecting selected_gate_frame_name
        and other_gate_frame_name, in the direction from selected to other.
        The shift is applied in the XY plane of the selected_gate_frame_name.
        """
        try:
            transform_selected_to_other = tf_buffer.lookup_transform(
                selected_gate_frame_name,
                other_gate_frame_name,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"Parallel shift for trajectory frames failed because of TF Error: {e}"
            )
            return transform_to_shift

        dx_selected_frame = transform_selected_to_other.transform.translation.x
        dy_selected_frame = transform_selected_to_other.transform.translation.y

        length = math.sqrt(dx_selected_frame**2 + dy_selected_frame**2)

        if length < self.MIN_GATE_SEPARATION_THRESHOLD:
            rospy.logwarn(
                f"Gate links are too close for parallel shift. Skipping shift."
            )
            return transform_to_shift

        unit_dx = dx_selected_frame / length
        unit_dy = dy_selected_frame / length

        shift_x = unit_dx * parallel_offset
        shift_y = unit_dy * parallel_offset

        transform_to_shift.transform.translation.x += shift_x
        transform_to_shift.transform.translation.y += shift_y
        return transform_to_shift

    def create_trajectory_frames(self) -> None:
        """
        Look up the current transforms, compute entrance and exit transforms,
        and broadcast them relative to target_gate_frame.
        """
        try:
            transform_gate_link_1 = self.tf_buffer.lookup_transform(
                self.odom_frame, self.gate_frame_1, rospy.Time(0), rospy.Duration(1)
            )
            transform_gate_link_2 = self.tf_buffer.lookup_transform(
                self.odom_frame, self.gate_frame_2, rospy.Time(0), rospy.Duration(1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        # Extract the translations (still in odom frame)
        gate_link_1_translation = self.get_translation(transform_gate_link_1)
        gate_link_2_translation = self.get_translation(transform_gate_link_2)

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
        entrance_pose, exit_pose = self.compute_entrance_and_exit(
            selected_gate_link_translation, other_gate_link_translation
        )

        # Create TransformStamped messages for entrance and exit
        entrance_transform = self.build_transform_message(
            self.entrance_frame, entrance_pose
        )
        exit_transform = self.build_transform_message(self.exit_frame, exit_pose)

        # Apply parallel shift if offset is significant
        if abs(self.parallel_shift_offset) > 1e-6:
            other_gate_frame_for_shift_direction: str
            if self.target_gate_frame == self.gate_frame_1:
                other_gate_frame_for_shift_direction = self.gate_frame_2
            else:
                other_gate_frame_for_shift_direction = self.gate_frame_1

            entrance_transform = self._shift_transform_parallel_to_gate_line(
                entrance_transform,
                self.target_gate_frame,  # Frame in which entrance_transform.transform is defined
                other_gate_frame_for_shift_direction,
                self.parallel_shift_offset,
                self.tf_buffer,
            )
            exit_transform = self._shift_transform_parallel_to_gate_line(
                exit_transform,
                self.target_gate_frame,  # Frame in which exit_transform.transform is defined
                other_gate_frame_for_shift_direction,
                self.parallel_shift_offset,
                self.tf_buffer,
            )

        self.send_transform(entrance_transform)
        self.send_transform(exit_transform)

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
        transform.header.frame_id = self.target_gate_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation
        return transform

    def send_transform(self, transform: TransformStamped):
        request = SetObjectTransformRequest()
        request.transform = transform
        response = self.set_object_transform_service.call(request)
        if not response.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {response.message}"
            )

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
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
