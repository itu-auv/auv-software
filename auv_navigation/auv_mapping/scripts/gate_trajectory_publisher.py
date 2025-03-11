#!/usr/bin/env python3
from typing import Tuple
import math
import rospy
import tf2_ros
import tf_conversions
import geometry_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class TransformServiceNode:
    def __init__(self):
        self.enable = False
        rospy.init_node("create_gate_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # this service is used to broadcast transforms
        self.set_object_transform_service = rospy.ServiceProxy(
            "map/set_object_transform", SetObjectTransform
        )
        # wait indefinitely for the service to be available
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.entrance_frame = "gate_entrance"
        self.exit_frame = "gate_exit"

        # gate links
        self.gate_frame_1 = rospy.get_param("~gate_frame_1", "gate_blue_arrow_link")
        self.gate_frame_2 = rospy.get_param("~gate_frame_2", "gate_red_arrow_link")
        self.set_enable_service = rospy.Service(
            "set_transform_gate_trajectory", SetBool, self.handle_enable_service
        )

        # Offset distance for the outward trajectory frame. The value is added to the gate frame.
        self.offset_outward = rospy.get_param("~offset_outward", 1.0)
        # Offset distance for the inward trajectory frame. The value is subtracted from the gate frame.
        self.offset_inward = rospy.get_param("~offset_inward", 1.7)

        # TODO: This shouldn't be hardcoded.
        self.selected_gate_frame = self.gate_frame_1

    def assign_selected_gate_translations(
        self,
        gate_link_1_translation: Tuple[float, float, float],
        gate_link_2_translation: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Determines selected_gate_link and other_gate_link based on self.selected_gate_frame.

        Returns:
            tuple: (selected_gate_link_translation, other_gate_link_translation)
        """
        if self.selected_gate_frame == self.gate_frame_1:
            selected_gate_link_translation = gate_link_1_translation
            other_gate_link_translation = gate_link_2_translation
        elif self.selected_gate_frame == self.gate_frame_2:
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
        Compute entrance and exit transforms based on the positions of the selected_gate_link and other_gate_link

        1. Define a perpendicular unit vector to the line between the two gate links.
        2. Calculate the two shifted positions:
            - One outward in the perpendicular line (shifted_position_outward).
            - One inward in the perpendicular line (shifted_position_inward).
        3. Determine which shifted position is closer to the world origin, and assign it as the entrance.
            The other shifted position is assigned as the exit.
        """

        MIN_GATE_SEPARATION_THRESHOLD = 0.15

        dx = other_gate_link_translation[0] - selected_gate_link_translation[0]
        dy = other_gate_link_translation[1] - selected_gate_link_translation[1]

        length = math.sqrt(dx**2 + dy**2)
        if length < MIN_GATE_SEPARATION_THRESHOLD:
            raise ValueError(
                "The gate links are almost identical or at the same position"
            )

        # Perpendicular unit vector to the line connecting selected_gate_link and other_gate_link frames.
        # For D = (dx, dy), a perpendicular vector is N = (-dy, dx).
        unit_perpendicular_x = -dy / length
        unit_perpendicular_y = dx / length

        # Calculate new positions by applying the perpendicular offsets.
        shifted_position_outward: Tuple[float, float, float] = (
            selected_gate_link_translation[0]
            + unit_perpendicular_x * self.offset_outward,
            selected_gate_link_translation[1]
            + unit_perpendicular_y * self.offset_outward,
            selected_gate_link_translation[2],
        )

        shifted_position_inward: Tuple[float, float, float] = (
            selected_gate_link_translation[0]
            - unit_perpendicular_x * self.offset_inward,
            selected_gate_link_translation[1]
            - unit_perpendicular_y * self.offset_inward,
            selected_gate_link_translation[2],
        )
        # Calculate orientations so that the frames both look at the selected gate link.
        angle_outward = math.atan2(
            selected_gate_link_translation[1] - shifted_position_outward[1],
            selected_gate_link_translation[0] - shifted_position_outward[0],
        )
        angle_inward = math.atan2(
            selected_gate_link_translation[1] - shifted_position_inward[1],
            selected_gate_link_translation[0] - shifted_position_inward[0],
        )
        shifted_quat_outward = tf_conversions.transformations.quaternion_from_euler(
            0, 0, angle_outward
        )
        shifted_quat_inward = tf_conversions.transformations.quaternion_from_euler(
            0, 0, angle_inward
        )

        # Determine which computed position is closer to the world origin.
        distance_to_shifted_outward = math.sqrt(
            shifted_position_outward[0] ** 2 + shifted_position_outward[1] ** 2
        )
        distance_to_shifted_inward = math.sqrt(
            shifted_position_inward[0] ** 2 + shifted_position_inward[1] ** 2
        )

        # Whichever is closer to odom is the entrance
        # The other is the exit
        if distance_to_shifted_outward < distance_to_shifted_inward:
            entrance_position, entrance_quat = (
                shifted_position_outward,
                shifted_quat_outward,
            )
            exit_position, exit_quat = (
                shifted_position_inward,
                shifted_quat_inward,
            )
        else:
            entrance_position, entrance_quat = (
                shifted_position_inward,
                shifted_quat_inward,
            )
            exit_position, exit_quat = (shifted_position_outward, shifted_quat_outward)

        # Finally, create the entrance and exit poses
        entrance_pose = Pose()
        entrance_pose.position = Point(*entrance_position)
        entrance_pose.orientation = Quaternion(*entrance_quat)

        exit_pose = Pose()
        exit_pose.position = Point(*exit_position)
        exit_pose.orientation = Quaternion(*exit_quat)

        return entrance_pose, exit_pose

    def create_trajectory_frames(self) -> None:
        """
        Look up the current transforms, compute entrance and exit transforms, and send them.
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

        # Extract the translations
        gate_link_1_translation = self.get_translation(transform_gate_link_1)
        gate_link_2_translation = self.get_translation(transform_gate_link_2)

        # Assign selected and other gate link translations
        selected_gate_link_translation, other_gate_link_translation = (
            self.assign_selected_gate_translations(
                gate_link_1_translation, gate_link_2_translation
            )
        )

        # Compute the entrance and exit frames from selected and other gate link translations
        entrance_pose, exit_pose = self.compute_entrance_and_exit(
            selected_gate_link_translation, other_gate_link_translation
        )

        # Create TransformStamped messages for entrance and exit
        entrance_transform = self.build_transform_message(
            self.entrance_frame, entrance_pose
        )
        exit_transform = self.build_transform_message(self.exit_frame, exit_pose)

        # Send transforms using the service
        self.send_transform(entrance_transform)
        self.send_transform(exit_transform)

    ##### ---- Transform related ---- #####
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

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def handle_enable_service(self, req):
        self.enable = req.data
        message = f"Gate trajectory transform publish is set to: {self.enable}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.enable:
                self.create_trajectory_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TransformServiceNode()
        node.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
