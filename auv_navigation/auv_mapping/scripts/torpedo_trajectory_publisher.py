#!/usr/bin/env python3

import numpy as np

from typing import Tuple
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class TorpedoTransformServiceNode:
    def __init__(self):
        self.enable_approach_view_frames = False
        self.enable_launch_frame = False
        rospy.init_node("create_torpedo_frames_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.torpedo_frame = rospy.get_param("~torpedo_frame", "torpedo_map_link")
        self.torpedo_closer_frame = rospy.get_param(
            "~torpedo_closer_frame", "torpedo_close_approach"
        )
        self.torpedo_big_hole_frame = rospy.get_param(
            "~torpedo_big_hole_frame", "torpedo_big_hole_link"
        )
        self.torpedo_medium_hole_frame = rospy.get_param(
            "~torpedo_medium_hole_frame", "torpedo_medium_hole_link"
        )
        self.torpedo_front_view_frame = rospy.get_param(
            "~torpedo_front_view_frame", "torpedo_front_view"
        )
        self.torpedo_launch_frame = rospy.get_param(
            "~torpedo_launch_frame", "torpedo_launch"
        )

        self.closer_frame_distance = rospy.get_param("~closer_frame_distance", 2.5)
        self.front_view_distance = rospy.get_param("~front_view_distance", 2.5)
        self.launch_distance = rospy.get_param("~launch_distance", 0.5)

        # Service for enabling/disabling close approach and front view frames
        self.set_approach_view_service = rospy.Service(
            "toggle_torpedo_approach_view_frames",
            SetBool,
            self.handle_approach_view_service,
        )
        # Service for enabling/disabling launch frame
        self.set_launch_service = rospy.Service(
            "toggle_torpedo_launch_frame", SetBool, self.handle_launch_service
        )

    def get_pose(self, transform: TransformStamped) -> Pose:
        """Converts a TransformStamped message to a Pose message."""
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        """Builds a TransformStamped message from a child frame ID and a Pose."""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform: TransformStamped):
        """Sends a transform using the set_object_transform service."""
        req = SetObjectTransformRequest()
        req.transform = transform
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(
                    f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for {transform.child_frame_id}: {e}")

    def create_approach_view_frames(self):
        """Creates and publishes the torpedo_close_approach and torpedo_front_view frames."""
        # --- Torpedo Closer Frame (torpedo_close_approach) ---
        try:
            transform_torpedo = self.tf_buffer.lookup_transform(
                self.odom_frame, self.torpedo_frame, rospy.Time(0), rospy.Duration(1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup for {self.torpedo_frame} failed: {e}")
            return  # Cannot create torpedo_close without torpedo_map_link

        torpedo_pose = self.get_pose(transform_torpedo)
        torpedo_pos = np.array(
            [torpedo_pose.position.x, torpedo_pose.position.y, torpedo_pose.position.z]
        )

        try:
            transform_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(1)
            )
            robot_pose = self.get_pose(transform_robot)
            robot_pos = np.array(
                [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"TF lookup for {self.robot_frame} failed, using torpedo's Z for closer frame: {e}"
            )
            robot_pos = np.array([0, 0, torpedo_pos[2]])  # Fallback for Z

        # Direction from robot to torpedo_map_link
        direction_robot_to_torpedo_2d = torpedo_pos[:2] - robot_pos[:2]
        total_distance_2d = np.linalg.norm(direction_robot_to_torpedo_2d)

        if total_distance_2d == 0:
            rospy.logwarn(
                "Robot and torpedo are at the same XY position! Cannot determine direction for torpedo_close_approach."
            )
            return

        direction_unit_robot_to_torpedo_2d = (
            direction_robot_to_torpedo_2d / total_distance_2d
        )

        # For torpedo_close_approach, we want it to be *behind* the torpedo_map_link relative to the robot,
        # and looking *towards* torpedo_map_link.
        # So, its position is torpedo_pos - (unit vector from robot to torpedo) * distance
        closer_pos_2d = torpedo_pos[:2] - (
            direction_unit_robot_to_torpedo_2d * self.closer_frame_distance
        )
        closer_pos = np.append(closer_pos_2d, torpedo_pos[2])  # Use torpedo's Z

        # The yaw for torpedo_close_approach should point towards torpedo_map_link
        # This is the direction_unit_robot_to_torpedo_2d itself
        closer_yaw = np.arctan2(
            direction_unit_robot_to_torpedo_2d[1], direction_unit_robot_to_torpedo_2d[0]
        )
        closer_q = tf.transformations.quaternion_from_euler(0, 0, closer_yaw)

        closer_orientation = Pose().orientation
        closer_orientation.x = closer_q[0]
        closer_orientation.y = closer_q[1]
        closer_orientation.z = closer_q[2]
        closer_orientation.w = closer_q[3]

        closer_pose = Pose()
        closer_pose.position.x, closer_pose.position.y, closer_pose.position.z = (
            closer_pos
        )
        closer_pose.orientation = closer_orientation

        closer_transform = self.build_transform_message(
            self.torpedo_closer_frame, closer_pose
        )
        self.send_transform(closer_transform)

        # --- Torpedo Front View Frame ---
        try:
            transform_big_hole = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_big_hole_frame,
                rospy.Time(0),
                rospy.Duration(1),
            )
            transform_medium_hole = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_medium_hole_frame,
                rospy.Time(0),
                rospy.Duration(1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"TF lookup for big/medium hole failed, cannot create front view frame: {e}"
            )
            return  # Cannot proceed with this frame without hole links

        big_hole_pose = self.get_pose(transform_big_hole)
        medium_hole_pose = self.get_pose(transform_medium_hole)

        big_hole_pos = np.array(
            [
                big_hole_pose.position.x,
                big_hole_pose.position.y,
                big_hole_pose.position.z,
            ]
        )
        medium_hole_pos = np.array(
            [
                medium_hole_pose.position.x,
                medium_hole_pose.position.y,
                medium_hole_pose.position.z,
            ]
        )

        big_to_medium_vector_2d = medium_hole_pos[:2] - big_hole_pos[:2]

        if np.linalg.norm(big_to_medium_vector_2d) == 0:
            rospy.logwarn(
                "Big hole and medium hole are at the same XY position! Cannot define torpedo_front_view."
            )
            return

        # Perpendicular vector to big_hole_link -> medium_hole_link
        # We need this vector to point *away* from the line connecting the holes,
        # in the general direction the robot would face the torpedo board.
        perp_vector_candidate_1 = np.array(
            [-big_to_medium_vector_2d[1], big_to_medium_vector_2d[0]]
        )
        perp_vector_candidate_2 = -perp_vector_candidate_1

        # Determine which perpendicular direction is "front" by comparing with robot's relative position
        # Assuming robot approaches from the "front" of the torpedo board
        if np.linalg.norm(
            robot_pos[:2] - (big_hole_pos[:2] + perp_vector_candidate_1)
        ) < np.linalg.norm(
            robot_pos[:2] - (big_hole_pos[:2] + perp_vector_candidate_2)
        ):
            torpedo_front_direction_2d = perp_vector_candidate_1
        else:
            torpedo_front_direction_2d = perp_vector_candidate_2

        torpedo_front_unit_2d = torpedo_front_direction_2d / np.linalg.norm(
            torpedo_front_direction_2d
        )

        # Calculate torpedo_front_view frame
        # Position is relative to torpedo_map_link, moved *forward* by front_view_distance
        torpedo_front_view_pos_2d = torpedo_pos[:2] + (
            torpedo_front_unit_2d * self.front_view_distance
        )
        torpedo_front_view_pos = np.append(torpedo_front_view_pos_2d, torpedo_pos[2])

        # Orientation for torpedo_front_view: looking *towards* torpedo_map_link
        # So, the direction vector for orientation is -torpedo_front_unit_2d
        front_view_orientation_vector_2d = -torpedo_front_unit_2d
        torpedo_front_view_yaw = np.arctan2(
            front_view_orientation_vector_2d[1], front_view_orientation_vector_2d[0]
        )
        torpedo_front_view_q = tf.transformations.quaternion_from_euler(
            0, 0, torpedo_front_view_yaw
        )

        torpedo_front_view_orientation = Pose().orientation
        torpedo_front_view_orientation.x = torpedo_front_view_q[0]
        torpedo_front_view_orientation.y = torpedo_front_view_q[1]
        torpedo_front_view_orientation.z = torpedo_front_view_q[2]
        torpedo_front_view_orientation.w = torpedo_front_view_q[3]

        torpedo_front_view_pose = Pose()
        (
            torpedo_front_view_pose.position.x,
            torpedo_front_view_pose.position.y,
            torpedo_front_view_pose.position.z,
        ) = torpedo_front_view_pos
        torpedo_front_view_pose.orientation = torpedo_front_view_orientation

        torpedo_front_view_transform = self.build_transform_message(
            self.torpedo_front_view_frame, torpedo_front_view_pose
        )
        self.send_transform(torpedo_front_view_transform)

    def create_launch_frame(self):
        """Creates and publishes the torpedo_launch frame."""
        try:
            transform_big_hole = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_big_hole_frame,
                rospy.Time(0),
                rospy.Duration(1),
            )
            transform_medium_hole = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.torpedo_medium_hole_frame,
                rospy.Time(0),
                rospy.Duration(1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"TF lookup for big/medium hole failed, cannot create launch frame: {e}"
            )
            return  # Cannot proceed with this frame without hole links

        big_hole_pose = self.get_pose(transform_big_hole)
        medium_hole_pose = self.get_pose(transform_medium_hole)

        big_hole_pos = np.array(
            [
                big_hole_pose.position.x,
                big_hole_pose.position.y,
                big_hole_pose.position.z,
            ]
        )
        medium_hole_pos = np.array(
            [
                medium_hole_pose.position.x,
                medium_hole_pose.position.y,
                medium_hole_pose.position.z,
            ]
        )
        try:
            transform_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(1)
            )
            robot_pose = self.get_pose(transform_robot)
            robot_pos = np.array(
                [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"TF lookup for {self.robot_frame} failed, assuming robot_pos for determining 'front' direction: {e}"
            )
            # Fallback if robot_frame is not available, try to infer 'front' from hole positions
            # This is a less robust fallback, but better than nothing
            robot_pos = np.array(
                [0, 0, big_hole_pos[2]]
            )  # Placeholder, will be corrected below if needed

        big_to_medium_vector_2d = medium_hole_pos[:2] - big_hole_pos[:2]

        if np.linalg.norm(big_to_medium_vector_2d) == 0:
            rospy.logwarn(
                "Big hole and medium hole are at the same XY position! Cannot define torpedo_launch."
            )
            return

        # Perpendicular vector to big_hole_link -> medium_hole_link
        perp_vector_candidate_1 = np.array(
            [-big_to_medium_vector_2d[1], big_to_medium_vector_2d[0]]
        )
        perp_vector_candidate_2 = -perp_vector_candidate_1

        # Determine which perpendicular direction is "front" based on robot's position
        # This assumes the robot approaches from the "front" side of the torpedo board
        if np.linalg.norm(
            robot_pos[:2] - (big_hole_pos[:2] + perp_vector_candidate_1)
        ) < np.linalg.norm(
            robot_pos[:2] - (big_hole_pos[:2] + perp_vector_candidate_2)
        ):
            torpedo_front_direction_2d = perp_vector_candidate_1
        else:
            torpedo_front_direction_2d = perp_vector_candidate_2

        torpedo_front_unit_2d = torpedo_front_direction_2d / np.linalg.norm(
            torpedo_front_direction_2d
        )

        # Calculate torpedo_launch frame
        # Position is relative to big_hole_link, moved *forward* by launch_distance
        torpedo_launch_pos_2d = big_hole_pos[:2] + (
            torpedo_front_unit_2d * self.launch_distance
        )
        torpedo_launch_pos = np.append(torpedo_launch_pos_2d, big_hole_pos[2])

        # Orientation for torpedo_launch: looking *towards* torpedo_big_hole_link
        # So, the direction vector for orientation is -torpedo_front_unit_2d
        launch_orientation_vector_2d = -torpedo_front_unit_2d
        torpedo_launch_yaw = np.arctan2(
            launch_orientation_vector_2d[1], launch_orientation_vector_2d[0]
        )
        torpedo_launch_q = tf.transformations.quaternion_from_euler(
            0, 0, torpedo_launch_yaw
        )

        torpedo_launch_orientation = Pose().orientation
        torpedo_launch_orientation.x = torpedo_launch_q[0]
        torpedo_launch_orientation.y = torpedo_launch_q[1]
        torpedo_launch_orientation.z = torpedo_launch_q[2]
        torpedo_launch_orientation.w = torpedo_launch_q[3]

        torpedo_launch_pose = Pose()
        (
            torpedo_launch_pose.position.x,
            torpedo_launch_pose.position.y,
            torpedo_launch_pose.position.z,
        ) = torpedo_launch_pos
        torpedo_launch_pose.orientation = torpedo_launch_orientation

        torpedo_launch_transform = self.build_transform_message(
            self.torpedo_launch_frame, torpedo_launch_pose
        )
        self.send_transform(torpedo_launch_transform)

    def handle_approach_view_service(self, req: SetBoolRequest) -> SetBoolResponse:
        """Callback for the toggle_torpedo_approach_view_frames service."""
        self.enable_approach_view_frames = req.data
        message = f"Torpedo 'close approach' and 'front view' frames publish is set to: {self.enable_approach_view_frames}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_launch_service(self, req: SetBoolRequest) -> SetBoolResponse:
        """Callback for the toggle_torpedo_launch_frame service."""
        self.enable_launch_frame = req.data
        message = (
            f"Torpedo 'launch' frame publish is set to: {self.enable_launch_frame}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        """Main loop of the ROS node."""
        rate = rospy.Rate(5.0)  # Publish rate in Hz
        while not rospy.is_shutdown():
            if self.enable_approach_view_frames:
                self.create_approach_view_frames()
            if self.enable_launch_frame:
                self.create_launch_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TorpedoTransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
