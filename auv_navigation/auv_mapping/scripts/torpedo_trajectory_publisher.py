#!/usr/bin/env python3

import numpy as np

from typing import Tuple
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class TorpedoTransformServiceNode:
    def __init__(self):
        # Enable states for different frames
        self.enable_launch = False
        self.enable_other_frames = False
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

        # Create separate services for different frames
        self.set_launch_frame_service = rospy.Service(
            "toggle_torpedo_launch_frame", SetBool, self.handle_launch_frame_service
        )
        self.set_other_frames_service = rospy.Service(
            "toggle_torpedo_other_frames", SetBool, self.handle_other_frames_service
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
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(
                    f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def create_torpedo_frames(self):
        # --- Torpedo Closer Frame (torpedo_close) ---
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
                "Robot and torpedo are at the same XY position! Cannot determine direction for torpedo_close."
            )
            return

        direction_unit_robot_to_torpedo_2d = (
            direction_robot_to_torpedo_2d / total_distance_2d
        )

        # For torpedo_close, we want it to be *behind* the torpedo_map_link relative to the robot,
        # and looking *towards* torpedo_map_link.
        # So, its position is torpedo_pos - (unit vector from robot to torpedo) * distance
        closer_pos_2d = torpedo_pos[:2] - (
            direction_unit_robot_to_torpedo_2d * self.closer_frame_distance
        )
        closer_pos = np.append(closer_pos_2d, torpedo_pos[2])  # Use torpedo's Z

        # The yaw for torpedo_close should point towards torpedo_map_link
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
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup for {self.torpedo_big_hole_frame} failed: {e}")
            return  # Cannot create front view frame without big_hole frame

        big_hole_pose = self.get_pose(transform_big_hole)
        big_hole_pos = np.array(
            [
                big_hole_pose.position.x,
                big_hole_pose.position.y,
                big_hole_pose.position.z,
            ]
        )

        # --- Torpedo Medium Hole Frame (torpedo_medium) ---
        try:
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
            rospy.logwarn(f"TF lookup for {self.torpedo_medium_hole_frame} failed: {e}")
            return  # Cannot create front view frame without medium_hole frame

        medium_hole_pose = self.get_pose(transform_medium_hole)
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
                "Big hole and medium hole are at the same XY position! Cannot define torpedo_front."
            )
            return

        # Perpendicular vector to big_hole_link -> medium_hole_link
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

    def handle_launch_frame_service(self, req):
        self.enable_launch = req.data
        message = (
            f"Torpedo launch frame transform publish is set to: {self.enable_launch}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_other_frames_service(self, req):
        self.enable_other_frames = req.data
        message = f"Torpedo other frames (closer, front view) transform publish is set to: {self.enable_other_frames}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if self.enable_launch:
                self.create_launch_frame()
            if self.enable_other_frames:
                self.create_other_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TorpedoTransformServiceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
