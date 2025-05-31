#!/usr/bin/env python3
import rospy
import actionlib
from mbf_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from typing import Optional


def get_pose_from_transform(
    tf_buffer: tf2_ros.Buffer, frame_id: str, map_frame: str
) -> Optional[PoseStamped]:
    """Convert a TF frame to a PoseStamped message relative to map_frame.

    Args:
        tf_buffer: TF2 buffer to use for transform lookup
        frame_id: Target frame to convert to pose
        map_frame: Reference frame to express the pose in

    Returns:
        PoseStamped message or None if transform lookup fails
    """
    try:
        transform_stamped = tf_buffer.lookup_transform(
            map_frame, frame_id, rospy.Time(0), rospy.Duration(1.0)
        )
        pose_stamped = PoseStamped()
        pose_stamped.header = transform_stamped.header  # map frame
        pose_stamped.pose.position.x = transform_stamped.transform.translation.x
        pose_stamped.pose.position.y = transform_stamped.transform.translation.y
        pose_stamped.pose.position.z = transform_stamped.transform.translation.z
        pose_stamped.pose.orientation = transform_stamped.transform.rotation
        return pose_stamped
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        rospy.logerr(
            f"[MBF] TF lookup failed for frame '{frame_id}' to '{map_frame}': {e}"
        )
        return None


class MoveBaseClient:
    def __init__(self, server_name="/taluy/move_base_flex/move_base", map_frame="odom"):

        self.client = actionlib.SimpleActionClient(server_name, MoveBaseAction)
        rospy.logdebug(f"[MoveBaseClient] Waiting for action server {server_name}...")
        self.client.wait_for_server()
        rospy.logdebug(f"[MoveBaseClient] Connected to {server_name}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_frame = map_frame

    def send_waypoints(self, goal_poses_or_frames: list):
        """
        Sends a list of waypoints (PoseStamped objects or TF frame names) sequentially
        to the MBF action server.

        Args:
            goal_poses_or_frames: A list where each item can either be a
                                  geometry_msgs/PoseStamped message or a string
                                  representing a TF frame name.

        Returns:
            A tuple (all_successful, final_message, list_of_results)
            all_successful (bool): True if all waypoints were reached, False otherwise.
            final_message (str): Message detailing the outcome (success or first failure).
            list_of_results (list): List of result objects from each waypoint attempt.
                                    Contains None for waypoints that failed before sending
                                    (e.g. TF lookup failure or invalid type).
        """
        # TODO Allow single Goal objects that are not in a list
        if not isinstance(goal_poses_or_frames, list):
            rospy.logerr(
                "[MoveBaseClient] Invalid input: goal_poses_or_frames must be a list."
            )
            return False, "Input must be a list", []

        if not goal_poses_or_frames:
            rospy.logwarn("[MoveBaseClient] Received an empty list of waypoints.")
            return True, "No waypoints to process", []

        rospy.loginfo(
            f"[MoveBaseClient] Received {len(goal_poses_or_frames)} waypoints to process."
        )
        all_results = []
        final_message = "All waypoints reached successfully."
        all_successful = True

        # Process each waypoint in the list
        for i, goal_item in enumerate(goal_poses_or_frames):
            rospy.loginfo(
                f"[MoveBaseClient] Processing waypoint {i+1}/{len(goal_poses_or_frames)}: {goal_item}"
            )
            current_goal_pose = None

            if isinstance(
                goal_item, str
            ):  # If given a TF frame name, convert to PoseStamped
                rospy.logdebug(
                    f"[MoveBaseClient] Waypoint {i+1} is a TF frame '{goal_item}'. Attempting to look up transform to '{self.map_frame}'."
                )
                current_goal_pose = get_pose_from_transform(
                    self.tf_buffer, goal_item, self.map_frame
                )
                if current_goal_pose is None:
                    message = f"TF lookup failed for waypoint {i+1} ('{goal_item}')"
                    all_results.append(None)
                    all_successful = False
                    final_message = message
                    break  # Abort sequence if TF lookup fails
            elif isinstance(goal_item, PoseStamped):
                current_goal_pose = goal_item
            else:
                message = f"Invalid type for waypoint {i+1}. Must be PoseStamped or TF frame name (str). Type was: {type(goal_item)}"
                rospy.logerr(f"[MoveBaseClient] {message}")
                all_results.append(None)
                all_successful = False
                final_message = message
                break  # Abort sequence if invalid type

            goal = MoveBaseGoal()
            goal.target_pose = current_goal_pose

            rospy.logdebug(
                f"[MoveBaseClient] Sending waypoint {i+1} to {goal.target_pose.header.frame_id}: {goal.target_pose.pose.position.x}, {goal.target_pose.pose.position.y}"
            )
            self.client.send_goal(goal)
            self.client.wait_for_result()
            result = self.client.get_result()
            all_results.append(result)

            outcome = getattr(result, "outcome", -1)  # MBF outcome
            message_from_server = getattr(result, "message", "")

            if outcome == 0:  # Means SUCCESS (see MoveBaseResult.msg)
                rospy.loginfo(
                    f"[MoveBaseClient] Successfully reached waypoint {i+1}. Message: {message_from_server}"
                )
            else:
                rospy.logerr(
                    f"[MoveBaseClient] Failed to reach waypoint {i+1}. Outcome: {outcome}, Message: {message_from_server}"
                )
                all_successful = False
                final_message = f"Failed at waypoint {i+1} (Outcome: {outcome}): {message_from_server}"
                break  # Abort sequence

        if all_successful:
            rospy.loginfo("[MoveBaseClient] All waypoints processed successfully.")
        else:
            rospy.logerr(
                f"[MoveBaseClient] Waypoint sequence failed. Last message: {final_message}"
            )

        return all_successful, final_message, all_results


""" class ExeGoalClient:
    def __init__(self, server_name='/taluy/move_base_flex/exe_goal', map_frame="odom"):
        self.client = actionlib.SimpleActionClient(server_name, ExeGoalAction)
        rospy.logdebug(f"[ExeGoalClient] Waiting for action server {server_name}...")
        self.client.wait_for_server()
        rospy.logdebug(f"[ExeGoalClient] Connected to {server_name}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_frame = map_frame

    def send_goal(self, goal_pose_or_frame):
        if isinstance(goal_pose_or_frame, str):
            rospy.logdebug(f"[ExeGoalClient] Received TF frame '{goal_pose_or_frame}' as goal. Attempting to look up transform to '{self.map_frame}'.")
            goal_pose = get_pose_from_transform(self.tf_buffer, goal_pose_or_frame, self.map_frame)
            if goal_pose is None:
                return False, "TF lookup failed", None
        elif isinstance(goal_pose_or_frame, PoseStamped):
            goal_pose = goal_pose_or_frame
        else:
            rospy.logerr("[ExePathClient] Invalid goal type. Must be PoseStamped or TF frame name (str).")
            return False, "Invalid goal type", None

        goal = ExeGoalGoal()
        goal.target_pose = goal_pose

        rospy.logdebug(f"[ExePathClient] Sending goal to {goal.target_pose.header.frame_id}: {goal.target_pose.pose.position}")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        outcome = getattr(result, 'outcome', -1)
        message = getattr(result, 'message', '')
        return outcome == 0, message, result
"""
