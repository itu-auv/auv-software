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

    def send_goal(self, goal_pose_or_frame):
        # If given a TF frame, convert to PoseStamped
        if isinstance(goal_pose_or_frame, str):
            rospy.logdebug(
                f"[MoveBaseClient] Received TF frame '{goal_pose_or_frame}' as goal. Attempting to look up transform to '{self.map_frame}'."
            )
            goal_pose = get_pose_from_transform(
                self.tf_buffer, goal_pose_or_frame, self.map_frame
            )
            if goal_pose is None:
                return False, "TF lookup failed", None

        elif isinstance(goal_pose_or_frame, PoseStamped):
            goal_pose = goal_pose_or_frame
        else:
            rospy.logerr(
                "[MoveBaseClient] Invalid goal type. Must be PoseStamped or TF frame name (str)."
            )
            return False, "Invalid goal type", None

        # Build the goal
        goal = MoveBaseGoal()
        goal.target_pose = goal_pose

        rospy.logdebug(
            f"[MoveBaseClient] Sending goal to {goal.target_pose.header.frame_id}: {goal.target_pose.pose.position}"
        )
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        outcome = getattr(result, "outcome", -1)
        message = getattr(result, "message", "")
        # Outcomes: 0=SUCCESS, see mbf_msgs/MoveBaseResult for others
        return outcome == 0, message, result


"""class ExePathClient:
    def __init__(self, server_name='/taluy/move_base_flex/exe_path', map_frame="odom"):
        self.client = actionlib.SimpleActionClient(server_name, ExePathAction)
        rospy.logdebug(f"[ExePathClient] Waiting for action server {server_name}...")
        self.client.wait_for_server()
        rospy.logdebug(f"[ExePathClient] Connected to {server_name}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_frame = map_frame

    def send_goal(self, goal_pose_or_frame):
        if isinstance(goal_pose_or_frame, str):
            rospy.logdebug(f"[ExePathClient] Received TF frame '{goal_pose_or_frame}' as goal. Attempting to look up transform to '{self.map_frame}'.")
            goal_pose = get_pose_from_transform(self.tf_buffer, goal_pose_or_frame, self.map_frame)
            if goal_pose is None:
                return False, "TF lookup failed", None
        elif isinstance(goal_pose_or_frame, PoseStamped):
            goal_pose = goal_pose_or_frame
        else:
            rospy.logerr("[ExePathClient] Invalid goal type. Must be PoseStamped or TF frame name (str).")
            return False, "Invalid goal type", None

        goal = ExePathGoal()
        goal.target_pose = goal_pose

        rospy.logdebug(f"[ExePathClient] Sending goal to {goal.target_pose.header.frame_id}: {goal.target_pose.pose.position}")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        outcome = getattr(result, 'outcome', -1)
        message = getattr(result, 'message', '')
        return outcome == 0, message, result """
