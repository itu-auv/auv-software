#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
from mbf_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from typing import Optional


def send_goal(
    x: float,
    y: float,
    z: float = 0.0,
    frame_id: str = "odom",
    planner: Optional[str] = None,
    controller: Optional[str] = None,
) -> None:
    """
    Sends a navigation goal to the move_base_flex action server.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        z: Target Z coordinate (default: 0.0).
        frame_id: The frame ID for the goal pose (default: "odom").
        planner: The name of the planner plugin to use (optional).
        controller: The name of the controller plugin to use (optional).
    """
    # Create an action client called "move_base_flex" with action definition file "MoveBaseAction"
    # Assumes the node is launched within the "taluy" namespace
    client = actionlib.SimpleActionClient(
        "/taluy/move_base_flex/move_base", MoveBaseAction
    )

    rospy.loginfo("Waiting for move_base_flex action server...")
    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()
    rospy.loginfo("Action server found.")

    # Creates a goal to send to the action server.
    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = frame_id
    goal.target_pose.header.stamp = rospy.Time.now()

    # Set the goal pose
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = z
    # Simple orientation: facing forward along the frame's X-axis
    goal.target_pose.pose.orientation.w = 1.0

    # Optionally specify planner and controller
    if planner:
        goal.planner = planner
    if controller:
        goal.controller = controller

    rospy.loginfo(f"Sending goal: Position(x={x}, y={y}, z={z}) in '{frame_id}' frame")
    if planner:
        rospy.loginfo(f"Using planner: {planner}")
    if controller:
        rospy.loginfo(f"Using controller: {controller}")

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    rospy.loginfo("Waiting for result...")
    client.wait_for_result()

    # Prints out the result of executing the action
    state = client.get_state()
    result = client.get_result()

    rospy.loginfo(f"Action finished.")
    rospy.loginfo(f"Final state: {actionlib.GoalStatus.to_string(state)}")
    if result:
        rospy.loginfo(f"Outcome: {result.outcome}")
        rospy.loginfo(f"Message: {result.message}")
    else:
        rospy.loginfo("No result message received.")


if __name__ == "__main__":
    try:
        rospy.init_node("mbf_goal_sender_client")

        # --- Define your goal here ---
        target_x = 2.0
        target_y = 0.0
        target_z = 0.0  # Keep Z=0 for typical 2D/2.5D navigation
        target_frame = "odom"  # Should match your global_frame in costmaps
        target_planner = "carrot_planner"  # From your move_base_flex.yaml
        target_controller = "teb_local"  # From your move_base_flex.yaml
        # -----------------------------

        send_goal(
            target_x,
            target_y,
            target_z,
            target_frame,
            target_planner,
            target_controller,
        )

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test interrupted.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
