import numpy as np
from auv_msgs.msg import Pipe  # Assuming Pipe is used directly
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Tuple,
)  # Added Any, Tuple for broader compatibility if needed later
import rospy
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
import math  # For atan2


class Gate:
    """
    Represents a complete or partial slalom gate with left white, red, right white pipes.
    Each attribute is a Pipe or a placeholder if guessed.
    """

    def __init__(
        self, white_left: Pipe, red: Pipe, white_right: Pipe, direction: np.ndarray
    ) -> None:
        self.white_left: Pipe = white_left
        self.red: Pipe = red
        self.white_right: Pipe = white_right
        # The direction vector of the gate (2D unit vector)
        self.direction: np.ndarray = direction  # unit 2D vector along the gate


def create_gate_object(info: Dict[str, Any]) -> Optional["Gate"]:
    """
    Create a Gate object from a validated cluster info dict.
    Requires 'is_complete' to be true and exactly 3 pipes in the order: white, red, white.
    """
    if not info.get("is_complete", False):
        return None
    pipes = info["pipes"]
    if len(pipes) != 3:
        return None
    # By validate_cluster: [white, red, white]
    white_left = pipes[0]
    red = pipes[1]
    white_right = pipes[2]
    _, direction = info["line_model"]
    return Gate(white_left, red, white_right, direction)


def _calculate_orientation(
    gate_direction: np.ndarray,
    waypoint_position: np.ndarray,
    robot_position: np.ndarray,
) -> Quaternion:
    """
    Calculates the orientation (Quaternion) for the waypoint.
    The orientation is normal to the gate and points away from the robot.

    Args:
        gate_direction: A 2D numpy array representing the unit vector along the gate [dx, dy].
        waypoint_position: A 2D numpy array for the waypoint's position [x, y].
        robot_position: A 2D numpy array for the robot's current position [x, y].

    Returns:
            geometry_msgs.msg.Quaternion representing the calculated orientation.
    """
    dx, dy = gate_direction[0], gate_direction[1]

    # the two possible normal vectors
    normal1 = np.array([-dy, dx])
    normal2 = np.array([dy, -dx])

    # Vector from robot to waypoint
    robot_to_waypoint_vector = waypoint_position - robot_position
    # Normalize for dot product comparison, handle zero vector case
    if np.linalg.norm(robot_to_waypoint_vector) > 1e-6:  # a small epsilon
        robot_to_waypoint_vector_normalized = robot_to_waypoint_vector / np.linalg.norm(
            robot_to_waypoint_vector
        )
    else:  # robot is at the waypoint, default to normal1
        robot_to_waypoint_vector_normalized = normal1

    # Determine which normal points "away" from the robot
    # (aligns better with robot_to_waypoint_vector)
    dot_product1 = np.dot(normal1, robot_to_waypoint_vector_normalized)
    dot_product2 = np.dot(normal2, robot_to_waypoint_vector_normalized)

    chosen_normal = normal1 if dot_product1 >= dot_product2 else normal2

    # Calculate yaw angle from the chosen normal vector
    # The chosen_normal is in odom frame, so yaw is relative to odom's X-axis
    yaw = math.atan2(chosen_normal[1], chosen_normal[0])

    # Convert yaw to quaternion (roll=0, pitch=0)
    q = quaternion_from_euler(0, 0, yaw)
    orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return orientation


def compute_navigation_targets(
    gate: "Gate",
    navigation_mode: str,
    robot_pose: Pose,  # Added robot_pose
) -> List[Pose]:
    """
    For a complete Gate, compute the navigation target Pose.
    The target is at the middle of the selected passage pipes.
    Its orientation is normal to the gate direction, pointing away from the robot.
    """

    if navigation_mode == "left":
        pipe_A = gate.white_left
        pipe_B = gate.red
    elif navigation_mode == "right":
        pipe_A = gate.red
        pipe_B = gate.white_right
    else:
        rospy.logwarn_throttle(
            5.0,
            f"Unknown navigation_mode: {navigation_mode}. Defaulting to left.",
        )
        pipe_A = gate.white_left
        pipe_B = gate.red

    pipe_A_pos = np.array([pipe_A.position.x, pipe_A.position.y])
    pipe_B_pos = np.array([pipe_B.position.x, pipe_B.position.y])
    midpoint_pos = (pipe_A_pos + pipe_B_pos) / 2
    waypoint = Pose()
    waypoint.position.x = midpoint_pos[0]
    waypoint.position.y = midpoint_pos[1]
    waypoint.position.z = 0.0

    # Orientation
    robot_position_2d = np.array([robot_pose.position.x, robot_pose.position.y])
    waypoint.orientation = _calculate_orientation(
        gate_direction=gate.direction,
        waypoint_position=midpoint_pos,  # This is the 2D position of the waypoint
        robot_position=robot_position_2d,
    )

    return [waypoint]
