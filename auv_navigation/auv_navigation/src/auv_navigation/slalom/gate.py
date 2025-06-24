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
from geometry_msgs.msg import Pose


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


def compute_navigation_targets(
    gate: "Gate",
    navigation_mode: str,
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
    waypoint.position.z = (
        0.0  # Assuming 2D navigation for slalom, Z can be refined later
    )

    # Orientation
    # TODO: Calculate proper orientation: normal to the gate (gate.direction),
    # and pointing "away" from the robot.
    # gate.direction is a 2D unit vector [dx, dy]. A normal could be (-dy, dx) or (dy, -dx).
    # The "away" part needs the robot's current pose relative to the gate to determine.
    # For now, using a placeholder (identity quaternion: no rotation).
    waypoint.orientation.x = 0.0
    waypoint.orientation.y = 0.0
    waypoint.orientation.z = 0.0
    waypoint.orientation.w = 1.0

    return [waypoint]  # Return a list containing the single Pose target
