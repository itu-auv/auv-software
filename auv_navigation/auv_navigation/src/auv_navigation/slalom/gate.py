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
        # The left white pipe (from robot's perspective)
        self.white_left: Pipe = white_left
        # The red pipe (center of the gate)
        self.red: Pipe = red
        # The right white pipe (from robot's perspective)
        self.white_right: Pipe = white_right
        # The direction vector of the gate (2D unit vector)
        self.direction: np.ndarray = direction  # unit 2D vector along the gate


def create_gate_object(info: Dict[str, Any]) -> Optional["Gate"]:
    """
    Create a Gate object from a validated cluster info dict.
    Requires 'is_complete' to be true and exactly 3 pipes in the order: white, red, white.
    """

    if not info.get("is_complete", False):
        rospy.logdebug("[CreateGate] Cluster not complete.")
        return None

    pipes: List[Pipe] = info["pipes"]
    if len(pipes) != 3:
        rospy.logdebug(
            f"[CreateGate] Expected 3 pipes for a complete gate, got {len(pipes)}."
        )
        return None

    # Assuming pipes are sorted: White (left), Red (middle), White (right)
    white_left = pipes[0]
    red = pipes[1]
    white_right = pipes[2]

    if not (
        white_left.color == "white"
        and red.color == "red"
        and white_right.color == "white"
    ):
        rospy.logwarn_throttle(
            5.0,
            f"[CreateGate] Pipe colors do not match W-R-W for a complete gate. Got: {[p.color for p in pipes]}. This should have been caught by validate_cluster if is_complete is True.",
        )
        return None

    _, direction = info["line_model"]  # line_model is (origin_point, unit_direction)
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

    pipe_A: Optional[Pipe] = None
    pipe_B: Optional[Pipe] = None

    # Determine which pipes to use based on navigation mode
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

    if pipe_A is None or pipe_B is None:  # Should not happen if Gate object is valid
        rospy.logerr("[ComputeNavTargets] Invalid pipes for navigation mode.")
        return []

    pipe_A_pos = np.array([pipe_A.position.x, pipe_A.position.y])
    pipe_B_pos = np.array([pipe_B.position.x, pipe_B.position.y])
    midpoint_pos = (pipe_A_pos + pipe_B_pos) / 2.0

    waypoint = Pose()
    waypoint.position.x = midpoint_pos[0]
    waypoint.position.y = midpoint_pos[1]
    waypoint.position.z = (pipe_A.position.z + pipe_B.position.z) / 2.0

    normal_vec = np.array([-gate.direction[1], gate.direction[0]])
    yaw = np.arctan2(normal_vec[1], normal_vec[0])

    waypoint.orientation.x = 0.0
    waypoint.orientation.y = 0.0
    waypoint.orientation.z = np.sin(yaw / 2.0)
    waypoint.orientation.w = np.cos(yaw / 2.0)

    return [waypoint]
