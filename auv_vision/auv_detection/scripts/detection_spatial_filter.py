import numpy as np
from geometry_msgs.msg import Pose
from typing import Tuple

def _transform_pose_to_gate_frame(pose: Pose, gate_angle: float, gate_center: Tuple[float, float]) -> Tuple[float, float]:
    """
    Transforms the pose from the odom frame to the gate's reference frame.
    The gate frame's origin is at the gate_center, and its x-axis points
    perpendicular to the gate, away from it. The y-axis is parallel to the gate.

    Args:
        pose: The pose to transform.
        gate_angle: The angle of the gate in the odom frame (in radians).
        gate_center: The center of the gate in the odom frame.

    Returns:
        A tuple (x, y) representing the pose's position in the gate frame.
    """
    # Implementation to be added.
    return (0.0, 0.0)

def _is_in_trapezoid(
    point: Tuple[float, float],
    gate_width: float,
    min_dist: float,
    max_dist: float,
) -> bool:
    """
    Checks if a point is inside the valid trapezoidal area in the gate frame.

    Args:
        point: The (x, y) coordinates of the point in the gate frame.
        gate_width: The width of the gate.
        min_dist: The minimum distance from the gate (defines the near side of the trapezoid).
        max_dist: The maximum distance from the gate (defines the far side of the trapezoid).

    Returns:
        True if the point is inside the trapezoid, False otherwise.
    """
    # Implementation to be added.
    return True

def is_pose_valid(
    pose: Pose,
    gate_angle: float,
    gate_width: float,
    gate_center: Tuple[float, float],
    min_dist: float,
    max_dist: float,
) -> bool:
    """
    Filters a pose based on its spatial relationship to a gate.

    The valid area is a trapezoid in front of the gate, defined by the gate's
    position and orientation.

    Args:
        pose: The pose to check, in the odom frame.
        gate_angle: The angle of the gate in the odom frame (in radians).
        gate_width: The width of the gate.
        gate_center: The center of the gate in the odom frame.
        min_dist: The minimum distance from the gate.
        max_dist: The maximum distance from the gate.

    Returns:
        True if the pose is within the valid area, False otherwise.
    """
    transformed_point = _transform_pose_to_gate_frame(pose, gate_angle, gate_center)
    return _is_in_trapezoid(transformed_point, gate_width, min_dist, max_dist)
