"""
Slalom heading and lateral control computation.

Takes 3 closest pipes, sorts by x, selects pair based on mode (left/right).
"""

from typing import List, Optional, Tuple, Any


def compute_slalom_control(
    detections: List[Any],
    mode: str = "left",
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[Any, Any]]]:
    """
    Compute heading and lateral errors for slalom navigation.

    Returns:
        (heading_error, lateral_error, selected_pair) or (None, None, None) if < 2 pipes
    """
    if len(detections) < 2:
        return None, None, None

    # 3 closest by depth
    closest = sorted(detections, key=lambda d: d.depth)[:3]

    # sort by x (left to right)
    by_x = sorted(closest, key=lambda d: d.centroid[0])

    # select pair
    if len(by_x) == 2:
        left, right = by_x[0], by_x[1]
    elif mode == "left":
        left, right = by_x[0], by_x[1]
    else:
        left, right = by_x[-2], by_x[-1]

    heading_error = 0.5 * (left.centroid[0] + right.centroid[0])
    lateral_error = left.depth - right.depth

    return heading_error, lateral_error, (left, right)
