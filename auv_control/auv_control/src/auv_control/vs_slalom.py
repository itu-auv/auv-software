"""
Slalom heading and lateral control computation.

Selection logic (v2 - Row-Based):
1. Group detections into rows using Adaptive K-Means clustering
2. Select closest row (Row 0)
3. Sort Row 0 pipes by x-centroid (left to right)
4. Select pair based on mode:
   - left:  pipes[0] + pipes[1] (left + center)
   - right: pipes[-2] + pipes[-1] (center + right)
"""

from typing import List, Optional, Tuple, Any
from auv_control.slalom_row_detector import group_pipes_by_row


def select_slalom_pair(
    detections: List[Any], mode: str = "left"
) -> Optional[Tuple[Any, Any]]:
    """
    Select 2 pipes from closest row based on mode.

    Args:
        detections: List of PipeDetection with .depth and .centroid attributes
        mode: "left" or "right"

    Returns:
        Tuple of (pipe_left, pipe_right) sorted by x-centroid, or None if < 2 pipes
    """
    if len(detections) < 2:
        return None

    # group into rows
    rows = group_pipes_by_row(detections)

    if not rows:
        return None

    # get closest row
    active_row = rows[0]

    if len(active_row) < 2:
        # not enough pipes in closest row, try merging with next row
        if len(rows) > 1:
            active_row = active_row + rows[1]
        if len(active_row) < 2:
            return None

    # sort by x-centroid (left to right)
    by_x = sorted(active_row, key=lambda d: d.centroid[0])

    if len(by_x) == 2:
        return (by_x[0], by_x[1])

    # 3+ pipes available
    if mode == "left":
        return (by_x[0], by_x[1])  # left + center
    else:  # right
        return (by_x[-2], by_x[-1])  # center + right


def compute_slalom_control(
    detections: List[Any],
    mode: str = "left",
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[Any, Any]]]:
    """
    Compute heading and lateral errors for slalom navigation.

    Args:
        detections: List of PipeDetection objects
        mode: "left" or "right" - which side we're passing

    Returns:
        Tuple of:
        - heading_error: float in [-1, 1], average of x-centroids (None if < 2 pipes)
        - lateral_error: float, raw depth difference (None if < 2 pipes)
        - selected_pair: Tuple of the 2 selected pipes for debug (None if < 2 pipes)
    """
    pair = select_slalom_pair(detections, mode)

    if pair is None:
        return None, None, None

    pipe_left, pipe_right = pair

    # Heading: average x-centroid (already normalized to [-1, 1])
    heading_error = 0.5 * (pipe_left.centroid[0] + pipe_right.centroid[0])

    # Lateral: raw depth difference, PID handles scaling
    # Positive = left pipe farther -> strafe right
    # Negative = right pipe farther -> strafe left
    lateral_error = pipe_left.depth - pipe_right.depth

    return heading_error, lateral_error, pair


# Legacy function for backward compatibility
def compute_slalom_heading(detections: List[Any]) -> Optional[float]:
    """
    Legacy function - compute heading only using 2 closest pipes.
    """
    if len(detections) < 2:
        return None

    sorted_pipes = sorted(detections, key=lambda d: d.depth)
    pipe_a, pipe_b = sorted_pipes[0], sorted_pipes[1]

    heading = 0.5 * (pipe_a.centroid[0] + pipe_b.centroid[0])
    return heading
