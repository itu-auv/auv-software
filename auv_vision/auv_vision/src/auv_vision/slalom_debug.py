"""
Enhanced debug visualization for slalom navigation.

Draws on RGB image:
- Segmentation masks (semi-transparent green overlay)
- Line connecting selected pipe pair
- Target circle at midpoint
- Depth labels above selected pipes
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any


def create_slalom_debug(
    rgb: np.ndarray,
    mask: np.ndarray,
    all_detections: List[Any],
    selected_pair: Optional[Tuple[Any, Any]],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create enhanced debug visualization.

    Args:
        rgb: Original RGB image (H, W, 3)
        mask: Segmentation mask from SlalomSegmentor (H, W)
        all_detections: List of all PipeDetection objects
        selected_pair: Tuple of (pipe_left, pipe_right) or None
        alpha: Mask overlay transparency

    Returns:
        Debug visualization image (BGR format for ROS/OpenCV)
    """
    h, w = rgb.shape[:2]

    # Convert RGB to BGR if needed (ROS uses RGB, OpenCV uses BGR)
    vis = rgb.copy()
    if vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # Resize mask to match RGB image if needed
    if mask is not None and mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(
            mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # 1. Overlay segmentation mask (green, semi-transparent)
    if mask is not None and mask.any():
        overlay = vis.copy()
        overlay[mask > 0] = [0, 200, 0]  # Green
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    if selected_pair is None:
        return vis

    pipe_left, pipe_right = selected_pair

    # Convert normalized centroids to pixel coords
    def to_pixel(centroid: Tuple[float, float]) -> Tuple[int, int]:
        nx, ny = centroid
        px = int((nx + 1) / 2 * w)
        py = int((ny + 1) / 2 * h)
        return (px, py)

    pt_left = to_pixel(pipe_left.centroid)
    pt_right = to_pixel(pipe_right.centroid)

    # FORCE HORIZONTAL ALIGNMENT
    # Average the Y-coordinates to make points horizontally aligned
    avg_y = (pt_left[1] + pt_right[1]) // 2
    pt_left = (pt_left[0], avg_y)
    pt_right = (pt_right[0], avg_y)

    # 2. Draw simple horizontal line between aligned points
    cv2.line(vis, pt_left, pt_right, (255, 255, 0), 3)  # Cyan horizontal

    # 3. Draw target circle at exact midpoint
    mid_x = (pt_left[0] + pt_right[0]) // 2
    mid_y = avg_y
    cv2.circle(vis, (mid_x, mid_y), 15, (0, 0, 255), 3)  # Red circle
    cv2.circle(vis, (mid_x, mid_y), 5, (0, 0, 255), -1)  # Red dot

    # 4. Draw depth labels above selected pipes
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    for pipe, pt in [(pipe_left, pt_left), (pipe_right, pt_right)]:
        label = f"d:{pipe.depth:.0f}"
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = pt[0] - text_size[0] // 2
        text_y = pt[1] - 30  # Above the centroid

        # Background rectangle for readability
        cv2.rectangle(
            vis,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            vis, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
        )

        # Draw centroid marker
        cv2.circle(vis, pt, 8, (255, 0, 255), -1)  # Magenta dot

    return vis
