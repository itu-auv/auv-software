import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Any


def create_slalom_debug(
    rgb: np.ndarray,
    mask: np.ndarray,
    all_detections: List[Any],
    selected_pair: Optional[Tuple[Any, Any]],
    alpha: float = 0.7,
    depth_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    h, w = rgb.shape[:2]

    vis = rgb.copy()
    if vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    vis = (vis.astype(np.float32) * 0.3).astype(np.uint8)

    if mask is not None and mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(
            mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    if mask is not None and mask.any():
        vis[mask > 0] = [0, 255, 0]

    if depth_shape is not None:
        scale_x = w / depth_shape[1]
        scale_y = h / depth_shape[0]
    else:
        scale_x, scale_y = 1.0, 1.0

    def bbox_center(pipe) -> Tuple[int, int]:
        x, y, bw, bh = pipe.bbox
        px = int((x + bw / 2) * scale_x)
        py = int((y + bh / 2) * scale_y)
        return (px, py)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    selected_set = set()
    if selected_pair:
        selected_set = {id(selected_pair[0]), id(selected_pair[1])}

    det_coverage_mask = np.zeros((h, w), dtype=np.uint8)

    for det in all_detections:
        pt = bbox_center(det)
        is_selected = id(det) in selected_set

        x, y, bw, bh = det.bbox
        vx = int(x * scale_x)
        vy = int(y * scale_y)
        vw = int(bw * scale_x)
        vh = int(bh * scale_y)
        cv2.rectangle(det_coverage_mask, (vx, vy), (vx + vw, vy + vh), 255, -1)

        label = f"d:{det.depth:.2f}"
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = pt[0] - text_size[0] // 2
        text_y = pt[1] - 15

        bg_color = (0, 80, 0) if is_selected else (0, 0, 255)
        text_color = (0, 255, 0) if is_selected else (255, 255, 255)
        marker_color = (255, 0, 255) if is_selected else (0, 0, 255)

        cv2.rectangle(
            vis,
            (text_x - 3, text_y - text_size[1] - 3),
            (text_x + text_size[0] + 3, text_y + 3),
            bg_color,
            -1,
        )
        cv2.putText(
            vis, label, (text_x, text_y), font, font_scale, text_color, thickness
        )
        cv2.circle(vis, pt, 5, marker_color, -1)

    if mask is not None and mask.any():
        orphan_mask = cv2.bitwise_and(mask, cv2.bitwise_not(det_coverage_mask))
        contours, _ = cv2.findContours(
            orphan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2

            label = "?"
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy - 15

            cv2.rectangle(
                vis,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 2),
                (50, 50, 50),
                -1,
            )
            cv2.putText(
                vis,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
            )

    if selected_pair is None:
        return vis

    pipe_left, pipe_right = selected_pair
    pt_left = bbox_center(pipe_left)
    pt_right = bbox_center(pipe_right)

    avg_y = (pt_left[1] + pt_right[1]) // 2
    pt_left = (pt_left[0], avg_y)
    pt_right = (pt_right[0], avg_y)

    cv2.line(vis, pt_left, pt_right, (255, 255, 0), 3)

    mid_x = (pt_left[0] + pt_right[0]) // 2
    mid_y = avg_y
    cv2.circle(vis, (mid_x, mid_y), 15, (0, 0, 255), 3)
    cv2.circle(vis, (mid_x, mid_y), 5, (0, 0, 255), -1)

    return vis
