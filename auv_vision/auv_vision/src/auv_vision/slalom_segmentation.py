"""
Slalom pipe segmentation using Depth + RGB sensor fusion.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class PipeDetection:
    """Single detected pipe."""
    label: str = "pipe"
    color: str = "unknown"
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    depth: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)  # normalized [-1, 1]
    mask: np.ndarray = field(default_factory=lambda: np.array([]))


# Tunable thresholds
MIN_CONTOUR_AREA = 500
MIN_ASPECT_RATIO = 2.0
MIN_SOLIDITY = 0.5
MORPH_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MORPH_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Float32 depth -> uint8 for thresholding."""
    return cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _threshold_depth(norm_depth: np.ndarray) -> np.ndarray:
    """Otsu binarization to separate foreground from background."""
    _, mask = cv2.threshold(norm_depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _apply_morphology(mask: np.ndarray) -> np.ndarray:
    """Open to remove noise, close with vertical kernel to solidify pipes."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_CLOSE_KERNEL)
    return mask


def _contour_touches_border(cnt: np.ndarray, img_shape: Tuple[int, int]) -> bool:
    """Check if contour bbox touches image border."""
    h, w = img_shape[:2]
    x, y, bw, bh = cv2.boundingRect(cnt)
    return x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h


def _classify_color(rgb: np.ndarray, mask: np.ndarray) -> str:
    """
    Classify pipe color using masked mean in HSV.
    Red: H near 0 or 170-180, high S.
    White: low S, high V.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    h, s, v = mean_hsv

    if s > 80:
        if h < 15 or h > 165:
            return "red"
        if 35 < h < 85:
            return "green"
        if 85 < h < 130:
            return "blue"
    elif s < 60 and v > 180:
        return "white"
    return "unknown"


def segment_slalom_pipes(
    rgb: np.ndarray,
    depth: np.ndarray,
    min_area: int = MIN_CONTOUR_AREA,
    min_aspect: float = MIN_ASPECT_RATIO,
    min_solidity: float = MIN_SOLIDITY,
    relax_aspect_on_border: bool = True,
) -> List[PipeDetection]:
    """
    Detect slalom pipes using depth-based segmentation + color classification.

    Args:
        rgb: RGB image (H, W, 3), uint8.
        depth: Float32 depth map from DA3 (higher = closer).
        min_area: Minimum contour area in pixels.
        min_aspect: Minimum height/width ratio (pipes are tall).
        min_solidity: Minimum area/hull_area ratio.
        relax_aspect_on_border: Relax aspect ratio for border-touching contours.

    Returns:
        List of PipeDetection objects.
    """
    h_img, w_img = depth.shape[:2]
    norm_depth = _normalize_depth(depth)
    mask = _threshold_depth(norm_depth)
    mask = _apply_morphology(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[PipeDetection] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / float(w) if w > 0 else 0

        touches_border = _contour_touches_border(cnt, (h_img, w_img))
        required_aspect = 1.2 if (relax_aspect_on_border and touches_border) else min_aspect
        if aspect < required_aspect:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        # Create per-contour mask for color extraction
        cnt_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        color = _classify_color(rgb, cnt_mask)

        # Median depth value within contour
        masked_depth = depth[cnt_mask > 0]
        median_depth = float(np.median(masked_depth)) if masked_depth.size > 0 else 0.0

        # Normalized centroid (center of image = 0, edges = +/- 1)
        cx = x + w / 2
        cy = y + h / 2
        norm_cx = (cx / w_img - 0.5) * 2
        norm_cy = (cy / h_img - 0.5) * 2

        # Confidence heuristic: larger, closer, and more solid = higher confidence
        conf = min(1.0, (area / 5000) * solidity * (median_depth / 255))

        det = PipeDetection(
            label="pipe",
            color=color,
            confidence=round(conf, 3),
            bbox=(x, y, w, h),
            depth=round(median_depth, 2),
            centroid=(round(norm_cx, 3), round(norm_cy, 3)),
            mask=cnt_mask,
        )
        detections.append(det)

    # Sort by depth descending (closest first)
    detections.sort(key=lambda d: d.depth, reverse=True)
    return detections
