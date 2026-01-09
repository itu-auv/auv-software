"""
Slalom Pipe Segmentation from depth images.

Pipeline:
1. Normalize depth to uint8
2. Suppress horizontal structures (vertical morphological opening)
3. Multi-scale black top-hat to extract narrow vertical dark structures
4. Otsu binarization
5. Geometric filtering (aspect ratio, area bounds)
6. Color classification (red vs white) using RGB image
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path


@dataclass
class PipeDetection:
    """Detected pipe with metadata."""

    label: str  # "pipe"
    color: str  # "red" or "white" (placeholder for future color classification)
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    aspect_ratio: float
    centroid: Tuple[float, float]  # normalized to [-1, 1] for visual servoing
    confidence: float
    depth: float  # mean depth in bbox region


def classify_pipe_color_lab(
    rgb: np.ndarray,
    contour: np.ndarray,
    threshold: float = 10.0,
) -> Tuple[str, float]:
    """
    Classify pipe color using LAB 'a' channel.

    Uses segmentation mask (not bbox) for accurate pixel sampling.
    LAB 'a' channel: 128=neutral, >128=red, <128=green

    Args:
        rgb: RGB image
        contour: Pipe contour from segmentation
        threshold: Redness threshold above neutral (128)

    Returns:
        (color, confidence) where color is "red" or "white"
    """
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    if cv2.countNonZero(mask) == 0:
        return "white", 0.5

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1]

    pipe_a = np.median(a_channel[mask > 0])
    redness = float(pipe_a) - 128.0

    if redness > threshold:
        confidence = min(1.0, redness / 50.0)
        return "red", confidence
    else:
        confidence = min(1.0, abs(redness) / 50.0)
        return "white", confidence


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """Normalize depth map to 0-255 range for morphological operations."""
    d_min = np.nanmin(depth)
    d_max = np.nanmax(depth)
    depth_clean = np.nan_to_num(depth, nan=d_max)

    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)

    normalized = (depth_clean - d_min) / (d_max - d_min)
    return (normalized * 255).astype(np.uint8)


def suppress_horizontal_structures(
    img: np.ndarray, height_ratio: float = 0.15
) -> np.ndarray:
    """
    Remove horizontal structures using vertical morphological opening.
    Tall vertical kernel preserves vertical pipes, destroys horizontal floor.
    """
    h = img.shape[0]
    kernel_h = max(15, int(h * height_ratio)) | 1  # Ensure odd

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def multi_scale_black_tophat(
    img: np.ndarray,
    width_ratios: List[float] = [0.015, 0.025, 0.04, 0.06],
    height_ratio: float = 0.09,
) -> np.ndarray:
    """
    Multi-scale black top-hat transform to detect dark vertical structures.
    Combines responses across multiple kernel widths to handle varying pipe widths.
    """
    h, w = img.shape
    combined = np.zeros_like(img)
    kernel_h = max(25, int(h * height_ratio)) | 1

    for w_ratio in width_ratios:
        kernel_w = max(5, int(w * w_ratio)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        combined = np.maximum(combined, tophat)

    return combined


def geometric_filtering(
    binary: np.ndarray,
    min_aspect_ratio: float = 3.2,
    min_area_ratio: float = 0.00015,
    max_area_ratio: float = 0.05,
) -> List[dict]:
    """
    Filter contours by geometric properties.

    Args:
        binary: Binary mask (uint8)
        min_aspect_ratio: Minimum height/width ratio
        min_area_ratio: Minimum contour area as fraction of image area
        max_area_ratio: Maximum contour area as fraction of image area
    """
    h, w = binary.shape
    img_area = h * w
    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = bh / (bw + 1e-8)

        if aspect > min_aspect_ratio and min_area < area < max_area:
            valid.append(
                {
                    "contour": cnt,
                    "bbox": (x, y, bw, bh),
                    "area": area,
                    "aspect_ratio": aspect,
                }
            )

    return valid


def segment_slalom_pipes(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    edge_crop_ratio: float = 0.0,
    min_aspect_ratio: float = 2.0,
    min_area_ratio: float = 0.00015,
    max_area_ratio: float = 0.05,
    return_contours: bool = False,
) -> Union[List[PipeDetection], Tuple[List[PipeDetection], List[np.ndarray]]]:
    """
    Full pipeline: Depth -> Pipe detections.

    Args:
        depth: Depth image (float32 or uint8/uint16 from PNG)
        rgb: Optional RGB image for color classification
        edge_crop_ratio: Fraction of image to crop from edges
        min_aspect_ratio: Minimum height/width ratio for pipe candidates
        min_area_ratio: Minimum area as fraction of image
        max_area_ratio: Maximum area as fraction of image
        return_contours: If True, returns (detections, contours) tuple

    Returns:
        List of PipeDetection objects, or (detections, contours) if return_contours=True
    """
    h_orig, w_orig = depth.shape[:2]

    # Edge cropping
    if edge_crop_ratio > 0:
        cx = int(w_orig * edge_crop_ratio)
        cy = int(h_orig * edge_crop_ratio)
        depth = depth[cy : h_orig - cy, cx : w_orig - cx]
        crop_offset = (cx, cy)
    else:
        crop_offset = (0, 0)

    h, w = depth.shape[:2]

    # Pipeline
    normalized = normalize_depth_to_uint8(depth)
    h_suppressed = suppress_horizontal_structures(normalized)
    tophat = multi_scale_black_tophat(h_suppressed)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = geometric_filtering(
        binary,
        min_aspect_ratio=min_aspect_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )

    # Convert to PipeDetection objects
    detections = []
    for c in contours:
        x, y, bw, bh = c["bbox"]

        # Adjust bbox for crop offset
        x_abs = x + crop_offset[0]
        y_abs = y + crop_offset[1]

        # Centroid in cropped image coords
        cx_img = x + bw / 2
        cy_img = y + bh / 2

        # Normalize centroid to [-1, 1]
        norm_cx = (cx_img / w - 0.5) * 2
        norm_cy = (cy_img / h - 0.5) * 2

        # Mean depth in bbox
        roi_depth = depth[y : y + bh, x : x + bw]
        mean_depth = float(np.nanmean(roi_depth))

        # Color classification using LAB-a with contour mask
        if rgb is not None:
            pipe_color, color_conf = classify_pipe_color_lab(rgb, c["contour"])
        else:
            pipe_color, color_conf = "white", 0.5

        det = PipeDetection(
            label="pipe",
            color=pipe_color,
            bbox=(x_abs, y_abs, bw, bh),
            area=c["area"],
            aspect_ratio=c["aspect_ratio"],
            centroid=(norm_cx, norm_cy),
            confidence=color_conf,
            depth=mean_depth,
        )
        detections.append(det)

    if return_contours:
        return detections, [c["contour"] for c in contours]
    return detections


def create_colored_visualization(
    rgb: np.ndarray,
    detections: List[PipeDetection],
    contours: List[np.ndarray],
) -> np.ndarray:
    """
    Create visualization with pipes colored by classification.
    Red pipes shown in red, white pipes shown in white.
    """
    vis = rgb.copy()

    for det, cnt in zip(detections, contours):
        color = (255, 50, 50) if det.color == "red" else (255, 255, 255)

        overlay = vis.copy()
        cv2.drawContours(overlay, [cnt], -1, color, -1)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        cv2.drawContours(vis, [cnt], -1, color, 2)

        x, y, _, _ = det.bbox
        label = f"{det.color} ({det.confidence:.0%})"
        cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis
