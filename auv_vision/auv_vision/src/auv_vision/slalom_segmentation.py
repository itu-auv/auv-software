#!/usr/bin/env python3
"""
Slalom Segmentation Module
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter


@dataclass
class PipeDetection:
    label: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    centroid: Tuple[float, float]  # normalized [-1, 1]
    depth: float


class SlalomSegmentor:
    def __init__(self):
        # Configuration
        self.grad_threshold = 8
        self.boost_factor = 5.0

        # Filtering constraints
        self.min_aspect_ratio = 10.0
        self.min_length_ratio = 0.05  # min 5% of image height

        # Pairing constraints
        self.max_pipe_width = 80  # pixels
        self.min_vertical_overlap = 0.5  # overlap ratio for pairing

    def process(self, depth: np.ndarray, return_debug: bool = False) -> Dict[str, Any]:
        """
        Process depth image to detect pipes.

        Args:
            depth: Input depth image (float32 or uint8)
            return_debug: If True, returns intermediate images

        Returns:
            Dict containing 'mask', 'detections', and optionally debug images.
        """
        h, w = depth.shape[:2]

        # 1. Gradient & Enhancement
        grad_signed, enhanced = self._compute_enhanced_gradient(depth)

        # 2. Extract & Filter Edge Components
        blue_binary = enhanced[:, :, 0]
        red_binary = enhanced[:, :, 2]

        blue_comps = self._filter_edge_components(blue_binary, h)
        red_comps = self._filter_edge_components(red_binary, h)

        # 3. Pair Edges & Generate Mask
        mask = self._pair_components(blue_comps, red_comps, (h, w))

        # 4. Generate Detections
        detections = self._extract_detections(mask, depth)

        result = {"mask": mask, "detections": detections, "pipe_count": len(detections)}

        if return_debug:
            # Create debug visualization
            debug_vis = self._create_debug_vis(enhanced, blue_comps, red_comps, mask)
            result.update(
                {
                    "grad_signed": grad_signed,
                    "enhanced": enhanced,
                    "debug_vis": debug_vis,
                }
            )

        return result

    def _compute_enhanced_gradient(
        self, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Sobel X and enhance contrast."""
        # Normalize
        d_min, d_max = np.nanmin(depth), np.nanmax(depth)
        if d_max - d_min < 1e-6:
            normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Sobel X
        grad_x = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3)

        # Create signed visualization (Blue=Left, Red=Right)
        grad_norm = grad_x / (np.abs(grad_x).max() + 1e-8)
        blue_ch = (np.clip(-grad_norm, 0, 1) * 255).astype(np.uint8)
        red_ch = (np.clip(grad_norm, 0, 1) * 255).astype(np.uint8)

        grad_signed = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        grad_signed[:, :, 0] = blue_ch
        grad_signed[:, :, 2] = red_ch

        # Enhance contrast
        blue_enh = np.clip(
            blue_ch.astype(np.float32) * self.boost_factor, 0, 255
        ).astype(np.uint8)
        red_enh = np.clip(red_ch.astype(np.float32) * self.boost_factor, 0, 255).astype(
            np.uint8
        )

        # Threshold
        blue_enh[blue_ch < self.grad_threshold] = 0
        red_enh[red_ch < self.grad_threshold] = 0

        enhanced = np.zeros_like(grad_signed)
        enhanced[:, :, 0] = blue_enh
        enhanced[:, :, 2] = red_enh

        return grad_signed, enhanced

    def _filter_edge_components(
        self, binary_img: np.ndarray, img_height: int
    ) -> List[Dict]:
        """
        Find connected components and filter by rotated bbox aspect ratio.
        Returns list of valid component dicts.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_binary := (binary_img > 0).astype(np.uint8) * 255
        )

        valid_comps = []
        min_len_px = img_height * self.min_length_ratio

        for i in range(1, num_labels):
            # Get component mask
            comp_mask = (labels == i).astype(np.uint8) * 255

            # Find contours to get rotated bbox
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # MinAreaRect for robust AR calculation
            rect = cv2.minAreaRect(contours[0])
            (cx, cy), (w, h), angle = rect

            length = max(w, h)
            thickness = min(w, h) + 1e-8
            ar = length / thickness

            if ar >= self.min_aspect_ratio and length >= min_len_px:
                # Also get axis-aligned bounds for faster layout logic later
                x, y, bw, bh, area = stats[i]
                valid_comps.append(
                    {
                        "id": i,
                        "mask": comp_mask,
                        "x": x,
                        "y": y,
                        "w": bw,
                        "h": bh,
                        "cx": cx,
                        "cy": cy,
                        "rect": rect,
                    }
                )

        # Sort by x coordinate
        valid_comps.sort(key=lambda c: c["x"])
        return valid_comps

    def _pair_components(
        self, blues: List[Dict], reds: List[Dict], shape: Tuple[int, int]
    ) -> np.ndarray:
        """Pair blue (left) and red (right) components to create pipe mask."""
        mask = np.zeros(shape, dtype=np.uint8)

        used_reds = set()

        for b in blues:
            best_r = None
            min_dist = self.max_pipe_width + 1

            # Find closest red component to the right
            for r in reds:
                if r["id"] in used_reds:
                    continue

                # Basic position check: Red must be to the right of Blue
                # Using centroids for robustness
                if r["cx"] <= b["cx"]:
                    continue

                dist = r["x"] - (b["x"] + b["w"])  # gap distance

                # Check constraints
                if dist < 0:
                    dist = 0  # overlap

                if dist < min_dist and dist < self.max_pipe_width:
                    # Check vertical alignment (y-overlap)
                    y_start = max(b["y"], r["y"])
                    y_end = min(b["y"] + b["h"], r["y"] + r["h"])
                    overlap_h = y_end - y_start

                    min_h = min(b["h"], r["h"])
                    if overlap_h > min_h * self.min_vertical_overlap:
                        min_dist = dist
                        best_r = r

            if best_r:
                used_reds.add(best_r["id"])
                # Fill between components
                self._fill_between(mask, b["mask"], best_r["mask"])

        return mask

    def _fill_between(
        self, canvas: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray
    ):
        """Fill space between two component masks row by row."""
        rows, cols = canvas.shape

        # Combine to find ROI to speed up
        combined = left_mask | right_mask
        y_indices, x_indices = np.where(combined > 0)

        if y_indices.size == 0:
            return

        y_min, y_max = y_indices.min(), y_indices.max()

        for y in range(y_min, y_max + 1):
            l_cols = np.where(left_mask[y] > 0)[0]
            r_cols = np.where(right_mask[y] > 0)[0]

            if l_cols.size > 0 and r_cols.size > 0:
                x1 = l_cols.max()  # Rightmost pixel of left edge
                x2 = r_cols.min()  # Leftmost pixel of right edge

                if x2 > x1:
                    canvas[y, x1 : x2 + 1] = 255

    def _extract_detections(
        self, mask: np.ndarray, depth: np.ndarray
    ) -> List[PipeDetection]:
        """Extract bbox and metadata from final mask using MODE depth."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h, w = mask.shape

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Min area filter for final blobs
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Normalized centroid
            cx = x + bw / 2
            cy = y + bh / 2
            norm_cx = (cx / w - 0.5) * 2
            norm_cy = (cy / h - 0.5) * 2

            # Create a mask for just this contour
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

            # Extract depth values within this pipe's mask
            pipe_depths = depth[contour_mask > 0]

            # Calculate MODE depth (most frequent value)
            if pipe_depths.size > 0:
                depth_ints = pipe_depths.astype(int)
                counter = Counter(depth_ints)
                mode_depth = float(counter.most_common(1)[0][0])
            else:
                mode_depth = 0.0

            detections.append(
                PipeDetection(
                    label="pipe",
                    bbox=(x, y, bw, bh),
                    area=area,
                    centroid=(norm_cx, norm_cy),
                    depth=mode_depth,
                )
            )

        return detections

    def _create_debug_vis(self, enhanced, blues, reds, mask):
        """Create visualization of intermediate steps."""
        vis = enhanced.copy()

        # Draw all valid components
        for b in blues:
            box = cv2.boxPoints(b["rect"]).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (255, 150, 0), 1)  # Cyan-ish for blue

        for r in reds:
            box = cv2.boxPoints(r["rect"]).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 150, 255), 1)  # Orange-ish for red

        # Overlay mask
        vis[:, :, 1] = np.maximum(vis[:, :, 1], mask // 2)

        return vis
