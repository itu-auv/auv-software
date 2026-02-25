#!/usr/bin/env python3
"""
Slalom Segmentation Module
"""

import math
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
    length: float


class SlalomSegmentor:
    def __init__(self, fx: Optional[float] = None, cx: Optional[float] = None):
        self.fx = fx
        self.cx = cx

        # Configuration
        self.grad_threshold = 8
        self.boost_factor = 5.0

        # Filtering constraints
        self.min_aspect_ratio = 10.0
        self.min_length_ratio = 0.05
        self.min_area_ratio = 0.00013

        # Pairing constraints
        self.max_pipe_width_ratio = 0.167  # ~80px for 480px width
        self.min_vertical_overlap = 0.5

        # Rope cleanup & Length calculation parameters
        self.cleanup_min_area = 30
        self.safe_zone_start_ratio = 0.2
        self.safe_zone_end_ratio = 0.6
        self.width_threshold_ratio = 0.8
        self.fitline_dist_type = cv2.DIST_L2
        self.fitline_param = 0
        self.fitline_reps = 0.01
        self.fitline_aeps = 0.01

    def pixel_to_yaw(self, pixel_x: float, image_width: int) -> float:
        if self.fx is not None and self.cx is not None:
            return math.atan((pixel_x - self.cx) / self.fx)
        return (pixel_x / image_width - 0.5) * 2

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

        # TODO remove - debug depth range
        import rospy

        rospy.loginfo_throttle(
            5.0,
            f"[SEG DEBUG] depth min={depth.min():.4f}, max={depth.max():.4f}, dtype={depth.dtype}",
        )

        # 1. Gradient & Enhancement
        grad_signed, enhanced = self._compute_enhanced_gradient(depth)

        # 2. Extract & Filter Edge Components
        blue_binary = enhanced[:, :, 0]
        red_binary = enhanced[:, :, 2]

        blue_valid, blue_rejected = self._filter_edge_components(blue_binary, h)
        red_valid, red_rejected = self._filter_edge_components(red_binary, h)

        # 3. Pair Edges & Generate Mask
        mask, paired_blues, paired_reds, unpaired_blues, unpaired_reds = (
            self._pair_components(blue_valid, red_valid, (h, w))
        )

        # 4. Refine Mask (Rope Cleanup)
        refined_mask = self._refine_mask(mask)

        # 5. Generate Detections
        detections, rejected_contours = self._extract_detections(refined_mask, depth)

        result = {
            "mask": refined_mask,
            "detections": detections,
            "pipe_count": len(detections),
        }

        if return_debug:
            # Create debug visualization
            debug_vis = self._create_debug_vis(
                enhanced, blue_valid, red_valid, refined_mask
            )

            debug_components = self._draw_component_debug(
                enhanced.copy(), blue_valid, blue_rejected, red_valid, red_rejected, h
            )
            debug_pairing = self._draw_pairing_debug(
                enhanced.copy(),
                paired_blues,
                paired_reds,
                unpaired_blues,
                unpaired_reds,
            )
            debug_detections = self._draw_detection_debug(
                enhanced.copy(), refined_mask, detections, rejected_contours
            )

            result.update(
                {
                    "grad_signed": grad_signed,
                    "enhanced": enhanced,
                    "debug_vis": debug_vis,
                    "debug_components": debug_components,
                    "debug_pairing": debug_pairing,
                    "debug_detections": debug_detections,
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

        grad_x = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3)

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
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Find connected components and filter by rotated bbox aspect ratio.
        Returns (valid_comps, rejected_comps) with rejection reasons.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_binary := (binary_img > 0).astype(np.uint8) * 255
        )

        valid_comps = []
        rejected_comps = []
        min_len_px = img_height * self.min_length_ratio

        for i in range(1, num_labels):
            comp_mask = (labels == i).astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            rect = cv2.minAreaRect(contours[0])
            (cx, cy), (w, h), angle = rect

            length = max(w, h)
            thickness = min(w, h) + 1e-8
            ar = length / thickness

            x, y, bw, bh, area = stats[i]
            comp_info = {
                "id": i,
                "mask": comp_mask,
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "cx": cx,
                "cy": cy,
                "rect": rect,
                "ar": ar,
                "length": length,
            }

            if ar < self.min_aspect_ratio:
                comp_info["reject_reason"] = "AR_LOW"
                rejected_comps.append(comp_info)
            elif length < min_len_px:
                comp_info["reject_reason"] = "LEN_SHORT"
                rejected_comps.append(comp_info)
            else:
                valid_comps.append(comp_info)

        valid_comps.sort(key=lambda c: c["x"])
        return valid_comps, rejected_comps

    def _pair_components(
        self, blues: List[Dict], reds: List[Dict], shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        Pair blue (left) and red (right) components to create pipe mask.
        Returns (mask, paired_blues, paired_reds, unpaired_blues, unpaired_reds)
        """
        mask = np.zeros(shape, dtype=np.uint8)
        max_pipe_width = int(shape[1] * self.max_pipe_width_ratio)

        used_blues = set()
        used_reds = set()
        paired_blues = []
        paired_reds = []

        for b in blues:
            best_r = None
            min_dist = max_pipe_width + 1

            for r in reds:
                if r["id"] in used_reds:
                    continue

                if r["cx"] <= b["cx"]:
                    continue

                dist = r["x"] - (b["x"] + b["w"])

                if dist < 0:
                    dist = 0

                if dist < min_dist and dist < max_pipe_width:
                    y_start = max(b["y"], r["y"])
                    y_end = min(b["y"] + b["h"], r["y"] + r["h"])
                    overlap_h = y_end - y_start

                    min_h = min(b["h"], r["h"])
                    if overlap_h > min_h * self.min_vertical_overlap:
                        min_dist = dist
                        best_r = r

            if best_r:
                used_blues.add(b["id"])
                used_reds.add(best_r["id"])
                paired_blues.append(b)
                paired_reds.append(best_r)
                self._fill_between(mask, b["mask"], best_r["mask"])

        unpaired_blues = [b for b in blues if b["id"] not in used_blues]
        unpaired_reds = [r for r in reds if r["id"] not in used_reds]

        return mask, paired_blues, paired_reds, unpaired_blues, unpaired_reds

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

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        output_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cv2.contourArea(cnt) < self.cleanup_min_area:
                continue

            single_cnt_mask = np.zeros_like(mask)
            cv2.drawContours(single_cnt_mask, [cnt], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(cnt)

            widths = []
            for row in range(y, y + h):
                row_pixels = np.count_nonzero(single_cnt_mask[row, x : x + w])
                widths.append(row_pixels)

            safe_zone = widths[
                int(h * self.safe_zone_start_ratio) : int(h * self.safe_zone_end_ratio)
            ]
            ref_width = np.median(safe_zone) if safe_zone else widths[0]

            cut_y_relative = h

            for i in range(h - 1, int(h * self.safe_zone_start_ratio), -1):
                if widths[i] >= ref_width * self.width_threshold_ratio:
                    cut_y_relative = i
                    break

            cut_y_absolute = y + cut_y_relative
            output_mask[y:cut_y_absolute, x : x + w] = single_cnt_mask[
                y:cut_y_absolute, x : x + w
            ]

        return output_mask

    def _extract_detections(
        self, mask: np.ndarray, depth: np.ndarray
    ) -> Tuple[List[PipeDetection], List[Dict]]:
        """
        Extract bbox and metadata from final mask.
        Returns (detections, rejected_contours)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        rejected_contours = []
        h, w = mask.shape
        min_area = self.min_area_ratio * h * w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)

            if area < min_area:
                rejected_contours.append(
                    {
                        "contour": cnt,
                        "bbox": (x, y, bw, bh),
                        "area": area,
                        "reject_reason": "AREA_TOO_SMALL",
                    }
                )
                continue

            cx = x + bw / 2
            cy = y + bh / 2
            yaw_x = self.pixel_to_yaw(cx, w)
            norm_cy = (cy / h - 0.5) * 2

            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

            pipe_depths = depth[contour_mask > 0]

            if pipe_depths.size > 0:
                mode_depth = float(np.median(pipe_depths))
            else:
                mode_depth = 0.0

            [vx, vy, x0, y0] = cv2.fitLine(
                cnt,
                self.fitline_dist_type,
                self.fitline_param,
                self.fitline_reps,
                self.fitline_aeps,
            )
            pts = cnt.reshape(-1, 2)
            projections = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy

            min_proj = np.min(projections)
            max_proj = np.max(projections)

            p1 = (int(x0 + min_proj * vx), int(y0 + min_proj * vy))
            p2 = (int(x0 + max_proj * vx), int(y0 + max_proj * vy))

            length = float(np.linalg.norm(np.array(p1) - np.array(p2)))

            detections.append(
                PipeDetection(
                    label="pipe",
                    bbox=(x, y, bw, bh),
                    area=area,
                    centroid=(yaw_x, norm_cy),
                    depth=mode_depth,
                    length=length,
                )
            )

            setattr(detections[-1], "_line_pts", (p1, p2))

        return detections, rejected_contours

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

    def _draw_component_debug(
        self,
        base_img: np.ndarray,
        blue_valid: List[Dict],
        blue_rejected: List[Dict],
        red_valid: List[Dict],
        red_rejected: List[Dict],
        img_height: int,
    ) -> np.ndarray:
        """
        Visualize component filtering results.
        Green = valid, Red = rejected with reason label.
        """
        vis = base_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1

        min_len_px = img_height * self.min_length_ratio

        def draw_comp(comp, color, show_reason=False):
            box = cv2.boxPoints(comp["rect"]).astype(np.int32)
            cv2.drawContours(vis, [box], 0, color, 1)

            cx, cy = int(comp["cx"]), int(comp["cy"])
            ar = comp.get("ar", 0)
            length = comp.get("length", 0)

            label = f"AR:{ar:.1f} L:{int(length)}"
            if show_reason and "reject_reason" in comp:
                label += f" ({comp['reject_reason']})"

            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy - 5

            cv2.rectangle(
                vis,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis, label, (text_x, text_y), font, font_scale, color, thickness
            )

        for comp in blue_valid:
            draw_comp(comp, (0, 255, 0))
        for comp in red_valid:
            draw_comp(comp, (0, 255, 0))
        for comp in blue_rejected:
            draw_comp(comp, (0, 0, 255), show_reason=True)
        for comp in red_rejected:
            draw_comp(comp, (0, 0, 255), show_reason=True)

        # legend
        cv2.putText(
            vis,
            f"min_AR={self.min_aspect_ratio:.1f} min_L={int(min_len_px)}px",
            (5, 15),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

        return vis

    def _draw_pairing_debug(
        self,
        base_img: np.ndarray,
        paired_blues: List[Dict],
        paired_reds: List[Dict],
        unpaired_blues: List[Dict],
        unpaired_reds: List[Dict],
    ) -> np.ndarray:
        """
        Visualize pairing results.
        Green line = paired, Yellow = unpaired.
        """
        vis = base_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # draw paired connections
        for b, r in zip(paired_blues, paired_reds):
            b_center = (int(b["cx"]), int(b["cy"]))
            r_center = (int(r["cx"]), int(r["cy"]))
            cv2.line(vis, b_center, r_center, (0, 255, 0), 2)
            cv2.circle(vis, b_center, 4, (255, 150, 0), -1)
            cv2.circle(vis, r_center, 4, (0, 150, 255), -1)

        # draw unpaired
        for comp in unpaired_blues:
            box = cv2.boxPoints(comp["rect"]).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 255, 255), 2)  # yellow
            cv2.putText(
                vis,
                "NO_PAIR",
                (int(comp["cx"]) - 20, int(comp["cy"])),
                font,
                0.35,
                (0, 255, 255),
                1,
            )

        for comp in unpaired_reds:
            box = cv2.boxPoints(comp["rect"]).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 255, 255), 2)  # yellow
            cv2.putText(
                vis,
                "NO_PAIR",
                (int(comp["cx"]) - 20, int(comp["cy"])),
                font,
                0.35,
                (0, 255, 255),
                1,
            )

        # legend
        cv2.putText(
            vis,
            f"Paired:{len(paired_blues)} Unpaired:{len(unpaired_blues)+len(unpaired_reds)}",
            (5, 15),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

        return vis

    def _draw_detection_debug(
        self,
        base_img: np.ndarray,
        mask: np.ndarray,
        detections: List[PipeDetection],
        rejected_contours: List[Dict],
    ) -> np.ndarray:
        """
        Visualize detection extraction results.
        Green = accepted, Red = rejected (area too small).
        """
        vis = base_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1

        # overlay mask faintly
        vis[:, :, 1] = np.maximum(vis[:, :, 1], mask // 3)

        # draw accepted detections
        for det in detections:
            x, y, bw, bh = det.bbox
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            label = f"A:{int(det.area)} d:{det.depth:.2f}"
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + bw // 2 - text_size[0] // 2
            text_y = y + bh // 2

            cv2.rectangle(
                vis,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 2),
                (0, 80, 0),
                -1,
            )
            cv2.putText(
                vis, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness
            )

            if hasattr(det, "_line_pts"):
                p1, p2 = getattr(det, "_line_pts")
                cv2.line(vis, p1, p2, (0, 0, 255), 2)
                cv2.putText(
                    vis,
                    f"{int(det.length)}px",
                    (x + bw + 5, y + bh // 2),
                    font,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # draw rejected contours
        for rej in rejected_contours:
            x, y, bw, bh = rej["bbox"]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 1)

            label = f"A:{int(rej['area'])} (SMALL)"
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + bw // 2 - text_size[0] // 2
            text_y = y + bh // 2

            cv2.rectangle(
                vis,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 2),
                (0, 0, 80),
                -1,
            )
            cv2.putText(
                vis, label, (text_x, text_y), font, font_scale, (0, 0, 255), thickness
            )

        # legend
        h, w = mask.shape[:2]
        min_area = self.min_area_ratio * h * w
        cv2.putText(
            vis,
            f"Detected:{len(detections)} Rejected:{len(rejected_contours)} (min_area={int(min_area)})",
            (5, 15),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

        return vis
