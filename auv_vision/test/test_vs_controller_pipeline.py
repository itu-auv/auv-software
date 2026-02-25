#!/usr/bin/env python3
"""
Test script for VS controller pipeline stages:
- Segmentation (SlalomSegmentor)
- Pipe selection (compute_slalom_control)
- Debug image creation (create_slalom_debug)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import argparse


def add_local_paths() -> None:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        auv_control_src = parent / "auv_control" / "auv_control" / "src"
        auv_vision_src = parent / "auv_vision" / "auv_vision" / "src"
        if auv_control_src.exists() and auv_vision_src.exists():
            sys.path.insert(0, str(auv_control_src))
            sys.path.insert(0, str(auv_vision_src))
            return
    raise RuntimeError("Failed to locate auv_control/auv_vision src paths.")


add_local_paths()

import cv2
import numpy as np

from auv_control.vs_slalom import compute_slalom_control
from auv_vision.slalom_debug import create_slalom_debug
from auv_vision.slalom_segmentation import PipeDetection, SlalomSegmentor


@dataclass
class SelectionResult:
    mode: str
    heading: Optional[float]
    lateral: Optional[float]
    pair: Optional[Tuple[PipeDetection, PipeDetection]]


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "auv_vision").exists() and (parent / "auv_control").exists():
            return parent
    raise RuntimeError("Repo root not found (expected auv_vision and auv_control).")


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar_h = 36
    result = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[bar_h:, :] = img
    cv2.putText(
        result, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return result


def hconcat_resize(images: List[np.ndarray], height: int) -> np.ndarray:
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = max(1, int(w * height / h))
        resized.append(cv2.resize(img, (new_w, height)))
    return cv2.hconcat(resized)


def draw_detections(rgb_bgr: np.ndarray, detections: List[PipeDetection]) -> np.ndarray:
    annotated = rgb_bgr.copy()
    h, w = annotated.shape[:2]
    for idx, det in enumerate(detections, start=1):
        x, y, bw, bh = det.bbox
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"{idx}: d={det.depth:.0f}"
        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        px = int((det.centroid[0] + 1) / 2 * w)
        py = int((det.centroid[1] + 1) / 2 * h)
        cv2.circle(annotated, (px, py), 4, (0, 255, 255), -1)
    return annotated


def overlay_mask(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return rgb_bgr.copy()
    vis = rgb_bgr.copy()
    overlay = vis.copy()
    overlay[mask > 0] = (0, 200, 0)
    return cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth_clean = np.nan_to_num(depth, nan=0.0)
    depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
    depth_u8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def load_depth(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Failed to load depth image: {path}")

    note = ""
    if raw.ndim == 2:
        depth = raw.astype(np.float32)
        depth_vis = depth_to_colormap(depth)
        return depth, depth_vis, note

    # If depth is already colorized (3/4 channel), keep it for visualization.
    if raw.shape[2] >= 3:
        depth_vis = raw[:, :, :3].copy()
        depth_gray = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2GRAY).astype(np.float32)
        note = "depth image appears colorized; using grayscale for segmentation"
        return depth_gray, depth_vis, note

    depth = raw.astype(np.float32)
    depth_vis = depth_to_colormap(depth)
    return depth, depth_vis, note


def compute_selection(detections: List[PipeDetection], mode: str) -> SelectionResult:
    heading, lateral, pair = compute_slalom_control(detections, mode=mode)
    return SelectionResult(mode=mode, heading=heading, lateral=lateral, pair=pair)


def create_vs_debug(
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    detections: List[PipeDetection],
    selection: SelectionResult,
) -> np.ndarray:
    debug_vis = create_slalom_debug(
        rgb=rgb_bgr,
        mask=mask,
        all_detections=detections,
        selected_pair=selection.pair,
        alpha=0.4,
    )
    if selection.heading is not None and selection.lateral is not None:
        info_text = (
            f"Mode: {selection.mode} | Heading: {selection.heading:.2f} | "
            f"LatErr: {selection.lateral:.2f}"
        )
    else:
        info_text = f"Mode: {selection.mode} | No Pair"
    cv2.putText(
        debug_vis,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return debug_vis


def summarize_pair(
    detections: List[PipeDetection],
    selection: SelectionResult,
) -> str:
    if selection.pair is None:
        return "no pair"
    det_to_idx: Dict[int, int] = {id(d): i for i, d in enumerate(detections, start=1)}
    left, right = selection.pair
    left_idx = det_to_idx.get(id(left), -1)
    right_idx = det_to_idx.get(id(right), -1)
    return (
        f"pair=({left_idx},{right_idx}) "
        f"depths=({left.depth:.0f},{right.depth:.0f}) "
        f"cx=({left.centroid[0]:.2f},{right.centroid[0]:.2f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test VS controller pipeline. Defaults to slalom_test_images unless "
            "--rgb and --depth are provided."
        )
    )
    parser.add_argument("--rgb", type=Path, help="Path to RGB image")
    parser.add_argument("--depth", type=Path, help="Path to depth image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory (default: script dir timestamp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    rgb_dir = repo_root / "auv_vision" / "slalom_test_images"
    depth_dir = repo_root / "auv_vision" / "slalom_depth_test_images" / "da3mono-large"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / f"vs_controller_test_{timestamp}"
    )
    output_dir.mkdir(exist_ok=True)

    segmentor = SlalomSegmentor()
    summary_lines = []

    print("=" * 60)
    print("VS CONTROLLER PIPELINE TEST")
    print("=" * 60)
    print(f"RGB dir: {rgb_dir}")
    print(f"Depth dir: {depth_dir}")
    print(f"Output dir: {output_dir}")

    if args.rgb and args.depth:
        image_pairs = [("custom", args.rgb, args.depth)]
    else:
        image_pairs = []
        for n in range(1, 10):
            rgb_path = rgb_dir / f"test_image_{n}.jpg"
            depth_path = depth_dir / f"test_image_{n}_depth.png"
            image_pairs.append((str(n), rgb_path, depth_path))

    for label, rgb_path, depth_path in image_pairs:
        if not (rgb_path.exists() and depth_path.exists()):
            print(f"Skipping {label}: missing files")
            continue

        print(f"Processing image {label}...")
        img_out_dir = output_dir / f"image_{label}"
        img_out_dir.mkdir(exist_ok=True)

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        try:
            depth, depth_vis, depth_note = load_depth(depth_path)
        except ValueError:
            depth = None
            depth_vis = None
            depth_note = ""

        if rgb_bgr is None or depth is None:
            print(f"Skipping {label}: failed to load images")
            continue

        h, w = depth.shape[:2]
        rgb_bgr = cv2.resize(rgb_bgr, (w, h))

        # 1) Segmentation
        result = segmentor.process(depth, return_debug=True)
        mask = result["mask"]
        detections = result["detections"]

        # 2) Pipe selection
        sel_left = compute_selection(detections, mode="left")
        sel_right = compute_selection(detections, mode="right")

        # 3) Debug visualizations (match controller text overlay)
        debug_left = create_vs_debug(rgb_bgr, mask, detections, sel_left)
        debug_right = create_vs_debug(rgb_bgr, mask, detections, sel_right)

        # Save step outputs
        cv2.imwrite(str(img_out_dir / "01_rgb.png"), rgb_bgr)
        cv2.imwrite(str(img_out_dir / "02_depth_colormap.png"), depth_vis)
        cv2.imwrite(str(img_out_dir / "03_mask.png"), mask)
        cv2.imwrite(
            str(img_out_dir / "04_mask_overlay.png"), overlay_mask(rgb_bgr, mask)
        )
        cv2.imwrite(
            str(img_out_dir / "05_detections.png"), draw_detections(rgb_bgr, detections)
        )
        cv2.imwrite(str(img_out_dir / "06_debug_left.png"), debug_left)
        cv2.imwrite(str(img_out_dir / "07_debug_right.png"), debug_right)

        # Composite summary
        row1 = hconcat_resize(
            [
                add_title(rgb_bgr, "RGB"),
                add_title(depth_vis, "Depth"),
                add_title(overlay_mask(rgb_bgr, mask), "Mask Overlay"),
            ],
            height=280,
        )
        row2 = hconcat_resize(
            [
                add_title(draw_detections(rgb_bgr, detections), "Detections"),
                add_title(debug_left, "Debug Left"),
                add_title(debug_right, "Debug Right"),
            ],
            height=280,
        )
        composite = cv2.vconcat([row1, row2])
        cv2.imwrite(str(img_out_dir / "COMPOSITE.png"), composite)

        line = (
            f"image_{label}: pipes={len(detections)} "
            f"left[{summarize_pair(detections, sel_left)}] "
            f"right[{summarize_pair(detections, sel_right)}]"
        )
        if depth_note:
            line = f"{line} | note={depth_note}"
        summary_lines.append(line)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
