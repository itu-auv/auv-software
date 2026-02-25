#!/usr/bin/env python3
"""
Slalom Segmentation Pipeline Test - visualizes each stage and saves debug output.

Outputs:
- Pipeline stages (depth, h-suppressed, tophat, binary)
- Mask overlay on RGB (transparent mask for coverage inspection)
- Masked RGB with black background (pipe regions only)
- High-resolution outputs for zoom inspection
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

import sys

sys.path.insert(0, str(Path(__file__).parent / "auv_vision" / "src"))

from auv_vision.slalom_segmentation import (
    normalize_depth_to_uint8,
    suppress_horizontal_structures,
    multi_scale_black_tophat,
    geometric_filtering,
    segment_slalom_pipes,
)

# High resolution scale factor for detailed inspection
HIRES_SCALE = 3


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    """Add title bar to image."""
    h, w = img.shape[:2]
    bar_h = 35
    result = np.zeros((h + bar_h, w, 3), dtype=np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result[bar_h:, :] = img
    cv2.putText(
        result, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return result


def hconcat_resize(images: list, height: int = 400) -> np.ndarray:
    """Horizontally concatenate images at uniform height."""
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * height / h)
        resized.append(cv2.resize(img, (new_w, height)))
    return cv2.hconcat(resized)


def upscale(img: np.ndarray, scale: int = HIRES_SCALE) -> np.ndarray:
    """Upscale image for high-res inspection."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def create_mask_overlay(
    rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Create RGB with semi-transparent mask overlay.
    Shows segmentation coverage on original image.
    """
    overlay = rgb.copy()
    mask_colored = np.zeros_like(rgb)
    mask_colored[:, :, 1] = mask  # Green channel for mask
    mask_colored[:, :, 2] = mask  # Also yellow tint

    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        rgb[mask_bool], 1 - alpha, mask_colored[mask_bool], alpha, 0
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    return overlay


def create_masked_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create RGB with black background - only mask regions visible.
    Shows exactly what the segmentation captured.
    """
    result = np.zeros_like(rgb)
    mask_bool = mask > 0
    result[mask_bool] = rgb[mask_bool]
    return result


def test_pipeline(rgb_path: Path, depth_path: Path, output_dir: Path) -> int:
    """Run pipeline on single image pair, save debug visualizations."""
    img_name = rgb_path.stem
    print(f"\n{'=' * 50}\nProcessing: {img_name}\n{'=' * 50}")

    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    h, w = depth.shape[:2]
    rgb = cv2.resize(rgb, (w, h))

    print(f"Size: {w}x{h}, Depth range: [{depth.min():.1f}, {depth.max():.1f}]")

    # Run full pipeline
    detections, contours = segment_slalom_pipes(depth, return_contours=True)
    print(f"Detected {len(detections)} pipe(s)")
    for i, det in enumerate(detections):
        print(
            f"  Pipe {i + 1}: bbox={det.bbox}, AR={det.aspect_ratio:.1f}, depth={det.depth:.1f}"
        )

    # Debug intermediates
    normalized = normalize_depth_to_uint8(depth)
    h_suppressed = suppress_horizontal_structures(normalized)
    tophat = multi_scale_black_tophat(h_suppressed)
    _, binary_raw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_cleaned = suppress_horizontal_structures(binary_raw, height_ratio=0.10)

    # Create combined mask from all detected contours
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(combined_mask, [cnt], -1, 255, -1)

    # Create visualizations
    mask_overlay = create_mask_overlay(rgb, combined_mask)
    masked_rgb = create_masked_rgb(rgb, combined_mask)

    # Save outputs
    out_dir = output_dir / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "1_rgb.png"), rgb)
    cv2.imwrite(
        str(out_dir / "2_depth.png"), cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
    )
    cv2.imwrite(str(out_dir / "3_h_suppressed.png"), h_suppressed)
    cv2.imwrite(str(out_dir / "4_tophat.png"), tophat)
    cv2.imwrite(str(out_dir / "5_binary_raw.png"), binary_raw)
    cv2.imwrite(str(out_dir / "5b_binary_cleaned.png"), binary_cleaned)

    # Annotated result
    annotated = rgb.copy()
    for det in detections:
        x, y, bw, bh = det.bbox
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            f"AR:{det.aspect_ratio:.1f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(out_dir / "6_result.png"), annotated)

    # Save mask visualizations (normal + high-res)
    cv2.imwrite(str(out_dir / "7_mask_overlay.png"), mask_overlay)
    cv2.imwrite(str(out_dir / "8_masked_rgb.png"), masked_rgb)

    # High-resolution versions for detailed inspection
    cv2.imwrite(str(out_dir / "7_mask_overlay_HIRES.png"), upscale(mask_overlay))
    cv2.imwrite(str(out_dir / "8_masked_rgb_HIRES.png"), upscale(masked_rgb))
    cv2.imwrite(str(out_dir / "1_rgb_HIRES.png"), upscale(rgb))

    # Composite
    row1 = hconcat_resize(
        [
            add_title(rgb, "RGB"),
            add_title(cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA), "Depth"),
            add_title(cv2.cvtColor(h_suppressed, cv2.COLOR_GRAY2BGR), "H-Suppress"),
        ],
        height=300,
    )
    row2 = hconcat_resize(
        [
            add_title(cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR), "TopHat"),
            add_title(cv2.cvtColor(binary_raw, cv2.COLOR_GRAY2BGR), "Binary Raw"),
            add_title(
                cv2.cvtColor(binary_cleaned, cv2.COLOR_GRAY2BGR), "Binary Cleaned"
            ),
        ],
        height=300,
    )
    row3 = hconcat_resize(
        [
            add_title(annotated, f"Result: {len(detections)} pipes"),
            add_title(mask_overlay, "Mask Overlay"),
            add_title(masked_rgb, "Masked RGB (black bg)"),
        ],
        height=300,
    )
    cv2.imwrite(str(out_dir / "COMPOSITE.png"), cv2.vconcat([row1, row2, row3]))

    print(f"Saved to: {out_dir}/")
    return len(detections)


def main():
    base = Path(__file__).parent
    rgb_dir = base / "slalom_test_images"
    depth_dir = base / "slalom_depth_test_images" / "da3mono-large"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base / f"test_segment_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    # Test ALL images (1-9)
    test_images = list(range(1, 10))
    tested = 0
    total = 0

    for n in test_images:
        rgb_path = rgb_dir / f"test_image_{n}.jpg"
        depth_path = depth_dir / f"test_image_{n}_depth.png"

        if rgb_path.exists() and depth_path.exists():
            total += test_pipeline(rgb_path, depth_path, output_dir)
            tested += 1
        else:
            print(f"Skipping test_image_{n}: missing files")

    print(
        f"\n{'=' * 50}\nTotal: {total} pipes across {tested} images\nOutput: {output_dir}\n{'=' * 50}"
    )


if __name__ == "__main__":
    main()
