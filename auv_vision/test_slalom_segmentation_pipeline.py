#!/usr/bin/env python3
"""
Slalom Segmentation Pipeline Test - visualizes each stage and saves debug output.
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
    detections = segment_slalom_pipes(depth)
    print(f"Detected {len(detections)} pipe(s)")
    for i, det in enumerate(detections):
        print(
            f"  Pipe {i + 1}: bbox={det.bbox}, AR={det.aspect_ratio:.1f}, conf={det.confidence:.2f}"
        )

    # Debug intermediates
    normalized = normalize_depth_to_uint8(depth)
    h_suppressed = suppress_horizontal_structures(normalized)
    tophat = multi_scale_black_tophat(h_suppressed)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save outputs
    out_dir = output_dir / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "1_rgb.png"), rgb)
    cv2.imwrite(
        str(out_dir / "2_depth.png"), cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
    )
    cv2.imwrite(str(out_dir / "3_h_suppressed.png"), h_suppressed)
    cv2.imwrite(str(out_dir / "4_tophat.png"), tophat)
    cv2.imwrite(str(out_dir / "5_binary.png"), binary)

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
            add_title(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Binary"),
            add_title(annotated, f"Result: {len(detections)} pipes"),
        ],
        height=300,
    )
    cv2.imwrite(str(out_dir / "COMPOSITE.png"), cv2.vconcat([row1, row2]))

    print(f"Saved to: {out_dir}/")
    return len(detections)


def main():
    base = Path(__file__).parent
    rgb_dir = base / "slalom_test_images"
    depth_dir = base / "slalom_depth_test_images" / "da3mono-large"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base / f"test_output_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    test_images = [3, 4, 5, 6]
    total = sum(
        test_pipeline(
            rgb_dir / f"test_image_{n}.jpg",
            depth_dir / f"test_image_{n}_depth.png",
            output_dir,
        )
        for n in test_images
        if (rgb_dir / f"test_image_{n}.jpg").exists()
        and (depth_dir / f"test_image_{n}_depth.png").exists()
    )

    print(
        f"\n{'=' * 50}\nTotal: {total} pipes across {len(test_images)} images\nOutput: {output_dir}\n{'=' * 50}"
    )


if __name__ == "__main__":
    main()
