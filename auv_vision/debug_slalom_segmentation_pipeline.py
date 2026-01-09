"""
Pipeline Debug Visualization (OpenCV only - no matplotlib)

Her aşamayı adım adım görselleştirir ve kaydeder.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from slalom_segmentation import (
    normalize_depth_to_uint8,
    suppress_horizontal_structures,
    multi_scale_black_tophat,
    geometric_filtering,
)


def add_title(img, title, font_scale=0.7):
    """Görüntüye başlık ekle."""
    h, w = img.shape[:2]
    # Üste siyah bar
    bar_h = 35
    result = np.zeros((h + bar_h, w, 3), dtype=np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result[bar_h:, :] = img
    cv2.putText(
        result,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
    )
    return result


def apply_colormap(img, cmap=cv2.COLORMAP_PLASMA):
    """Grayscale'e colormap uygula."""
    return cv2.applyColorMap(img, cmap)


def hconcat_resize(images, height=400):
    """Görüntüleri yatay birleştir, aynı yüksekliğe getir."""
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * height / h)
        resized.append(cv2.resize(img, (new_w, height)))
    return cv2.hconcat(resized)


def vconcat_resize(images, width=800):
    """Görüntüleri dikey birleştir, aynı genişliğe getir."""
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_h = int(h * width / w)
        resized.append(cv2.resize(img, (width, new_h)))
    return cv2.vconcat(resized)


def debug_pipeline(rgb_path: Path, depth_path: Path, output_dir: Path):
    """Tek görüntü için debug visualization."""

    img_name = rgb_path.stem
    print(f"\n{'=' * 60}")
    print(f"Processing: {img_name}")
    print(f"{'=' * 60}")

    # Load
    rgb_orig = cv2.imread(str(rgb_path))
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

    h, w = depth_raw.shape[:2]
    rgb_h, rgb_w = rgb_orig.shape[:2]

    # Resize RGB to match depth dimensions for annotation overlay
    rgb = cv2.resize(rgb_orig, (w, h))
    print(f"Image size: {w}x{h}")
    print(f"Depth range: [{depth_raw.min():.2f}, {depth_raw.max():.2f}]")

    # ========== PIPELINE STAGES ==========

    # Step 1: Normalize
    normalized = normalize_depth_to_uint8(depth_raw)
    print(
        f"\n[Step 1] Normalized to uint8: range [{normalized.min()}, {normalized.max()}]"
    )

    # Step 2: Horizontal suppression
    kernel_h_suppress = max(15, int(h * 0.15)) | 1
    h_suppressed = suppress_horizontal_structures(normalized)
    print(f"[Step 2] Horizontal suppression: kernel (1, {kernel_h_suppress})")

    # Step 2b: Difference (removed structures)
    diff = cv2.absdiff(normalized, h_suppressed)
    print(f"         Removed pixels: {np.sum(diff > 10)} (threshold > 10)")

    # Step 3: Multi-scale top-hat
    width_ratios = [0.015, 0.025, 0.04, 0.06]
    height_ratio = 0.09
    kernel_h_tophat = max(25, int(h * height_ratio)) | 1
    kernel_widths = [max(5, int(w * r)) | 1 for r in width_ratios]

    tophat = multi_scale_black_tophat(h_suppressed)
    print(f"[Step 3] Multi-scale top-hat:")
    print(f"         Kernel height: {kernel_h_tophat}")
    print(f"         Kernel widths: {kernel_widths}")
    print(f"         Output range: [{tophat.min()}, {tophat.max()}]")

    # Step 4: Otsu
    otsu_thresh, binary = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"[Step 4] Otsu threshold: {otsu_thresh:.0f}")
    print(f"         White pixels: {np.sum(binary > 0)}")

    # Step 5: Geometric filtering
    contours = geometric_filtering(binary, min_aspect_ratio=2.0)
    print(f"[Step 5] Geometric filtering: {len(contours)} pipe(s) detected")
    for i, c in enumerate(contours):
        x, y, bw, bh = c["bbox"]
        print(
            f"         Pipe {i + 1}: bbox=({x},{y},{bw},{bh}), "
            f"area={c['area']:.0f}, AR={c['aspect_ratio']:.2f}"
        )

    # ========== INDIVIDUAL TOP-HAT SCALES ==========

    print(f"\n[Top-Hat Breakdown]")
    individual_tophats = []
    for i, w_ratio in enumerate(width_ratios):
        kernel_w = max(5, int(w * w_ratio)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h_tophat))
        th = cv2.morphologyEx(h_suppressed, cv2.MORPH_BLACKHAT, kernel)
        individual_tophats.append(th)
        print(f"         Scale {i + 1} (kw={kernel_w}): max response = {th.max()}")

    # ========== SAVE INDIVIDUAL STAGES ==========

    out_subdir = output_dir / img_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_subdir / "0_original_rgb.png"), rgb)
    cv2.imwrite(
        str(out_subdir / "1_depth_raw_colorized.png"), apply_colormap(normalized)
    )
    cv2.imwrite(str(out_subdir / "2_normalized.png"), normalized)
    cv2.imwrite(str(out_subdir / "3_horizontal_suppressed.png"), h_suppressed)
    cv2.imwrite(
        str(out_subdir / "3b_removed_horizontal.png"),
        apply_colormap(diff, cv2.COLORMAP_HOT),
    )
    cv2.imwrite(str(out_subdir / "4_tophat_combined.png"), tophat)
    cv2.imwrite(str(out_subdir / "5_binary_otsu.png"), binary)

    for i, th in enumerate(individual_tophats):
        cv2.imwrite(str(out_subdir / f"4_{i + 1}_tophat_kw{kernel_widths[i]}.png"), th)

    # Annotated result
    annotated = rgb.copy()
    for c in contours:
        x, y, bw, bh = c["bbox"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            f"AR:{c['aspect_ratio']:.1f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(out_subdir / "6_final_annotated.png"), annotated)

    # ========== COMPOSITE IMAGE ==========

    # Row 1: RGB, Depth, Normalized, H-Suppressed
    row1_imgs = [
        add_title(rgb, "1. Original RGB"),
        add_title(
            apply_colormap(normalized),
            f"2. Depth [{depth_raw.min():.1f}-{depth_raw.max():.1f}]",
        ),
        add_title(cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), "3. Normalized"),
        add_title(
            cv2.cvtColor(h_suppressed, cv2.COLOR_GRAY2BGR),
            f"4. H-Suppress k=(1,{kernel_h_suppress})",
        ),
    ]
    row1 = hconcat_resize(row1_imgs, height=300)

    # Row 2: Diff, TopHat, Binary, Annotated
    row2_imgs = [
        add_title(apply_colormap(diff, cv2.COLORMAP_HOT), "Removed (horizontal)"),
        add_title(
            cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR), f"5. TopHat kw={kernel_widths}"
        ),
        add_title(
            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), f"6. Otsu t={otsu_thresh:.0f}"
        ),
        add_title(annotated, f"7. Result: {len(contours)} pipe(s)"),
    ]
    row2 = hconcat_resize(row2_imgs, height=300)

    # Combine
    composite = cv2.vconcat([row1, row2])
    cv2.imwrite(str(out_subdir / "COMPOSITE.png"), composite)

    # ========== TOP-HAT SCALES COMPOSITE ==========

    th_imgs = [
        add_title(
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
            f"kw={kernel_widths[i]} ({width_ratios[i] * 100:.1f}%)",
        )
        for i, th in enumerate(individual_tophats)
    ]
    th_imgs.append(
        add_title(cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR), "Combined (max)")
    )

    th_row1 = hconcat_resize(th_imgs[:3], height=250)
    th_row2 = hconcat_resize(
        th_imgs[3:]
        + [add_title(cv2.cvtColor(h_suppressed, cv2.COLOR_GRAY2BGR), "Input")],
        height=250,
    )
    tophat_composite = cv2.vconcat([th_row1, th_row2])
    cv2.imwrite(str(out_subdir / "TOPHAT_SCALES.png"), tophat_composite)

    print(f"\n✓ Saved to: {out_subdir}/")
    return len(contours)


def main():
    base = Path(__file__).parent
    rgb_dir = base / "slalom_test_images"
    depth_dir = base / "slalom_depth_test_images" / "da3mono-large"

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base / f"debug_output_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    test_images = [3, 4, 5, 6]
    total_pipes = 0

    for img_num in test_images:
        rgb_path = rgb_dir / f"test_image_{img_num}.jpg"
        depth_path = depth_dir / f"test_image_{img_num}_depth.png"

        if not rgb_path.exists():
            print(f"ERROR: RGB not found: {rgb_path}")
            continue
        if not depth_path.exists():
            print(f"ERROR: Depth not found: {depth_path}")
            continue

        pipes = debug_pipeline(rgb_path, depth_path, output_dir)
        total_pipes += pipes

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_pipes} pipes detected across {len(test_images)} images")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
