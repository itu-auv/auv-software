"""
Slalom Pipe Segmentation from depth images


1. Normalize depth to uint8
2. Suppress horizontal structures (vertical morphological opening)
3. extract narrow vertical dark structures (pipes)
4. Otsu binarization
5. Geometric filtering (aspect ratio, area bounds)

"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class PipeDetection:
    """Detected pipe with metadata."""

    bbox: Tuple[int, int, int, int]
    area: float
    aspect_ratio: float
    centroid: Tuple[float, float]  # normalized to [-1, 1] for visual servoing
    confidence: float


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """
    Normalize depth map to 0-255 range for morphological operations.
    """
    d_min = np.nanmin(depth)
    d_max = np.nanmax(depth)
    # if a depth estimation is NaN, assume it is at the maximum depth
    depth_clean = np.nan_to_num(depth, nan=d_max)

    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)

    normalized = (depth_clean - d_min) / (d_max - d_min)
    return (normalized * 255).astype(np.uint8)


def suppress_horizontal_structures(
    img: np.ndarray, height_ratio: float = 0.15
) -> np.ndarray:
    """
    Remove horizontal structures using vertical opening.

    """
    h = img.shape[0]
    kernel_h = max(15, int(h * height_ratio)) | 1  # Ensure odd

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def opening_by_reconstruction(
    img: np.ndarray, kernel: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    """
    Morphological opening by reconstruction.

    Classic opening erodes then dilates, which can permanently remove structures
    at image edges. Opening by reconstruction erodes to get seeds, then uses
    geodesic dilation to restore connectivity - structures connected to seeds
    are fully restored up to the original image boundary.

    Args:
        img: Input grayscale image (uint8)
        kernel: Structuring element for initial erosion
        max_iter: Maximum iterations for geodesic dilation convergence

    Returns:
        Reconstructed image preserving connected structures
    """
    eroded = cv2.erode(img, kernel)
    reconstructed = eroded.copy()
    dilation_kernel = np.ones((3, 3), dtype=np.uint8)

    for _ in range(max_iter):
        prev = reconstructed.copy()
        dilated = cv2.dilate(reconstructed, dilation_kernel)
        reconstructed = np.minimum(dilated, img)  # never exceed mask
        if np.array_equal(reconstructed, prev):
            break

    return reconstructed


def suppress_horizontal_geodesic(
    img: np.ndarray, height_ratio: float = 0.15
) -> np.ndarray:
    """
    Remove horizontal structures using opening by reconstruction.

    Preserves pipe ends better than classic opening because geodesic dilation
    restores connectivity from eroded seeds back to the original image boundary.

    Args:
        img: Normalized depth image (uint8)
        height_ratio: Kernel height as fraction of image height

    Returns:
        Processed image with horizontal structures removed
    """
    h = img.shape[0]
    kernel_h = max(15, int(h * height_ratio)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    return opening_by_reconstruction(img, kernel)


def suppress_horizontal_hybrid(
    img: np.ndarray, height_ratio: float = 0.15, mask_threshold: int = 20
) -> np.ndarray:
    """
    Remove horizontal structures by masking in original image.

    Key insight: Uses opening only to DETECT horizontal structures, then masks
    them in the original image. Pipes are never eroded - only horizontals are
    masked out.

    Args:
        img: Normalized depth image (uint8)
        height_ratio: Kernel height as fraction of image height
        mask_threshold: Minimum intensity difference to identify horizontal structure

    Returns:
        Original image with horizontal structures masked to background
    """
    h = img.shape[0]
    kernel_h = max(15, int(h * height_ratio)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))

    # classic opening to find what gets removed
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    removed = cv2.absdiff(img, opened)

    # binary mask of horizontal structures
    h_mask = (removed > mask_threshold).astype(np.uint8)
    h_mask = cv2.dilate(h_mask, np.ones((3, 3), dtype=np.uint8))  # catch edges

    # mask horizontals in original - set to background (255 = far = bright in depth)
    result = img.copy()
    result[h_mask > 0] = 255

    return result


def multi_scale_black_tophat(
    img: np.ndarray,
    width_ratios: List[float] = [0.015, 0.025, 0.04, 0.06],
    height_ratio: float = 0.09,
) -> np.ndarray:
    """
    Multi-scale black top-hat transform.

    Args:
        img: Grayscale image (uint8)
        width_ratios: Kernel widths as fractions of image width
        height_ratio: Kernel height as fraction of image height

    Returns:
        Combined response (max across all scales)
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


def vertical_extension(
    img: np.ndarray,
    low_ratio: float = 0.35,
    cleanup_kernel_h: int = 5,
) -> Tuple[np.ndarray, float, float]:
    """
    Vertical-only extension from Otsu core detection.

    The "Droplet" method: Use Otsu for high-precision core detection,
    then extend ONLY vertically (up/down) to recover faint pipe tips.
    Cannot leak horizontally into floor - physically impossible.

    Args:
        img: Input image (typically TopHat output)
        low_ratio: T_low = T_high * low_ratio (default 0.35)
        cleanup_kernel_h: Height of vertical opening kernel for cleanup (0 to disable)

    Returns:
        Tuple of (binary_mask, t_high, t_low)
    """
    h, w = img.shape

    # Step 1: Otsu for core detection (high precision anchors)
    t_high, core_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_low = t_high * low_ratio

    # Step 2: Vertical extension from core
    result = core_mask.copy()

    for col in range(w):
        # Find ON pixels in this column
        column_core = core_mask[:, col]
        on_pixels = np.where(column_core > 0)[0]

        if len(on_pixels) == 0:
            continue

        # Find continuous segments
        segments = []
        seg_start = on_pixels[0]
        seg_end = on_pixels[0]

        for i in range(1, len(on_pixels)):
            if on_pixels[i] == seg_end + 1:
                seg_end = on_pixels[i]
            else:
                segments.append((seg_start, seg_end))
                seg_start = on_pixels[i]
                seg_end = on_pixels[i]
        segments.append((seg_start, seg_end))

        # Extend each segment up and down
        for seg_top, seg_bottom in segments:
            # Extend UP
            y = seg_top - 1
            while y >= 0 and img[y, col] >= t_low:
                result[y, col] = 255
                y -= 1

            # Extend DOWN
            y = seg_bottom + 1
            while y < h and img[y, col] >= t_low:
                result[y, col] = 255
                y += 1

    # Step 3: Vertical opening cleanup (remove horizontal noise)
    if cleanup_kernel_h > 0:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cleanup_kernel_h))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel)

    return result, float(t_high), float(t_low)


def hysteresis_threshold(
    img: np.ndarray,
    low_ratio: float = 0.4,
    use_otsu_high: bool = True,
    close_kernel_size: int = 5,
) -> Tuple[np.ndarray, float, float]:
    """
    Hysteresis thresholding with morphological reconstruction.

    Solves the "fading pipe tips" problem: pipe centers have strong signal,
    tips have weak signal. Otsu cuts off weak tips. Hysteresis keeps weak
    pixels if they're connected to strong pixels.

    WARNING: This can leak into floor if pipe base connects to seabed.
    Consider using vertical_extension() instead.

    Args:
        img: Input image (typically TopHat output)
        low_ratio: T_low = T_high * low_ratio (default 0.4)
        use_otsu_high: Use Otsu for T_high (True) or percentile-based (False)
        close_kernel_size: Kernel size for gap bridging (0 to disable)

    Returns:
        Tuple of (binary_mask, t_high, t_low)
    """
    # Calculate thresholds
    if use_otsu_high:
        t_high, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t_high = np.percentile(img[img > 0], 85) if np.any(img > 0) else 128

    t_low = t_high * low_ratio

    # Strong pixels (definite pipe)
    strong = (img >= t_high).astype(np.uint8)

    # Weak pixels (potential pipe tips)
    weak = (img >= t_low).astype(np.uint8)

    # Morphological reconstruction: grow strong into weak regions
    # This is geodesic dilation of strong constrained by weak
    marker = strong.copy()
    mask = weak
    kernel = np.ones((3, 3), dtype=np.uint8)

    for _ in range(100):  # max iterations
        prev = marker.copy()
        dilated = cv2.dilate(marker, kernel)
        marker = cv2.bitwise_and(dilated, mask)
        if np.array_equal(marker, prev):
            break

    binary = marker * 255

    # Optional: bridge small gaps from silt/dropout
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    return binary, float(t_high), float(t_low)


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


def segment_pipes(
    depth: np.ndarray,
    edge_crop_ratio: float = 0.0,
    suppress_horizontal: bool = True,
    h_suppress_method: str = "classic",
    binarization_method: str = "otsu",
    hysteresis_low_ratio: float = 0.4,
    min_aspect_ratio: float = 2.0,
    min_area_ratio: float = 0.00015,
    max_area_ratio: float = 0.05,
    save_intermediates: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Full pipeline: Depth -> Pipe detections

    Args:
        depth: Depth image (float32 or uint8/uint16 from PNG)
        edge_crop_ratio: Fraction of image to crop from edges (e.g., 0.06 = 6%)
        suppress_horizontal: Whether to remove horizontal structures first
        h_suppress_method: Method for horizontal suppression:
            - "classic": Vertical morphological opening (original)
            - "geodesic": Opening by reconstruction (preserves connectivity)
            - "hybrid": Mask detection in original image (pipes never eroded)
            - "none": Skip H-suppress entirely, pass normalized to TopHat
        binarization_method: Method for converting TopHat to binary:
            - "otsu": Global Otsu threshold (original, may truncate faint pipe tips)
            - "hysteresis": Dual-threshold with connectivity (preserves faint tips)
        hysteresis_low_ratio: For hysteresis, T_low = T_high * ratio (default 0.4)
        min_aspect_ratio: Minimum height/width for pipe candidates
        min_area_ratio: Minimum area as fraction of image
        max_area_ratio: Maximum area as fraction of image
        save_intermediates: Whether to save debug images
        output_dir: Directory for debug images

    Returns:
        Dict with 'contours' (list of detections) and intermediate results
    """
    results = {}
    h_orig, w_orig = depth.shape[:2]

    # Edge cropping
    if edge_crop_ratio > 0:
        cx = int(w_orig * edge_crop_ratio)
        cy = int(h_orig * edge_crop_ratio)
        depth = depth[cy : h_orig - cy, cx : w_orig - cx]
        results["crop_offset"] = (cx, cy)
    else:
        results["crop_offset"] = (0, 0)

    h, w = depth.shape[:2]

    # Step 1: Normalize
    normalized = normalize_depth_to_uint8(depth)
    results["1_normalized"] = normalized

    # Step 2: Suppress horizontal structures
    if suppress_horizontal and h_suppress_method != "none":
        if h_suppress_method == "classic":
            processed = suppress_horizontal_structures(normalized)
        elif h_suppress_method == "geodesic":
            processed = suppress_horizontal_geodesic(normalized)
        elif h_suppress_method == "hybrid":
            processed = suppress_horizontal_hybrid(normalized)
        else:
            raise ValueError(f"Unknown h_suppress_method: {h_suppress_method}")
        results["2_horizontal_suppressed"] = processed
    else:
        processed = normalized

    # Step 3: Multi-scale black top-hat
    tophat = multi_scale_black_tophat(processed)
    results["3_tophat"] = tophat

    # Step 4: Binarization
    if binarization_method == "otsu":
        # Classic Otsu - may truncate faint pipe tips
        otsu_thresh, binary = cv2.threshold(
            tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        results["4_binary"] = binary
        results["otsu_threshold"] = otsu_thresh
        results["threshold_method"] = "otsu"
    elif binarization_method == "hysteresis":
        # Hysteresis - preserves faint tips connected to strong centers
        binary, t_high, t_low = hysteresis_threshold(
            tophat, low_ratio=hysteresis_low_ratio
        )
        results["4_binary"] = binary
        results["otsu_threshold"] = t_high  # for compatibility
        results["hysteresis_t_high"] = t_high
        results["hysteresis_t_low"] = t_low
        results["threshold_method"] = "hysteresis"
    elif binarization_method == "vertical":
        # Vertical-only extension - preserves tips, cannot leak to floor
        binary, t_high, t_low = vertical_extension(
            tophat, low_ratio=hysteresis_low_ratio
        )
        results["4_binary"] = binary
        results["otsu_threshold"] = t_high
        results["vertical_t_high"] = t_high
        results["vertical_t_low"] = t_low
        results["threshold_method"] = "vertical"
    else:
        raise ValueError(f"Unknown binarization_method: {binarization_method}")

    # Step 5: Geometric filtering
    contours = geometric_filtering(
        binary,
        min_aspect_ratio=min_aspect_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    results["contours"] = contours

    # Convert to PipeDetection objects
    detections = []
    for c in contours:
        x, y, bw, bh = c["bbox"]
        cx_img = x + bw / 2
        cy_img = y + bh / 2

        # Normalize centroid to [-1, 1]
        norm_cx = (cx_img / w - 0.5) * 2
        norm_cy = (cy_img / h - 0.5) * 2

        # Confidence based on area and aspect ratio
        conf = min(1.0, (c["area"] / (h * w * 0.01)) * min(c["aspect_ratio"] / 5, 1.0))

        det = PipeDetection(
            bbox=c["bbox"],
            area=c["area"],
            aspect_ratio=c["aspect_ratio"],
            centroid=(norm_cx, norm_cy),
            confidence=conf,
        )
        detections.append(det)

    results["detections"] = detections

    # Save intermediates if requested
    if save_intermediates and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_dir / "1_normalized.png"), normalized)
        if suppress_horizontal:
            cv2.imwrite(str(output_dir / "2_horizontal_suppressed.png"), processed)
        cv2.imwrite(str(output_dir / "3_tophat.png"), tophat)
        cv2.imwrite(str(output_dir / "4_binary.png"), binary)

        # Annotated visualization
        vis = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        for c in contours:
            x, y, bw, bh = c["bbox"]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"AR:{c['aspect_ratio']:.1f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        cv2.imwrite(str(output_dir / "5_annotated.png"), vis)

    return results


def main():
    """Test with sample depth images."""
    import argparse

    parser = argparse.ArgumentParser(description="Slalom pipe segmentation")
    parser.add_argument(
        "--input", type=str, required=True, help="Depth image or directory"
    )
    parser.add_argument(
        "--output", type=str, default="segmentation_results", help="Output directory"
    )
    parser.add_argument(
        "--edge-crop", type=float, default=0.0, help="Edge crop ratio (e.g., 0.06)"
    )
    parser.add_argument(
        "--no-horizontal-suppress",
        action="store_true",
        help="Disable horizontal suppression",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_dir():
        images = list(input_path.glob("*.png"))
    else:
        images = [input_path]

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        depth = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        print(f"  Shape: {depth.shape}, Range: [{depth.min():.1f}, {depth.max():.1f}]")

        result = segment_pipes(
            depth,
            edge_crop_ratio=args.edge_crop,
            suppress_horizontal=not args.no_horizontal_suppress,
            save_intermediates=True,
            output_dir=output_dir / img_path.stem,
        )

        print(f"  Otsu threshold: {result['otsu_threshold']:.1f}")
        print(f"  Detected {len(result['contours'])} pipe(s)")

        for i, det in enumerate(result["detections"]):
            print(
                f"    Pipe {i + 1}: bbox={det.bbox}, aspect={det.aspect_ratio:.1f}, conf={det.confidence:.2f}"
            )


if __name__ == "__main__":
    main()
