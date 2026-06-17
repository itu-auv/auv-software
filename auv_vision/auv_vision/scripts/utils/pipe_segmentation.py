#!/usr/bin/env python3
"""ROS-free OpenCV segmentation core for the yellow pipe.

This module is shared by three consumers (see seg_tuning/README.md):
  * the live ROS node  (opencv_seg_publisher.py)  -> runs on the robot Jetson
  * the offline tools  (seg_tuning/seg_eval.py, seg_optimize.py) -> run on the laptop
  * the GT drawing UI  (seg_gt_ui_node.py)         -> runs on the laptop

Having a single segmentation implementation means whatever the offline optimizer
tunes against the ground-truth mask is byte-for-byte what the live node produces.

It intentionally has NO rospy / ROS dependency so it can be imported and unit
tested anywhere (only numpy + OpenCV).

The yellow pipe is detected by colour, combining two complementary cues:
  * HSV inRange  - the primary signal for the bright, saturated near pipe.
  * Lab b-channel threshold - yellow has a high b ("yellowness") even when it is
    faded / desaturated by turbidity, so this rescues the distant pale pipe that
    HSV (low saturation) would miss.
The cues are merged with ``combine_mode`` and cleaned up with morphology and a
connected-component filter that rejects the thin orange tether and speckle while
filling the white marker holes on the pipe.
"""

from dataclasses import dataclass, fields, asdict

import cv2
import numpy as np

# combine_mode is stored on SegParams as a readable string, but serialised to an
# int at the cfg / yaml / dynparam boundary (dynamic_reconfigure enums are ints).
# The list index IS the int value -> keep this order stable and in sync with
# cfg/OpenCVSeg.cfg.
COMBINE_MODES = ["hsv_only", "lab_only", "and", "or"]


def combine_mode_to_int(mode):
    if isinstance(mode, int):
        return mode
    return COMBINE_MODES.index(mode)


def combine_mode_to_str(mode):
    if isinstance(mode, str):
        return mode
    return COMBINE_MODES[int(mode)]


@dataclass
class SegParams:
    """All tunable fields. Names match cfg/OpenCVSeg.cfg exactly so a dict round
    trips cleanly between the cfg callback, yaml files and dynparam."""

    # --- preprocessing ---
    blur_ksize: int = 5          # gaussian blur kernel (<=1 disables, forced odd)
    resize_width: int = 0        # downscale to this width before segmenting (0 = off)

    # --- HSV colour mask (OpenCV ranges: H 0-179, S/V 0-255) ---
    # Defaults calibrated on ornek_fotolar/: the pipe is far brighter (V~210-243)
    # than the olive/brown water (V~105), so a high v_min is the strongest cue.
    h_min: int = 18
    h_max: int = 44              # if h_min > h_max the hue range wraps around 180
    s_min: int = 70
    s_max: int = 255
    v_min: int = 150
    v_max: int = 255

    # --- Lab b-channel mask (b 0-255, higher = more yellow) ---
    lab_b_min: int = 180
    lab_b_max: int = 255

    # how to merge the HSV and Lab masks ("and" leans precision: both cues must agree)
    combine_mode: str = "and"

    # --- morphology (ellipse kernels; <=1 disables) ---
    open_kernel: int = 5         # erode-then-dilate: removes thin tether / speckle
    close_kernel: int = 21       # dilate-then-erode: fills white marker holes

    # --- connected-component filter ---
    min_area_px: int = 1000      # drop blobs smaller than this
    keep_largest: bool = False   # True: keep only the single largest blob
    max_components: int = 2      # else keep this many largest blobs
    max_aspect: float = 0.0      # reject blobs with bbox aspect > this (0 = off)
    min_fill: float = 0.0        # reject blobs with area/bbox_area < this (0 = off)
    fill_holes: bool = True      # flood-fill internal holes after filtering


# ---------------------------------------------------------------------------
# (de)serialisation helpers - the common schema across cfg / yaml / dynparam
# ---------------------------------------------------------------------------
def params_from_dict(d):
    """Build SegParams from a dict, ignoring unknown keys and coercing types.

    Accepts combine_mode as either an int (cfg/dynparam) or a string (yaml)."""
    p = SegParams()
    valid = {f.name: f.type for f in fields(SegParams)}
    for k, v in (d or {}).items():
        if k not in valid:
            continue
        if k == "combine_mode":
            setattr(p, k, combine_mode_to_str(v))
            continue
        t = valid[k]
        if t is int:
            setattr(p, k, int(round(float(v))))
        elif t is float:
            setattr(p, k, float(v))
        elif t is bool:
            setattr(p, k, bool(v))
        else:
            setattr(p, k, v)
    return p


def params_to_dict(p):
    """Serialise SegParams to a cfg/dynparam-compatible dict (combine_mode int)."""
    d = asdict(p)
    d["combine_mode"] = combine_mode_to_int(p.combine_mode)
    return d


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _hsv_mask(hsv, p):
    s_lo, s_hi = p.s_min, p.s_max
    v_lo, v_hi = p.v_min, p.v_max
    if p.h_min <= p.h_max:
        lower = np.array([p.h_min, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([p.h_max, s_hi, v_hi], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)
    # wraparound hue: [h_min..179] OR [0..h_max]
    m1 = cv2.inRange(hsv, np.array([p.h_min, s_lo, v_lo], dtype=np.uint8),
                     np.array([179, s_hi, v_hi], dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([0, s_lo, v_lo], dtype=np.uint8),
                     np.array([p.h_max, s_hi, v_hi], dtype=np.uint8))
    return cv2.bitwise_or(m1, m2)


def _lab_mask(lab, p):
    b = lab[:, :, 2]
    return cv2.inRange(b, int(p.lab_b_min), int(p.lab_b_max))


def _color_mask(bgr, p):
    mode = combine_mode_to_str(p.combine_mode)
    if mode == "hsv_only":
        return _hsv_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), p)
    if mode == "lab_only":
        return _lab_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab), p)
    hsv = _hsv_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), p)
    lab = _lab_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab), p)
    if mode == "and":
        return cv2.bitwise_and(hsv, lab)
    return cv2.bitwise_or(hsv, lab)  # "or"


def _morphology(mask, p):
    if p.open_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(p.open_kernel), _odd(p.open_kernel)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.close_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(p.close_kernel), _odd(p.close_kernel)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _fill_holes(mask):
    """Flood fill from the border, the un-reached background pixels are holes."""
    h, w = mask.shape[:2]
    ff = mask.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


def _component_filter(mask, p):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = []  # (area, label)
    for lbl in range(1, num):  # 0 is background
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < p.min_area_px:
            continue
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if p.max_aspect > 0.0 and min(bw, bh) > 0:
            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect > p.max_aspect:
                continue
        if p.min_fill > 0.0 and bw * bh > 0:
            fill = area / float(bw * bh)
            if fill < p.min_fill:
                continue
        keep.append((area, lbl))

    if not keep:
        return np.zeros_like(mask)

    keep.sort(reverse=True)  # largest area first
    n = 1 if p.keep_largest else max(1, int(p.max_components))
    keep = keep[:n]

    out = np.zeros_like(mask)
    for _, lbl in keep:
        out[labels == lbl] = 255
    return out


def segment(bgr, p):
    """Segment the yellow pipe.

    Args:
        bgr: HxWx3 uint8 BGR image.
        p:   SegParams.
    Returns:
        mono8 mask, 0 or 255, same H/W as the input (upscaled back if resized).
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((1, 1), np.uint8)

    full_h, full_w = bgr.shape[:2]
    work = bgr
    if p.resize_width and p.resize_width > 0 and p.resize_width < full_w:
        scale = p.resize_width / float(full_w)
        work = cv2.resize(bgr, (0, 0), fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    if p.blur_ksize and p.blur_ksize > 1:
        k = _odd(p.blur_ksize)
        work = cv2.GaussianBlur(work, (k, k), 0)

    mask = _color_mask(work, p)
    mask = _morphology(mask, p)
    mask = _component_filter(mask, p)
    if p.fill_holes:
        mask = _fill_holes(mask)

    if mask.shape[:2] != (full_h, full_w):
        mask = cv2.resize(mask, (full_w, full_h),
                          interpolation=cv2.INTER_NEAREST)
    return mask


# ---------------------------------------------------------------------------
# visualisation + metrics (used by the eval / optimize tools and node debug)
# ---------------------------------------------------------------------------
def make_debug_overlay(bgr, mask, alpha=0.5):
    """Return a side-by-side [original | mask | overlay] BGR image."""
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(bgr)
    tint[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(bgr, 1.0, tint, alpha, 0.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return np.hstack([bgr, mask3, overlay])


def binarize(mask):
    """Coerce any single-channel image to a clean 0/255 uint8 mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.uint8) * 255


def mask_metrics(pred, gt):
    """IoU / precision / recall between two 0/255 masks (resized to match)."""
    pred = binarize(pred)
    gt = binarize(gt)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    p = pred > 0
    g = gt > 0
    inter = float(np.count_nonzero(p & g))
    union = float(np.count_nonzero(p | g))
    pred_pos = float(np.count_nonzero(p))
    gt_pos = float(np.count_nonzero(g))
    iou = inter / union if union > 0 else 1.0
    precision = inter / pred_pos if pred_pos > 0 else (1.0 if gt_pos == 0 else 0.0)
    recall = inter / gt_pos if gt_pos > 0 else 1.0
    return {"iou": iou, "precision": precision, "recall": recall}
