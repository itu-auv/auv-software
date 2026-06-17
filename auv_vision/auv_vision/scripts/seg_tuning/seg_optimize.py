#!/usr/bin/env python3
"""Auto-tune the pipe segmentation against hand-drawn ground truth (no ROS).

  # single frame from the GT web UI session dir:
  python3 seg_optimize.py --session /tmp/seg_session --out best_params.yaml

  # explicit image/GT pairs (matched by order); averages IoU across all of them
  # to avoid overfitting to one near/far frame:
  python3 seg_optimize.py near.png far.png --gt near_gt.png --gt far_gt.png \
          --out best_params.yaml

Search: a coarse colour-threshold grid at reduced resolution, then a morphology /
component pass and a local colour refine at full resolution. Writes a
dynparam-loadable YAML plus a short report, and (with --eval) overlays.
"""

import argparse
import glob
import itertools
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipe_segmentation import (  # noqa: E402
    segment,
    make_debug_overlay,
    mask_metrics,
    binarize,
    SegParams,
    params_to_dict,
)
from seg_tuning.params_io import dump_params  # noqa: E402

COARSE_WIDTH = 480  # downscale for the (large) colour grid; thresholds transfer


def collect_session(session_dir):
    """Return [(image_path, gt_path), ...] from a session/dataset directory.

    Supports two layouts:
      * single pair:  <dir>/image.png + <dir>/gt.png
      * dataset:      <dir>/sample_NNNN/{image.png,gt.png}  (from the GT UI)
    """
    pairs = []
    single_img = os.path.join(session_dir, "image.png")
    single_gt = os.path.join(session_dir, "gt.png")
    if os.path.exists(single_img) and os.path.exists(single_gt):
        pairs.append((single_img, single_gt))
    for d in sorted(glob.glob(os.path.join(session_dir, "sample_*"))):
        img, gt = os.path.join(d, "image.png"), os.path.join(d, "gt.png")
        if os.path.exists(img) and os.path.exists(gt):
            pairs.append((img, gt))
    return pairs


def load_pairs(images, gts):
    pairs = []
    for img_path, gt_path in zip(images, gts):
        bgr = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            sys.exit("could not read image: %s" % img_path)
        if gt is None:
            sys.exit("could not read gt: %s" % gt_path)
        gt = binarize(gt)
        if np.count_nonzero(gt) == 0:
            print("WARNING: %s is all background (did you paint the pipe?)" % gt_path)
        pairs.append((bgr, gt, img_path))
    return pairs


def resized_pairs(pairs, width):
    out = []
    for bgr, gt, name in pairs:
        h, w = bgr.shape[:2]
        if width and width < w:
            s = width / float(w)
            b = cv2.resize(bgr, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            g = cv2.resize(gt, (b.shape[1], b.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        else:
            b, g = bgr, gt
        out.append((b, g, name))
    return out


def mean_iou(params, pairs):
    if not pairs:
        return 0.0
    tot = 0.0
    for bgr, gt, _ in pairs:
        tot += mask_metrics(segment(bgr, params), gt)["iou"]
    return tot / len(pairs)


def search(grid, base, pairs, label):
    """Brute-force over the cartesian product of grid (dict name->values)."""
    keys = list(grid.keys())
    best, best_iou = base, mean_iou(base, pairs)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    t0 = time.time()
    for i, combo in enumerate(combos):
        cand = SegParams(**vars(base))
        ok = True
        for k, v in zip(keys, combo):
            setattr(cand, k, v)
        if cand.h_min > cand.h_max:  # keep hue non-wrapping during search
            ok = False
        if not ok:
            continue
        iou = mean_iou(cand, pairs)
        if iou > best_iou:
            best, best_iou = cand, iou
    print("  [%s] %d combos in %.1fs -> best IoU %.3f"
          % (label, len(combos), time.time() - t0, best_iou))
    return best, best_iou


def optimize(pairs_full):
    coarse = resized_pairs(pairs_full, COARSE_WIDTH)

    # Stage A: colour thresholds, reduced resolution, minimal morphology.
    base = SegParams(blur_ksize=3, open_kernel=0, close_kernel=0,
                     min_area_px=40, keep_largest=False, max_components=3,
                     fill_holes=False, s_max=255, v_max=255, lab_b_max=255)
    grid_a = {
        "h_min": [12, 16, 20, 24],
        "h_max": [36, 42, 48, 55],
        "s_min": [30, 60, 90, 130],
        "v_min": [90, 130, 160, 190],
        "lab_b_min": [150, 165, 180, 195],
        "combine_mode": ["hsv_only", "and", "or"],
    }
    best, _ = search(grid_a, base, coarse, "colour-coarse")

    # Stage B: morphology + component filter at full resolution.
    best.blur_ksize = 5
    grid_b = {
        "open_kernel": [0, 3, 5, 9],
        "close_kernel": [0, 11, 21, 31],
        "min_area_px": [200, 800, 2000, 5000],
        "max_components": [1, 2, 3],
        "fill_holes": [True, False],
    }
    best, _ = search(grid_b, best, pairs_full, "morphology")

    # Stage C: optional shape-based tether rejection at full resolution.
    grid_c = {
        "min_fill": [0.0, 0.10, 0.18],
        "max_aspect": [0.0, 12.0],
    }
    best, _ = search(grid_c, best, pairs_full, "shape-filter")

    # Stage D: local colour refine at full resolution.
    grid_d = {
        "h_min": sorted({max(0, best.h_min - 3), best.h_min, best.h_min + 3}),
        "h_max": sorted({best.h_max - 3, best.h_max, min(179, best.h_max + 3)}),
        "s_min": sorted({max(0, best.s_min - 15), best.s_min, best.s_min + 15}),
        "v_min": sorted({max(0, best.v_min - 15), best.v_min, best.v_min + 15}),
        "lab_b_min": sorted({max(0, best.lab_b_min - 10), best.lab_b_min,
                             min(255, best.lab_b_min + 10)}),
    }
    best, best_iou = search(grid_d, best, pairs_full, "colour-refine")
    return best, best_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image file(s)")
    ap.add_argument("--gt", action="append", default=[],
                    help="ground-truth mask(s), matched to images by order")
    ap.add_argument("--session", help="dir containing image.png and gt.png")
    ap.add_argument("--out", default="best_params.yaml")
    ap.add_argument("--eval", action="store_true",
                    help="also write overlays for each image with the best params")
    args = ap.parse_args()

    images, gts = list(args.images), list(args.gt)
    if args.session:
        session_pairs = collect_session(args.session)
        if not session_pairs:
            sys.exit("no image/gt pairs found under %s (expected image.png+gt.png "
                     "or sample_*/ subdirs)" % args.session)
        images += [p[0] for p in session_pairs]
        gts += [p[1] for p in session_pairs]
    if not images:
        sys.exit("no images: pass image+--gt pairs or --session DIR")
    if len(images) != len(gts):
        sys.exit("got %d images but %d --gt; they must pair up"
                 % (len(images), len(gts)))

    pairs = load_pairs(images, gts)
    print("optimizing over %d image(s): %s"
          % (len(pairs), ", ".join(os.path.basename(p[2]) for p in pairs)))

    best, best_iou = optimize(pairs)

    # per-image breakdown with the winning params
    lines = ["best mean IoU=%.3f over %d image(s)" % (best_iou, len(pairs))]
    for bgr, gt, name in pairs:
        m = mask_metrics(segment(bgr, best), gt)
        lines.append("  %-28s IoU=%.3f P=%.3f R=%.3f"
                     % (os.path.basename(name), m["iou"],
                        m["precision"], m["recall"]))
    report = " | ".join(lines)
    dump_params(best, args.out, report=lines[0])

    print("\n".join(lines))
    print("\nparams:")
    for k, v in sorted(params_to_dict(best).items()):
        print("  %-16s %s" % (k, v))
    print("\nwrote %s" % args.out)
    print("push to robot:  rosrun dynamic_reconfigure dynparam load "
          "/taluy/opencv_seg_publisher %s" % args.out)

    if args.eval:
        for bgr, _gt, name in pairs:
            out = os.path.splitext(name)[0] + "_best_overlay.png"
            cv2.imwrite(out, make_debug_overlay(bgr, segment(bgr, best)))
            print("overlay -> %s" % out)


if __name__ == "__main__":
    main()
