#!/usr/bin/env python3
"""Evaluate the pipe segmentation on a single image (no ROS).

  python3 seg_eval.py <image> [--params p.yaml] [--gt gt.png] [--out-dir DIR]

Writes ``<name>_mask.png`` and ``<name>_overlay.png`` (original | mask | overlay)
into the output directory. If a ground-truth mask is given, prints IoU /
precision / recall. This is the agent's "look at the result" command.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipe_segmentation import (  # noqa: E402
    segment,
    make_debug_overlay,
    mask_metrics,
    binarize,
    params_to_dict,
    SegParams,
)
from seg_tuning.params_io import load_params  # noqa: E402


def error_map(bgr, pred, gt):
    """[original | GT | pred | error] where green=correct, red=false-positive,
    blue=false-negative."""
    pred, gt = binarize(pred), binarize(gt)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    g, pr = gt > 0, pred > 0
    err = bgr.copy()
    err[g & pr] = (0, 200, 0)
    err[pr & ~g] = (0, 0, 255)
    err[g & ~pr] = (255, 0, 0)
    return np.hstack([bgr, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), err])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--params", help="YAML param file (cfg/dynparam schema)")
    ap.add_argument("--gt", help="ground-truth mask PNG for metrics")
    ap.add_argument("--out-dir", default=None,
                    help="where to write outputs (default: alongside the image)")
    ap.add_argument("--diag", action="store_true",
                    help="with --gt, also write an error map "
                         "(green=correct, red=false-positive, blue=false-negative)")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit("could not read image: %s" % args.image)

    params = load_params(args.params) if args.params else SegParams()
    mask = segment(bgr, params)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    mask_path = os.path.join(out_dir, base + "_mask.png")
    overlay_path = os.path.join(out_dir, base + "_overlay.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, make_debug_overlay(bgr, mask))

    print("params:")
    for k, v in sorted(params_to_dict(params).items()):
        print("  %-16s %s" % (k, v))
    print("mask    -> %s" % mask_path)
    print("overlay -> %s" % overlay_path)

    if args.gt:
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            sys.exit("could not read gt: %s" % args.gt)
        m = mask_metrics(mask, gt)
        print("metrics: IoU=%.3f  precision=%.3f  recall=%.3f"
              % (m["iou"], m["precision"], m["recall"]))
        if args.diag:
            diag_path = os.path.join(out_dir, base + "_diag.png")
            cv2.imwrite(diag_path, error_map(bgr, mask, gt))
            print("diag    -> %s" % diag_path)


if __name__ == "__main__":
    main()
