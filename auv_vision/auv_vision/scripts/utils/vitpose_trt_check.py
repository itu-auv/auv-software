#!/usr/bin/env python3
"""Parity + timing: TensorRT engine vs torch checkpoint on a real image.

    python3 vitpose_trt_check.py <ckpt.pth> <model.engine> [--image img.jpg] [--bbox x,y,w,h] [--n 50]

Joint models: keypoint px diff, score diff, mask IoU. Objectness: box + score
+ prob-map diff. Then per-call timings for both backends.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vitpose_inference import (
    load_objectness,
    load_vitpose,
    load_sidecar,
)  # noqa: E402


def timeit(fn, n):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt")
    ap.add_argument("engine")
    ap.add_argument("--image", default=None)
    ap.add_argument(
        "--bbox", default=None, help="x,y,w,h for joint models (default: full frame)"
    )
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)[:, :, ::-1].copy()
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        print("no --image: using random noise (parity still meaningful, boxes not)")
    kind = load_sidecar(args.engine)["kind"]

    if kind == "joint":
        ref = load_vitpose(args.ckpt, device="cuda")
        trt = load_vitpose(args.engine)
        h, w = img.shape[:2]
        bbox = (
            tuple(float(v) for v in args.bbox.split(","))
            if args.bbox
            else (0.0, 0.0, float(w), float(h))
        )
        k1, s1, m1 = ref.predict(img, bbox)
        k2, s2, m2 = trt.predict(img, bbox)
        d = np.linalg.norm(k1 - k2, axis=1)
        print(
            f"keypoints: mean |dpx| {d.mean():.3f}, max {d.max():.3f}; scores max|d| {np.abs(s1 - s2).max():.4f}"
        )
        if m1 is not None:
            b1, b2 = m1 > ref.mask_threshold, m2 > trt.mask_threshold
            inter, union = (b1 & b2).sum(), (b1 | b2).sum()
            print(
                f"masks: IoU {inter / max(union, 1):.4f}, max|dprob| {np.abs(m1 - m2).max():.4f}"
            )
        print(
            f"timing: torch {timeit(lambda: ref.predict(img, bbox), args.n):.1f} ms/call, "
            f"TRT {timeit(lambda: trt.predict(img, bbox), args.n):.1f} ms/call (incl. pre/post)"
        )
    else:
        ref = load_objectness(args.ckpt, device="cuda")
        trt = load_objectness(args.engine)
        b1, s1, p1 = ref.predict(img, return_prob=True)
        b2, s2, p2 = trt.predict(img, return_prob=True)
        print(
            f"box torch {b1} score {s1:.3f}\nbox TRT   {b2} score {s2:.3f}\nprob map max|d| {np.abs(p1 - p2).max():.4f}"
        )
        print(
            f"timing: torch {timeit(lambda: ref.predict(img), args.n):.1f} ms/call, "
            f"TRT {timeit(lambda: trt.predict(img), args.n):.1f} ms/call"
        )


if __name__ == "__main__":
    main()
