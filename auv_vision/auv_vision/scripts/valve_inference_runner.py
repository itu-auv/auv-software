#!/usr/bin/env python3
"""
Standalone ViTPose inference viewer for the valve dataset.

Loads the trained ViTPose-B model and runs per-image inference on the output
of sim_auto_labeler.py, displaying both ground-truth labels and model
predictions side-by-side on each image.

Usage:
    python3 valve_inference_runner.py
    python3 valve_inference_runner.py --dataset ~/yolo_valve_dataset --model ./best.pth --conf 0.3

Controls:
    D / Right arrow  — next image
    A / Left arrow   — previous image
    S                — save annotated image to dataset/inference/
    Q                — quit
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np

# Import ValvePose from the same directory as this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vitpose_inference import ValvePose

BBOX_COLOR      = (220, 180,   0)   # yellow  — GT bbox
GT_KP_COLOR     = (255,   0,   0)   # blue    — GT keypoints
PRED_KP_COLOR   = (255, 220,   0)   # cyan    — predicted keypoints
SKELETON_COLOR  = (  0, 255,   0)   # green   — predicted skeleton

SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4),
            (4, 5), (5, 6), (6, 7), (7, 0)]

KEY_LEFT  = 65361
KEY_RIGHT = 65363

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pth")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.expanduser("~/yolo_valve_dataset"),
                        help="Dataset directory produced by sim_auto_labeler")
    parser.add_argument("--model",   default=_DEFAULT_MODEL,
                        help="Path to best.pth checkpoint")
    parser.add_argument("--conf",    type=float, default=0.3,
                        help="Keypoint confidence threshold for display")
    return parser.parse_args()


def read_label(label_path, img_w, img_h):
    """
    Parse a YOLO-pose label file and return (bbox_xywh, gt_keypoints).

    bbox_xywh   — (x, y, w, h) in pixel coords (top-left origin)
    gt_keypoints — list of (px, py, vis) in pixel coords
    """
    if not os.path.exists(label_path):
        return None, []

    with open(label_path) as f:
        line = f.readline().strip()

    fields = line.split()
    if len(fields) < 5:
        return None, []

    x_c, y_c, bw, bh = map(float, fields[1:5])
    x1 = (x_c - bw / 2) * img_w
    y1 = (y_c - bh / 2) * img_h
    bbox_xywh = (x1, y1, bw * img_w, bh * img_h)

    gt_kps = []
    kp_data = fields[5:]
    for i in range(0, len(kp_data) - 2, 3):
        kp_x = float(kp_data[i])     * img_w
        kp_y = float(kp_data[i + 1]) * img_h
        vis  = int(kp_data[i + 2])
        gt_kps.append((kp_x, kp_y, vis))

    return bbox_xywh, gt_kps


def draw_gt(img, label_path):
    """Draw ground-truth bounding box and keypoints."""
    h, w = img.shape[:2]
    bbox_xywh, gt_kps = read_label(label_path, w, h)

    if bbox_xywh is not None:
        x, y, bw, bh = bbox_xywh
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), BBOX_COLOR, 2)

    for kp_x, kp_y, vis in gt_kps:
        if vis == 2:
            px, py = int(kp_x), int(kp_y)
            img[py:py + 2, px:px + 2] = GT_KP_COLOR


def draw_predictions(img, kps, scores, conf_thr):
    """Draw predicted keypoints and skeleton connections."""
    for a, b in SKELETON:
        if scores[a, 0] > conf_thr and scores[b, 0] > conf_thr:
            cv2.line(img,
                     (int(kps[a, 0]), int(kps[a, 1])),
                     (int(kps[b, 0]), int(kps[b, 1])),
                     SKELETON_COLOR, 2)

    for i in range(len(kps)):
        if scores[i, 0] > conf_thr:
            cx, cy = int(kps[i, 0]), int(kps[i, 1])
            cv2.circle(img, (cx, cy), 6, PRED_KP_COLOR, -1)
            cv2.putText(img, f"{i+1} {scores[i,0]:.2f}",
                        (cx + 7, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, PRED_KP_COLOR, 1)


def main():
    args = parse_args()
    images_dir   = os.path.join(args.dataset, "images")
    labels_dir   = os.path.join(args.dataset, "labels")
    inference_dir = os.path.join(args.dataset, "inference")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_paths:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Loading model from {args.model} ...")
    model = ValvePose(args.model, device='cpu')
    print(f"Found {len(image_paths)} images.")
    print("  D/→ : next    A/← : prev    S : save    Q : quit")

    idx        = 0
    saved      = 0
    cache_idx  = -1
    cache_kps  = None
    cache_scores = None

    win = "Valve Inference  [ D/→ next | A/← prev | S save | Q quit ]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img_path   = image_paths[idx]
        stem       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            idx = min(idx + 1, len(image_paths) - 1)
            continue

        h, w = img.shape[:2]
        bbox_xywh, _ = read_label(label_path, w, h)

        # Run inference only when index changes (avoid re-running on every render).
        if idx != cache_idx and bbox_xywh is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cache_kps, cache_scores = model.predict(img_rgb, bbox_xywh)
            cache_idx = idx

        draw_gt(img, label_path)
        if cache_idx == idx and cache_kps is not None:
            draw_predictions(img, cache_kps, cache_scores, args.conf)

        status = (f"{stem}   {idx + 1}/{len(image_paths)}"
                  f"   conf>{args.conf:.2f}"
                  f"   GT=blue  PRED=cyan")
        cv2.putText(img, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(win, img)

        key = cv2.waitKey(0)
        ch  = key & 0xFF

        if ch == ord("q"):
            break
        elif ch == ord("d") or key == KEY_RIGHT:
            idx = min(idx + 1, len(image_paths) - 1)
        elif ch == ord("a") or key == KEY_LEFT:
            idx = max(idx - 1, 0)
        elif ch == ord("s"):
            os.makedirs(inference_dir, exist_ok=True)
            out_path = os.path.join(inference_dir, stem + "_inference.jpg")
            cv2.imwrite(out_path, img)
            saved += 1
            print(f"Saved {out_path}")

    cv2.destroyAllWindows()
    print(f"Done. {saved} image(s) saved.")


if __name__ == "__main__":
    main()
