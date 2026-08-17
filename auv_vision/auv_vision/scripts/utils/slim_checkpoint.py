#!/usr/bin/env python3
"""Slim a valve-vision joint training checkpoint for deployment.

Training checkpoints carry optimizer/scheduler/scaler state (~3x the model
size). This keeps only what inference needs — the model weights plus the
self-describing contract keys that let vitpose_inference auto-configure:

    model, active_heads, model_size, img_size, train_config

Usage (run wherever torch + the source checkpoint are available, e.g. on
dream against ~/valve-vision):

    python3 slim_checkpoint.py <in.pth> <out.pth>
"""

import sys

import torch

KEEP_KEYS = ("model", "active_heads", "model_size", "img_size", "train_config")


def slim(src: str, dst: str) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    missing = [key for key in KEEP_KEYS if key not in payload]
    if missing:
        raise KeyError(f"{src} lacks contract keys {missing} — not a joint checkpoint?")
    out = {key: payload[key] for key in KEEP_KEYS}
    # train_config is small (plain scalars) and documents provenance; keep all.
    torch.save(out, dst)
    kept_mb = sum(v.numel() * v.element_size() for v in out["model"].values()) / 1e6
    print(
        f"{src} -> {dst}: heads={tuple(out['active_heads'])} "
        f"size={out['model_size']} img={tuple(out['img_size'])} "
        f"model tensors {kept_mb:.0f} MB"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    slim(sys.argv[1], sys.argv[2])
