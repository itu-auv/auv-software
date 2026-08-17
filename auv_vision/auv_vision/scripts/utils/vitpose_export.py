#!/usr/bin/env python3
"""Export a valve-vision checkpoint (joint pose+seg or objectness) to ONNX +
sidecar JSON for the TensorRT backends in vitpose_inference.py.

    python3 vitpose_export.py <ckpt.pth> [--out DIR] [--opset 17] [--check]

Writes <DIR>/<ckpt stem>.onnx and <ckpt stem>.json (metadata predict() needs:
kind, img_size, K, C, mask_threshold, io names). Static batch 1, fp32 — the
FP16 decision is made at engine build (vitpose_build_engine.py). Then build
the engine ON THE TARGET GPU (engines are device-specific).

--check runs onnxruntime (CPU) against torch on a random input and reports the
max abs difference (expect < 1e-3 fp32).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vitpose_inference import ObjectnessDetector, VitposeModel  # noqa: E402


def export(ckpt, out_dir, opset, check):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, stem + ".onnx")
    json_path = os.path.join(out_dir, stem + ".json")

    if "active_heads" in payload:
        runner = VitposeModel(ckpt, device="cpu")
        kind = "joint"
        outputs = (["heatmaps"] if runner.num_kps else []) + (
            ["mask_logits"] if runner.num_masks else []
        )
        meta = dict(
            kind=kind,
            source=os.path.basename(ckpt),
            img_size=[runner.img_h, runner.img_w],
            active_heads=list(runner.active_heads),
            num_kps=runner.num_kps,
            num_masks=runner.num_masks,
            mask_threshold=runner.mask_threshold,
            input="input",
            outputs={name: name for name in outputs},
        )
        model = runner.model
        if len(outputs) == 1:
            # ONNX can't export a None output: wrap to the single live tensor
            # (pose-only -> heatmaps, seg-only -> mask_logits).
            index = 0 if runner.num_kps else 1

            class SingleHead(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(x)[index]

            model = SingleHead(model)
    else:
        runner = ObjectnessDetector(ckpt, device="cpu")
        kind = "objectness"
        outputs = ["logits"]
        meta = dict(
            kind=kind,
            source=os.path.basename(ckpt),
            img_size=[runner.img_h, runner.img_w],
            stride=runner.stride,
            input="input",
            outputs={"logits": "logits"},
        )
        model = runner.model

    model.eval()
    dummy = torch.zeros(1, 3, runner.img_h, runner.img_w, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=outputs,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    with open(json_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print(
        f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB) + {json_path}"
    )
    print(f"  kind={kind} input 1x3x{runner.img_h}x{runner.img_w} outputs={outputs}")

    if check:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        x = torch.rand(1, 3, runner.img_h, runner.img_w)
        with torch.no_grad():
            ref = model(x)
        ref = [ref] if torch.is_tensor(ref) else list(ref)
        got = sess.run(None, {"input": x.numpy()})
        for name, r, g in zip(outputs, ref, got):
            diff = float(np.abs(r.numpy() - g).max())
            print(
                f"  onnxruntime vs torch [{name}] shape {tuple(g.shape)} max|diff| {diff:.2e}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt")
    ap.add_argument(
        "--out", default=None, help="output dir (default: next to the ckpt)"
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true", help="onnxruntime parity check")
    args = ap.parse_args()
    export(
        args.ckpt,
        args.out or os.path.dirname(os.path.abspath(args.ckpt)),
        args.opset,
        args.check,
    )
