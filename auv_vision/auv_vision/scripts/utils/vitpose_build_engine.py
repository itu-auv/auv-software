#!/usr/bin/env python3
"""Build a TensorRT engine from a vitpose_export.py ONNX, on the target GPU.

    python3 vitpose_build_engine.py <model.onnx> [--out model.engine] [--fp16] [--workspace-gb 2]

Uses the TensorRT python API (present on JetPack; no trtexec needed). The
sidecar <model>.json is copied next to the engine. Equivalent CLI:
    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
Engines are specific to the GPU + TensorRT version they were built on.
"""
import argparse
import os
import shutil
import sys


def build(onnx_path, out_path, fp16, workspace_gb):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
            sys.exit(1)
    config = builder.create_builder_config()
    ws = int(workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):  # TRT >= 8.4
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    else:
        config.max_workspace_size = ws
    if fp16:
        if not builder.platform_has_fast_fp16:
            print(
                "warning: platform reports no fast fp16; building anyway",
                file=sys.stderr,
            )
        config.set_flag(trt.BuilderFlag.FP16)
    print(
        f"building {out_path} (fp16={fp16}, workspace {workspace_gb} GB) — minutes on Orin"
    )
    if hasattr(builder, "build_serialized_network"):
        blob = builder.build_serialized_network(network, config)
    else:  # TRT 7/8.0
        engine = builder.build_engine(network, config)
        blob = engine.serialize() if engine is not None else None
    if blob is None:
        sys.exit("engine build failed")
    with open(out_path, "wb") as handle:
        handle.write(bytearray(blob))
    sidecar_src = os.path.splitext(onnx_path)[0] + ".json"
    sidecar_dst = os.path.splitext(out_path)[0] + ".json"
    if os.path.isfile(sidecar_src) and os.path.abspath(sidecar_src) != os.path.abspath(
        sidecar_dst
    ):
        shutil.copyfile(sidecar_src, sidecar_dst)
    print(
        f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB) + {sidecar_dst}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("onnx")
    ap.add_argument("--out", default=None, help="default: <onnx stem>.engine")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    args = ap.parse_args()
    build(
        args.onnx,
        args.out or os.path.splitext(args.onnx)[0] + ".engine",
        args.fp16,
        args.workspace_gb,
    )
