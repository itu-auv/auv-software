#!/usr/bin/env python3
"""Regenerate the teknofest_pinger gate meshes from valve-vision's gate.gltf.

The gate the ViTPose gate model was trained on is a single asset in the
valve-vision repo (`assets/gate/gate.gltf`) — the same file the keypoint
geometry in gate_tetra_overview.md §2 is read out of. This script is the only
thing standing between it and Gazebo, so the sim gate is dimensionally the
model's gate rather than a lookalike.

    ./gate_gltf_to_collada.py ~/valve-vision/assets/gate/gate.gltf

Two conversions matter and both are silent failures if you skip them:

1. **Y-up -> Z-up.** glTF buffers are Y-up by spec; the model frame everything
   else uses (valve-vision's object frame, the SDF, our gate_link) is Z-up with
   +X = gate width, +Y = plane normal. We rotate +90 deg about X here and then
   rewrite COLLADA's `<up_axis>` to Z_UP, because trimesh writes Y_UP
   unconditionally regardless of the coordinates it just exported — and Gazebo
   *honours* up_axis, so leaving it would rotate the gate onto its side.
2. **One file per material.** COLLADA export carries one material per mesh, and
   trimesh's writes it as black `defaultmaterial` anyway, so the seven glTF
   primitives are merged into two meshes by material (frame / pinger socket)
   and the actual RAL colours are set in model.sdf's <material> blocks, which
   override the mesh's.

Geometry is verified against the asset's own `extras.parts` bboxes before
writing: if the two ever disagree, the conversion is wrong, not the asset.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import trimesh

# material name prefix -> output mesh stem
_STEMS = {"RAL9005": "gate_frame", "RAL1023": "gate_pinger"}


def convert(gltf_path, out_dir):
    with open(gltf_path) as handle:
        gltf = json.load(handle)

    materials = [m["name"] for m in gltf["materials"]]
    primitive_material = [p.get("material", 0) for p in gltf["meshes"][0]["primitives"]]

    scene = trimesh.load(gltf_path)
    names = list(scene.geometry.keys())  # primitive order
    if len(names) != len(primitive_material):
        raise RuntimeError(
            f"{len(names)} geometries but {len(primitive_material)} primitives"
        )

    # glTF Y-up -> model frame Z-up: (x, y, z) -> (x, -z, y).
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])

    groups = {}
    for name, material_index in zip(names, primitive_material):
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(rotation)
        groups.setdefault(material_index, []).append(mesh)

    _verify_against_extras(gltf, names, primitive_material, scene, rotation)

    written = []
    for material_index, meshes in sorted(groups.items()):
        merged = trimesh.util.concatenate(meshes)
        material = materials[material_index]
        stem = next(
            (s for prefix, s in _STEMS.items() if material.startswith(prefix)), None
        )
        if stem is None:
            raise RuntimeError(f"no output name for material '{material}'")
        path = os.path.join(out_dir, f"{stem}.dae")
        merged.export(path)
        _force_z_up(path)
        low, high = np.round(merged.bounds, 4)
        written.append(path)
        print(
            f"{material:16s} -> {os.path.basename(path)}  "
            f"{len(merged.faces):4d} faces  "
            f"x[{low[0]:.3f},{high[0]:.3f}] "
            f"y[{low[1]:.3f},{high[1]:.3f}] "
            f"z[{low[2]:.3f},{high[2]:.3f}]"
        )
    return written


def _verify_against_extras(gltf, names, primitive_material, scene, rotation):
    """Cross-check every rotated part against the bbox the asset declares."""
    parts = (gltf.get("extras") or {}).get("parts")
    if not parts:
        print("WARNING: asset carries no extras.parts; skipping bbox verification")
        return
    if len(parts) != len(names):
        raise RuntimeError(f"{len(parts)} declared parts vs {len(names)} geometries")
    for part, name in zip(parts, names):
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(rotation)
        expected = np.array([part["bbox_min_m"], part["bbox_max_m"]])
        if not np.allclose(mesh.bounds, expected, atol=1e-4):
            raise RuntimeError(
                f"{part['short']}: rotated bounds {mesh.bounds.tolist()} != "
                f"declared {expected.tolist()} — the Y-up/Z-up convention moved"
            )
    print(f"verified {len(parts)} parts against the asset's declared bboxes")


def _force_z_up(path):
    """trimesh hardcodes Y_UP; the coordinates we exported are Z-up."""
    with open(path) as handle:
        text = handle.read()
    fixed, count = re.subn(
        r"<up_axis>\s*Y_UP\s*</up_axis>", "<up_axis>Z_UP</up_axis>", text
    )
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one Y_UP up_axis, found {count}")
    with open(path, "w") as handle:
        handle.write(fixed)


def main():
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "models",
        "teknofest_pinger",
        "meshes",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gltf",
        nargs="?",
        default=os.path.expanduser("~/valve-vision/assets/gate/gate.gltf"),
        help="valve-vision gate.gltf (default: ~/valve-vision/assets/gate/gate.gltf)",
    )
    parser.add_argument("--out-dir", default=os.path.normpath(default_out))
    args = parser.parse_args()

    if not os.path.isfile(args.gltf):
        parser.error(f"no such file: {args.gltf}")
    os.makedirs(args.out_dir, exist_ok=True)
    convert(args.gltf, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
