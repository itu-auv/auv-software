#!/usr/bin/env python3
"""Regenerate the teknofest_tetra meshes from valve-vision's tetrahedron.gltf.

Companion to gate_gltf_to_collada.py, same idea: the sim object is the asset
the ViTPose tetra model was trained on, not a lookalike. Read
gate_tetra_overview.md §3 alongside this.

    ./tetra_gltf_to_collada.py ~/valve-vision/assets/tetra/tetrahedron.gltf

The asset is four single triangles (red = V0-V1-Apex, green = V1-V2-Apex,
blue = V2-V0-Apex, white base), so three things happen here that did not have
to happen for the gate:

1. **Winding.** A one-triangle "face" is invisible from behind under Gazebo's
   backface culling, so every triangle is rewound to face away from the solid's
   centroid. Get this wrong and the tetra renders as a hole.
2. **Colour space.** The tetra's baseColorFactors are LINEAR (its blue,
   0.0356/0.0742/0.1714, is exactly RAL 5000 #354D73 taken to linear), while
   Gazebo/OGRE1 does no gamma correction — feed them straight in and the solid
   comes out almost black. They are converted back to sRGB here. (gate.gltf, by
   contrast, stores display values already; the two assets disagree.)
3. **Letters.** The A/B/C glyphs are decals the Blender generator paints per
   render, so they do not exist in the asset. They are built here as thin white
   plates standing 2 mm off each face, upright toward the apex — the same
   convention the training data uses (`upright = "apex"`).

The letter -> colour assignment is the mission payload, so it is an explicit
argument rather than a default buried in code: --letters A=red,B=blue,C=green.
Note this fixes ONE arrangement; the real object's is whatever it is, and
tetra_unfold must derive the association from the image either way.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import trimesh
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon

# glTF material name -> output stem / SDF visual name
_COLOURS = {
    "RAL 3020 KIRMIZI": "red",
    "RAL 6018 YESIL": "green",
    "RAL 5000 MAVI": "blue",
    "white": "white",
}

# Face altitude of a 0.6 m regular tetrahedron (gate_tetra_overview.md §3.1).
_FACE_ALTITUDE = 0.519615
_GLYPH_HEIGHT = 0.35 * _FACE_ALTITUDE  # inside the generator's 0.25-0.50 range
_GLYPH_STANDOFF = 0.002  # m off the painted surface, as in generation


def linear_to_srgb(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _outward(triangle, centroid):
    """Rewind a triangle so its normal points away from the solid's centre."""
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    if np.dot(normal, triangle.mean(axis=0) - centroid) < 0:
        return triangle[[0, 2, 1]]
    return triangle


def _glyph_mesh(letter, triangle, font):
    """A white plate of `letter` lying on `triangle`, upright toward the apex.

    The apex is the vertex the two base vertices do not share with the ground:
    in this asset every coloured face is (base, base, apex), so it is the one
    with the highest z.
    """
    apex = triangle[np.argmax(triangle[:, 2])]
    base = triangle[[i for i in range(3) if not np.array_equal(triangle[i], apex)]]

    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    # In-face "up" = from the base edge midpoint toward the apex.
    up = apex - base.mean(axis=0)
    up -= np.dot(up, normal) * normal
    up /= np.linalg.norm(up)
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)

    path = TextPath((0, 0), letter, size=1.0, prop=font)
    polygons = [Polygon(p) for p in path.to_polygons() if len(p) >= 3]
    if not polygons:
        raise RuntimeError(f"font produced no outline for '{letter}'")
    # Outer ring first, the rest are holes (A has one, B has two).
    polygons.sort(key=lambda p: p.area, reverse=True)
    glyph = polygons[0]
    for hole in polygons[1:]:
        glyph = glyph.difference(hole)

    # FLAT, not extruded. Paint has no thickness, and more to the point an
    # extruded glyph shares its top vertices with its side walls, so the
    # averaged vertex normals COLLADA carries produce a gradient across every
    # letter instead of a flat colour. One planar polygon = one normal.
    vertices_2d, faces_2d = trimesh.creation.triangulate_polygon(glyph, engine="earcut")
    mesh = trimesh.Trimesh(
        vertices=np.c_[vertices_2d, np.zeros(len(vertices_2d))],
        faces=faces_2d,
        process=False,
    )
    # Normalize: centre on the glyph's own bounding box, scale to target height.
    bounds = mesh.bounds
    mesh.apply_translation(
        [
            -(bounds[0][0] + bounds[1][0]) / 2.0,
            -(bounds[0][1] + bounds[1][1]) / 2.0,
            0.0,
        ]
    )
    mesh.apply_scale(_GLYPH_HEIGHT / (bounds[1][1] - bounds[0][1]))

    # Local (x, y, z) -> (right, up, normal), origin at the face centroid,
    # lifted clear of the paint.
    frame = np.eye(4)
    frame[:3, 0] = right
    frame[:3, 1] = up
    frame[:3, 2] = normal
    frame[:3, 3] = triangle.mean(axis=0) + normal * _GLYPH_STANDOFF
    mesh.apply_transform(frame)
    return mesh


def convert(gltf_path, out_dir, letter_of_colour, font_name):
    with open(gltf_path) as handle:
        gltf = json.load(handle)

    materials = [m["name"] for m in gltf["materials"]]
    colours = [m["pbrMetallicRoughness"]["baseColorFactor"] for m in gltf["materials"]]
    primitive_material = [p.get("material", 0) for p in gltf["meshes"][0]["primitives"]]

    scene = trimesh.load(gltf_path)
    names = list(scene.geometry.keys())
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])

    faces = {}
    for name, material_index in zip(names, primitive_material):
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(rotation)
        colour = _COLOURS.get(materials[material_index])
        if colour is None:
            raise RuntimeError(f"unexpected material '{materials[material_index]}'")
        if len(mesh.faces) != 1:
            raise RuntimeError(f"{colour}: expected 1 triangle, got {len(mesh.faces)}")
        faces[colour] = mesh.vertices[mesh.faces[0]]

    missing = set(_COLOURS.values()) - set(faces)
    if missing:
        raise RuntimeError(f"asset is missing faces: {sorted(missing)}")

    centroid = np.vstack(list(faces.values())).mean(axis=0)
    font = FontProperties(family=font_name)

    written = []
    for colour, triangle in faces.items():
        triangle = _outward(triangle, centroid)
        mesh = trimesh.Trimesh(vertices=triangle, faces=[[0, 1, 2]], process=False)
        path = os.path.join(out_dir, f"tetra_{colour}.dae")
        mesh.export(path)
        _force_z_up(path)
        written.append(path)
        print(f"{colour:6s} face -> {os.path.basename(path)}")

    glyphs = []
    for letter, colour in sorted(letter_of_colour.items()):
        if colour not in faces:
            raise RuntimeError(f"letter {letter}: no '{colour}' face")
        glyphs.append(_glyph_mesh(letter, _outward(faces[colour], centroid), font))
        print(f"letter {letter} -> {colour} face")
    letters = trimesh.util.concatenate(glyphs)
    path = os.path.join(out_dir, "tetra_letters.dae")
    letters.export(path)
    _force_z_up(path)
    written.append(path)
    print(f"letters -> {os.path.basename(path)} ({len(letters.faces)} faces)")

    print("\nSDF <diffuse> values (linear -> sRGB):")
    for material, colour in zip(materials, colours):
        srgb = linear_to_srgb(colour[:3])
        hexcode = "#" + "".join(f"{int(round(v * 255)):02X}" for v in srgb)
        print(
            f"  {_COLOURS[material]:6s} {np.round(srgb, 3).tolist()}  {hexcode}"
            f"   (linear {np.round(colour[:3], 4).tolist()})"
        )
    return written


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


def _parse_letters(text):
    mapping = {}
    for pair in text.split(","):
        letter, _, colour = pair.partition("=")
        letter, colour = letter.strip(), colour.strip().lower()
        if not letter or colour not in ("red", "green", "blue"):
            raise argparse.ArgumentTypeError(f"bad letter spec '{pair}'")
        mapping[letter] = colour
    if len(set(mapping.values())) != len(mapping):
        raise argparse.ArgumentTypeError("two letters on one face")
    return mapping


def main():
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "models",
        "teknofest_tetra",
        "meshes",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gltf",
        nargs="?",
        default=os.path.expanduser("~/valve-vision/assets/tetra/tetrahedron.gltf"),
    )
    parser.add_argument("--out-dir", default=os.path.normpath(default_out))
    parser.add_argument(
        "--letters", type=_parse_letters, default="A=red,B=blue,C=green"
    )
    parser.add_argument("--font", default="DejaVu Sans")
    args = parser.parse_args()

    letters = args.letters
    if isinstance(letters, str):
        letters = _parse_letters(letters)
    if not os.path.isfile(args.gltf):
        parser.error(f"no such file: {args.gltf}")
    os.makedirs(args.out_dir, exist_ok=True)
    convert(args.gltf, args.out_dir, letters, args.font)
    return 0


if __name__ == "__main__":
    sys.exit(main())
