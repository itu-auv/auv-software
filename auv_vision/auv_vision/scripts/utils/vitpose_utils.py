#!/usr/bin/env python3
"""ViTPose pipeline utilities — ONE module, three sections.

House rule (Ufuk, 2026-08-10): **no new utils file per feature.** Shared
vitpose code goes in here as a new section, not a new module. Two files stay
outside on purpose:

  * ``vitpose_inference.py`` — imports torch; keeping it separate is what
    keeps the *process* node torch-free (it only ever needs numpy/cv2/scipy).
  * ``slim_checkpoint.py`` — standalone CLI tool, imported by nobody.

Navigation — jump to a banner (``grep -n "^# ===" vitpose_utils.py``):

  SECTION 1 — CONFIG
      Per-object YAML loading (``config/vitpose/<object>.yaml``), checkpoint
      path resolution, `model:`-section → VitposeModel kwargs.
      Used by: vitpose_detection_node, vitpose_process_node.

  SECTION 2 — PLANAR POSE FUSION
      ``FusedPlanarPoseEstimator``: 6-DOF pose of a known planar object from
      keypoints + one boundary mask (IPPE/iterative init, then joint LM over
      keypoint reprojection and dense mask-edge residuals). Plus the small
      projection helpers around it.
      Used by: vitpose_ops/gate_pose.

  SECTION 3 — TETRA LETTER↔FACE ASSOCIATION
      ``letter_face_membership`` (keypoint-in-mask / nearest-mask lookup),
      ``LetterFaceFilter`` (Bayesian filter over the 6 letter permutations),
      ``render_tetra_net`` (the unfolded-tetrahedron "triforce" image).
      Used by: vitpose_ops/tetra_unfold.

Everything here is ROS-free (numpy / cv2 / scipy / yaml / rospkg only) so it
can be exercised offline; the ROS glue lives in the nodes and the ops.
"""

import glob
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospkg
import yaml
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

_rospack = rospkg.RosPack()


# =============================================================================
# SECTION 1 — CONFIG
# =============================================================================
#
# One YAML per object under auv_vision/config/vitpose/<object>.yaml with three
# sections — `model:` (vitpose_inference), `detection:` (vitpose_detection_node),
# `process:` (vitpose_process_node). Schema: auv_vision/VITPOSE_PLAN.md §6.


def config_dir() -> str:
    return os.path.join(_rospack.get_path("auv_vision"), "config", "vitpose")


def resolve_config_path(name_or_path: str) -> str:
    """Accept an object name ("gate") or an absolute/relative YAML path."""
    if name_or_path.endswith((".yaml", ".yml")) or os.path.sep in name_or_path:
        path = os.path.abspath(os.path.expanduser(name_or_path))
    else:
        path = os.path.join(config_dir(), f"{name_or_path}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"vitpose config not found: {path}")
    return path


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Bare filenames resolve against auv_detection/models/ (house style)."""
    if os.path.isabs(checkpoint):
        return checkpoint
    return os.path.join(_rospack.get_path("auv_detection"), "models", checkpoint)


def load_object_config(name_or_path: str) -> dict:
    path = resolve_config_path(name_or_path)
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    for section in ("object", "model", "detection", "process"):
        if section not in config:
            raise ValueError(f"{path}: missing required section '{section}'")
    config["_path"] = path
    return config


def all_object_configs() -> dict:
    """{object name: config} for every YAML in the config dir."""
    configs = {}
    for path in sorted(glob.glob(os.path.join(config_dir(), "*.yaml"))):
        config = load_object_config(path)
        name = config["object"]
        if name in configs:
            raise ValueError(
                f"duplicate vitpose object '{name}' "
                f"({configs[name]['_path']} vs {path})"
            )
        configs[name] = config
    return configs


def model_kwargs(config: dict) -> dict:
    """Translate a config's `model:` section into VitposeModel kwargs."""
    model = config["model"]
    kwargs = dict(
        device=model.get("device", "cuda"),
        input_size=model.get("input_size"),
        decode=model.get("decode"),
        flip_tta=bool(model.get("flip_tta", False)),
        flip_pairs=model.get("flip_pairs"),
        mask_threshold=model.get("mask_threshold"),
    )
    return kwargs


# =============================================================================
# SECTION 2 — PLANAR POSE FUSION  (keypoint PnP + dense mask-edge refinement)
# =============================================================================
#
# Design + empirical verification: VITPOSE_PLAN.md §5 and the offline harness
# at dream:~/gate_fusion_verify (2026-08-09). Verified against pseudo-GT on the
# gate_joint_1000 val split:
#   - clean images: fused == kp-only == label-noise floor (~0.5 deg / 1 cm);
#   - occlusion workload (only the 4 near-collinear top kps survive, bottom of
#     the mask corrupted): kp-only 4.7 deg / 10 cm median, worst 30 deg —
#     fused 2 deg / 4 cm median, worst 6 deg, nearly flat in the occluded
#     fraction (the aperture side edges carry the pitch information).
#
# Formulation (RAPiD-style outer/inner loop over SE(3)):
#   outer: project the known planar boundary polygon under the current pose;
#          for each boundary sample, search along its image normal in the mask
#          probability map for the 0.5 crossing -> fixed 1D targets.
#   inner: scipy least_squares over (rvec, tvec), Huber loss:
#          - keypoint reprojection residuals, score-weighted; "soft" keypoints
#            (the gate post midpoints, which slide along their edge) contribute
#            only the residual component perpendicular to their projected
#            slide segment;
#          - mask residuals: normal-projected distance to the crossing targets,
#            sharpness-weighted.
#   The normal-search range shrinks per outer iteration (coarse-to-fine).
#
# Two robustness elements that verification proved necessary:
#   1. Two-sided LEVEL TEST per crossing: a genuine boundary edge separates
#      confident mask (P ~ 1) from confident background (P ~ 0); a
#      contamination boundary (occluder cut, uncertain region) has a mid-level
#      far side. Crossings failing the test carry zero weight. Without this,
#      an occluder edge near the true boundary captures the dense term and
#      fusion underperforms kp-only. (Leave-one-edge-out consensus was tried
#      and REJECTED: with few keypoints every edge is load-bearing, so the
#      leave-out pose drifts and good edges get dropped.)
#   2. IPPE degenerates outright (zero candidates) when the surviving
#      keypoints are near-collinear — exactly the occlusion case. estimate()
#      therefore falls back to ITERATIVE PnP seeded from a prior pose (last
#      accepted pose, or a canonical face-on guess).

FUSED_POSE_DEFAULTS = dict(
    score_gate=0.10,  # kps below this are dropped entirely
    huber_px=3.0,
    lam_mask=3.0,  # global mask-term weight multiplier
    n_per_edge=24,  # boundary samples per polygon edge
    search_schedule=(20.0, 12.0, 6.0, 3.0),  # px, per outer iteration
    search_step=0.5,  # px along the normal
    min_sharpness=0.02,  # prob/px; flatter crossings are dropped
    max_sharpness=0.5,  # weight clamp
    level_probe_px=4.0,  # distance beyond the crossing for the level test
    level_hi=0.80,  # one probe side must exceed this ...
    level_lo=0.18,  # ... and the other must be below this
    max_nfev=40,  # inner LM budget per outer iteration
)


def _contig(a, dtype=np.float64):
    return np.ascontiguousarray(a, dtype=dtype)


def solve_ippe(obj_pts, img_pts, K, D):
    """Planar PnP via solvePnPGeneric/IPPE. Returns [(rvec, tvec, err), ...]
    sorted by reprojection error — both plane-flip candidates. May return []
    (degenerate configurations) or raise cv2.error; callers must fall back."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        _contig(obj_pts).reshape(-1, 1, 3),
        _contig(img_pts).reshape(-1, 1, 2),
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE,
    )
    return [(rvecs[i].ravel(), tvecs[i].ravel(), float(errs[i])) for i in range(n)]


def project_points(pts3d, rvec, tvec, K, D):
    p, _ = cv2.projectPoints(
        _contig(pts3d).reshape(-1, 1, 3),
        np.asarray(rvec, np.float64),
        np.asarray(tvec, np.float64),
        K,
        D,
    )
    return p.reshape(-1, 2)


def plane_normal_cam(rvec, plane_normal_obj):
    """Object-frame plane normal expressed in the camera frame."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return R @ np.asarray(plane_normal_obj, dtype=np.float64)


def sample_bilinear(plane, xy):
    """Sample a (H, W) image at xy (N, 2) pixel coords; outside -> 0."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return map_coordinates(
        plane, [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=0.0
    )


class FusedPlanarPoseEstimator:
    """Pose of a known planar object from keypoints + one boundary mask.

    Args:
        model_points: (K, 3) object-frame keypoint positions, index = kp id.
        boundary_polygon: (M, 3) object-frame corners of the mask boundary
            polygon (closed implicitly), e.g. the gate aperture rectangle.
        plane_normal: (3,) object-frame normal of the object plane (for the
            flip guard), e.g. [0, 1, 0] for the gate.
        soft_slide_segments: {kp_id: ((3,), (3,))} object-frame segments for
            keypoints whose known error mode is sliding along an edge; they
            contribute only perpendicular-to-segment residuals.
        config: overrides for FUSED_POSE_DEFAULTS.
    """

    def __init__(
        self,
        model_points,
        boundary_polygon,
        plane_normal=(0.0, 1.0, 0.0),
        soft_slide_segments=None,
        **config,
    ):
        self.model_points = np.asarray(model_points, dtype=np.float64)
        self.boundary_polygon = np.asarray(boundary_polygon, dtype=np.float64)
        self.plane_normal = np.asarray(plane_normal, dtype=np.float64)
        self.soft = {
            int(k): (np.asarray(a, float), np.asarray(b, float))
            for k, (a, b) in (soft_slide_segments or {}).items()
        }
        self.cfg = dict(FUSED_POSE_DEFAULTS)
        unknown = set(config) - set(FUSED_POSE_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown FusedPlanarPoseEstimator config: {unknown}")
        self.cfg.update(config)
        self._bnd3d = self._sample_boundary_3d()

    # ------------------------------------------------------------ sampling

    def _sample_boundary_3d(self):
        n = self.cfg["n_per_edge"]
        pts = []
        m = len(self.boundary_polygon)
        for i in range(m):
            a = self.boundary_polygon[i]
            b = self.boundary_polygon[(i + 1) % m]
            t = (np.arange(n) + 0.5) / n
            pts.append(a[None] + t[:, None] * (b - a)[None])
        return np.concatenate(pts)

    def _edge_normals(self, pts2d):
        """Unit image normals per sample from each edge's local tangent."""
        n = self.cfg["n_per_edge"]
        normals = np.zeros_like(pts2d)
        for e in range(len(self.boundary_polygon)):
            seg = pts2d[e * n : (e + 1) * n]
            tang = np.gradient(seg, axis=0)
            tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
            normals[e * n : (e + 1) * n] = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        return normals

    def _find_crossings(self, mask, pts2d, normals, search_px):
        """Nearest 0.5 crossing along each sample's normal, level-tested.
        Returns (offsets, weights); weight 0 = no usable crossing."""
        cfg = self.cfg
        step = cfg["search_step"]
        us = np.arange(-search_px, search_px + 1e-9, step)
        offs = np.zeros(len(pts2d))
        wts = np.zeros(len(pts2d))
        for i in range(len(pts2d)):
            line = pts2d[i][None] + us[:, None] * normals[i][None]
            vals = sample_bilinear(mask, line) - 0.5
            sign = np.signbit(vals)
            flips = np.nonzero(sign[:-1] != sign[1:])[0]
            if len(flips) == 0:
                continue
            j = flips[np.argmin(np.abs(us[flips] + step * 0.5))]
            denom = vals[j + 1] - vals[j]
            frac = -vals[j] / denom if abs(denom) > 1e-12 else 0.5
            offs[i] = us[j] + frac * step
            sharp = abs(denom) / step
            if sharp < cfg["min_sharpness"]:
                continue
            probe = cfg["level_probe_px"]
            sides = sample_bilinear(
                mask,
                np.stack(
                    [
                        pts2d[i] + (offs[i] - probe) * normals[i],
                        pts2d[i] + (offs[i] + probe) * normals[i],
                    ]
                ),
            )
            if max(sides) < cfg["level_hi"] or min(sides) > cfg["level_lo"]:
                continue  # level test: not a mask/background boundary
            wts[i] = min(sharp, cfg["max_sharpness"])
        return offs, wts

    # ------------------------------------------------------------ solving

    def init_pose(self, kps, scores, K, D, prior=None):
        """Initial pose from gated keypoints.

        IPPE first (flip guard against the prior normal when available);
        ITERATIVE from the prior when IPPE degenerates. prior = (rvec, tvec)
        or None. Returns (rvec, tvec, used_ids) or None.
        """
        ids = np.nonzero(scores >= self.cfg["score_gate"])[0]
        if len(ids) < 4:
            return None
        try:
            cands = solve_ippe(self.model_points[ids], kps[ids], K, D)
        except cv2.error:
            cands = []
        cands = [c for c in cands if c[1][2] > 0]  # in front of the camera
        if cands:
            if prior is not None and len(cands) > 1:
                # Flip guard: among candidates whose reprojection error is
                # comparable (within 1.5x of the best), prefer the one whose
                # plane normal agrees with the prior.
                best_err = cands[0][2]
                close = [c for c in cands if c[2] <= 1.5 * best_err + 1e-9]
                n_prior = plane_normal_cam(prior[0], self.plane_normal)
                close.sort(
                    key=lambda c: -float(
                        plane_normal_cam(c[0], self.plane_normal) @ n_prior
                    )
                )
                return close[0][0], close[0][1], ids
            return cands[0][0], cands[0][1], ids
        if prior is None:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            _contig(self.model_points[ids]).reshape(-1, 1, 3),
            _contig(kps[ids]).reshape(-1, 1, 2),
            K,
            D,
            np.asarray(prior[0], np.float64).reshape(3, 1).copy(),
            np.asarray(prior[1], np.float64).reshape(3, 1).copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        return np.asarray(rvec).ravel(), np.asarray(tvec).ravel(), ids

    def refine(self, kps, scores, mask_prob, K, D, rvec0, tvec0):
        """Joint LM refinement from an initial pose. mask_prob None -> the
        keypoint-only path (same gating/robustifier, no dense term).

        Returns dict(rvec, tvec, kp_ids, n_crossings, n_samples).
        """
        cfg = self.cfg
        kp_ids = np.nonzero(scores >= cfg["score_gate"])[0]
        kp_w = np.clip(scores[kp_ids], 0.0, 1.0)
        use_mask = mask_prob is not None and cfg["lam_mask"] > 0
        pose = np.concatenate([np.asarray(rvec0, float), np.asarray(tvec0, float)])
        n_cross = 0

        for search_px in cfg["search_schedule"]:
            targets = normals = wts = None
            if use_mask:
                pts2d = project_points(self._bnd3d, pose[:3], pose[3:], K, D)
                normals = self._edge_normals(pts2d)
                offs, wts = self._find_crossings(mask_prob, pts2d, normals, search_px)
                targets = pts2d + offs[:, None] * normals
                wts = wts * cfg["lam_mask"]
                n_cross = int((wts > 0).sum())

            def residuals(p):
                r = []
                proj_kp = project_points(self.model_points[kp_ids], p[:3], p[3:], K, D)
                for j, kid in enumerate(kp_ids):
                    d = proj_kp[j] - kps[kid]
                    if kid in self.soft:
                        a, b = self.soft[kid]
                        seg = project_points(np.stack([a, b]), p[:3], p[3:], K, D)
                        t = seg[1] - seg[0]
                        t /= max(np.linalg.norm(t), 1e-9)
                        r.append(np.atleast_1d(kp_w[j] * (np.array([-t[1], t[0]]) @ d)))
                    else:
                        r.append(kp_w[j] * d)
                if use_mask:
                    pb = project_points(self._bnd3d, p[:3], p[3:], K, D)
                    r.append(wts * np.einsum("ij,ij->i", pb - targets, normals))
                return np.concatenate(r)

            sol = least_squares(
                residuals,
                pose,
                loss="huber",
                f_scale=cfg["huber_px"],
                max_nfev=cfg["max_nfev"],
                method="trf",
            )
            pose = sol.x
            if not use_mask:
                break  # kp-only needs no outer loop

        return dict(
            rvec=pose[:3],
            tvec=pose[3:],
            kp_ids=kp_ids,
            n_crossings=n_cross,
            n_samples=len(self._bnd3d),
        )

    def estimate(self, kps, scores, mask_prob, K, D, prior=None):
        """init_pose + refine in one call. Returns refine()'s dict or None."""
        init = self.init_pose(kps, scores, K, D, prior=prior)
        if init is None:
            return None
        rvec0, tvec0, _ = init
        return self.refine(kps, scores, mask_prob, K, D, rvec0, tvec0)

    # ------------------------------------------------------------ validation

    def boundary_iou(self, rvec, tvec, mask_binary, K, D):
        """IoU of the projected (filled) boundary polygon vs a binary mask.

        The cheap "very reliable pose" validator: catches flip- and
        outlier-driven solves that reproject fine on few keypoints but put
        the boundary in the wrong place. Returns 0.0 when nothing overlaps.
        """
        h, w = mask_binary.shape
        poly = project_points(self.boundary_polygon, rvec, tvec, K, D)
        render = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(render, [np.round(poly).astype(np.int32)], 1)
        inter = np.logical_and(render, mask_binary).sum()
        union = np.logical_or(render, mask_binary).sum()
        return float(inter) / float(union) if union else 0.0


# =============================================================================
# SECTION 3 — TETRA LETTER↔FACE ASSOCIATION
# =============================================================================
#
# The TEKNOFEST "yildizlar" tetra (gate_tetra_overview.md §3): a 60 cm regular
# tetrahedron on its white base, three coloured side faces (red/green/blue),
# one of the letters A/B/C painted on each. The mission payload is the
# association letter <-> face colour. The letter keypoints are per-image glyph
# centres, NOT fixed 3D points, so there is no pose to solve here — only an
# association to decide, and to keep deciding as the view changes.
#
# Two-step design:
#
#   1. PER FRAME (letter_face_membership): keypoint-in-mask lookup on the soft
#      mask probabilities, with a nearest-mask fallback for letters that land
#      just outside every face. Letters below the score gate contribute
#      nothing at all — a letter on a fully averted face gets no supervision
#      during training, so its prediction is unconstrained garbage and the
#      score is the only defence (overview §3.4).
#
#   2. OVER TIME (LetterFaceFilter): there are exactly 6 possible worlds — the
#      bijections {A,B,C} -> {red,green,blue} — so instead of assigning per
#      frame and voting, keep a log-posterior over those 6 hypotheses and
#      update it with each frame's log-likelihood, with exponential forgetting.
#      Consequences that fall out for free:
#        - one-letter-per-face is structural, not a tie-break rule;
#        - two confidently-seen letters pin the third (the "inferred" case)
#          quantitatively, via the marginals;
#        - p_best is a real confidence, usable as a lock criterion;
#        - clamping the log-posterior keeps the filter reversible: a wrong
#          early lock decays away in a few hundred ms instead of sticking.
#
# The chirality (which way the colours run around the solid) is NOT estimated
# and not cross-checked — by decision it is a config constant, used only to
# lay the colours out in the rendered net.

TETRA_DEFAULTS = dict(
    min_score=0.35,  # letters below this contribute no evidence at all
    inside_min=0.25,  # mask prob at the kp that counts as "in this face"
    max_mask_distance_px=40.0,  # nearest-mask fallback reach
    proximity_weight=0.6,  # fallback evidence is worth less than a hit
    eps=0.04,  # membership floor -> bounds one frame's influence
    tau=3.0,  # s, evidence forgetting time constant
    gain=0.25,  # per-frame log-likelihood gain
    logp_clip=8.0,  # log-posterior clamp (reversibility)
    # Lock criteria. 0.99/10 over 0.9/5 by measurement (dream:~/tetra_unfold_check):
    # with 35% coherently-wrong frames it cuts first-lock errors 6.8% -> 1.2%,
    # costing one extra frame of latency on clean data (4 frames, 0.4 s @10 Hz).
    lock_threshold=0.99,  # p_best needed to call the association LOCKED
    min_evidence=10.0,  # ... and this much accumulated letter evidence
    min_direct_evidence=1.0,  # below this, a letter's colour is "inferred"
)

# perm[letter index] = colour index. Fixed order so posterior indices are stable.
LETTER_PERMUTATIONS: List[Tuple[int, ...]] = list(itertools.permutations(range(3)))

# Drawing colours (BGR) for the three tetra mask classes, from the viz palette
# in gate_tetra_overview.md §3.3. Overridable from the op params.
TETRA_FACE_COLORS = {
    "red": (60, 60, 235),
    "green": (90, 210, 60),
    "blue": (255, 120, 70),
}

# Which slot of the net each face occupies, going CLOCKWISE in the image.
# The net of a regular tetrahedron is one big triangle split into four: three
# corner triangles (the coloured side faces) around a central inverted one
# (the white base). Slot 0 = top, then clockwise.
_NET_SLOT_ORDER = {"cw": (0, 1, 2), "ccw": (0, 2, 1)}


@dataclass
class LetterFaceEstimate:
    """Read-out of the filter state (see LetterFaceFilter.result())."""

    posterior: np.ndarray  # (6,) probability per permutation
    best_perm: Tuple[int, ...]  # perm[letter] = colour, the MAP hypothesis
    p_best: float  # posterior of best_perm
    marginals: np.ndarray  # (3, 3) P(letter l on colour c)
    evidence: np.ndarray  # (3,) decayed direct-observation mass per letter
    inferred: np.ndarray  # (3,) bool: assignment rests on the permutation
    locked: bool  # confident enough to act on
    age: float  # s since the last frame that carried evidence

    def colour_of(self, letter_index: int) -> int:
        return self.best_perm[letter_index]

    def confidence_of(self, letter_index: int) -> float:
        return float(self.marginals[letter_index, self.best_perm[letter_index]])


def letter_face_membership(
    pixels: np.ndarray,
    scores: np.ndarray,
    mask_probs: np.ndarray,
    mask_threshold: float = 0.5,
    **config,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Per-frame evidence that letter l sits on face colour c.

    Args:
        pixels: (L, 2) letter keypoints in source-image px.
        scores: (L,) raw heatmap peak values.
        mask_probs: (C, H, W) face probabilities in [0, 1], source resolution.
        mask_threshold: calibrated binarization threshold (fallback only).
        config: overrides for the TETRA_DEFAULTS keys used here
            (min_score, inside_min, max_mask_distance_px, proximity_weight, eps).

    Returns:
        m: (L, C) membership, each active row a distribution over colours.
        w: (L,) evidence weight per letter (0 = ignore this letter).
        sources: per-letter "in" | "near" | "none" | "low" (debug/telemetry).

    The nearest-mask distance transforms are computed lazily — only when some
    active letter falls outside every mask — because they cost ~1 ms/class at
    640x480 and most frames never need them.
    """
    cfg = dict(TETRA_DEFAULTS)
    cfg.update(config)
    n_letters = len(pixels)
    n_colors = len(mask_probs)
    m = np.full((n_letters, n_colors), 1.0 / n_colors)
    w = np.zeros(n_letters)
    sources = ["low"] * n_letters
    distance_maps = None

    for i in range(n_letters):
        if scores[i] < cfg["min_score"]:
            continue
        raw = np.array(
            [float(sample_bilinear(plane, pixels[i])[0]) for plane in mask_probs]
        )
        if raw.max() >= cfg["inside_min"]:
            source = "in"
        else:
            # Fallback: how far is the keypoint from each face?
            if distance_maps is None:
                distance_maps = [
                    cv2.distanceTransform(
                        (plane < mask_threshold).astype(np.uint8), cv2.DIST_L2, 3
                    )
                    for plane in mask_probs
                ]
            x = int(np.clip(round(pixels[i][0]), 0, mask_probs.shape[2] - 1))
            y = int(np.clip(round(pixels[i][1]), 0, mask_probs.shape[1] - 1))
            distances = np.array([float(dm[y, x]) for dm in distance_maps])
            raw = cfg["proximity_weight"] * np.clip(
                1.0 - distances / cfg["max_mask_distance_px"], 0.0, 1.0
            )
            source = "near" if raw.max() > 0 else "none"
        total = raw.sum()
        if total <= 0:
            sources[i] = source
            continue
        row = np.clip(raw / total, cfg["eps"], None)
        m[i] = row / row.sum()
        w[i] = float(scores[i])
        sources[i] = source
    return m, w, sources


class LetterFaceFilter:
    """Bayesian filter over the 6 letter->colour permutations (see banner)."""

    def __init__(self, **config):
        self.cfg = dict(TETRA_DEFAULTS)
        unknown = set(config) - set(TETRA_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown LetterFaceFilter config: {unknown}")
        self.cfg.update(config)
        self._log_lut = None
        self.reset()

    def reset(self):
        self.logp = np.zeros(len(LETTER_PERMUTATIONS))
        self.evidence = np.zeros(3)
        self._last_time: Optional[float] = None
        self._last_evidence_time: Optional[float] = None

    def update(self, m: np.ndarray, w: np.ndarray, now: float) -> None:
        """Fold one frame's membership/weights in at time `now` (seconds)."""
        cfg = self.cfg
        decay = 1.0
        if self._last_time is not None:
            dt = max(now - self._last_time, 0.0)
            decay = float(np.exp(-dt / cfg["tau"]))
        self._last_time = now

        self.logp *= decay
        self.evidence *= decay

        if w.max() > 0:
            log_m = np.log(np.clip(m, 1e-9, None))
            loglik = np.array(
                [
                    sum(w[l] * log_m[l, perm[l]] for l in range(3))
                    for perm in LETTER_PERMUTATIONS
                ]
            )
            self.logp += cfg["gain"] * loglik
            self.evidence += w
            self._last_evidence_time = now

        self.logp -= self.logp.max()
        self.logp = np.clip(self.logp, -cfg["logp_clip"], 0.0)

    def result(self, now: Optional[float] = None) -> LetterFaceEstimate:
        cfg = self.cfg
        posterior = np.exp(self.logp)
        posterior /= posterior.sum()
        best = int(np.argmax(posterior))
        best_perm = LETTER_PERMUTATIONS[best]

        marginals = np.zeros((3, 3))
        for p, perm in enumerate(LETTER_PERMUTATIONS):
            for letter, colour in enumerate(perm):
                marginals[letter, colour] += posterior[p]

        total_evidence = float(self.evidence.sum())
        locked = bool(
            posterior[best] >= cfg["lock_threshold"]
            and total_evidence >= cfg["min_evidence"]
        )
        # "Inferred" = this letter's colour rests on the permutation constraint
        # rather than on its own observations. Only meaningful once the frame
        # as a whole carries enough evidence — otherwise every letter is
        # trivially "under-observed" in the first frames.
        inferred = (self.evidence < cfg["min_direct_evidence"]) & (
            total_evidence >= cfg["min_evidence"]
        )
        if now is None or self._last_evidence_time is None:
            age = float("inf") if self._last_evidence_time is None else 0.0
        else:
            age = max(now - self._last_evidence_time, 0.0)
        return LetterFaceEstimate(
            posterior=posterior,
            best_perm=best_perm,
            p_best=float(posterior[best]),
            marginals=marginals,
            evidence=self.evidence.copy(),
            inferred=inferred,
            locked=locked,
            age=age,
        )


def format_association(
    estimate: LetterFaceEstimate,
    letter_names: Sequence[str],
    mask_classes: Sequence[str],
) -> str:
    """The published one-liner, e.g.
    "A:red B:blue C:green p=0.97 state=LOCKED inferred=C"."""
    pairs = " ".join(
        f"{letter_names[l]}:{mask_classes[estimate.best_perm[l]]}"
        for l in range(len(letter_names))
    )
    state = "LOCKED" if estimate.locked else "UNCERTAIN"
    if estimate.evidence.sum() <= 0:
        state = "NO_DATA"
    text = f"{pairs} p={estimate.p_best:.2f} state={state}"
    inferred = [letter_names[l] for l in range(3) if estimate.inferred[l]]
    if inferred:
        text += f" inferred={','.join(inferred)}"
    return text


# ------------------------------------------------------------------ the net


def _net_triangles(size: int, top: int, bottom: int):
    """Corner-slot triangles + the central base triangle for the net.

    Slot 0 = top, 1 = bottom-right, 2 = bottom-left (i.e. clockwise), which is
    what _NET_SLOT_ORDER indexes into.
    """
    height = size - top - bottom
    side = min(size * 0.92, height / (np.sqrt(3) / 2.0))
    cx = size / 2.0
    tri_h = side * np.sqrt(3) / 2.0
    y0 = top + (height - tri_h) / 2.0
    apex = np.array([cx, y0])
    left = np.array([cx - side / 2.0, y0 + tri_h])
    right = np.array([cx + side / 2.0, y0 + tri_h])
    m_al = (apex + left) / 2.0
    m_ar = (apex + right) / 2.0
    m_lr = (left + right) / 2.0
    # Vertex 0 of each corner triangle is its OUTER vertex (the one not shared
    # with the base triangle) — labels are placed along centroid -> vertex 0.
    corners = [
        np.stack([apex, m_ar, m_al]),  # slot 0: top
        np.stack([right, m_lr, m_ar]),  # slot 1: bottom-right
        np.stack([left, m_al, m_lr]),  # slot 2: bottom-left
    ]
    base = np.stack([m_al, m_ar, m_lr])
    return corners, base


def _fill(image, tri, color):
    cv2.fillConvexPoly(image, np.round(tri).astype(np.int32), color, cv2.LINE_AA)


def _outline(image, tri, color=(30, 30, 30), thickness=2):
    cv2.polylines(
        image, [np.round(tri).astype(np.int32)], True, color, thickness, cv2.LINE_AA
    )


def _centered_text(image, text, center, scale, color, thickness, font=None):
    font = font or cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(image, text, org, font, scale, (20, 20, 20), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, scale, color, thickness, cv2.LINE_AA)


def render_tetra_net(
    estimate: LetterFaceEstimate,
    letter_names: Sequence[str],
    mask_classes: Sequence[str],
    chirality: str = "cw",
    face_colors: Optional[dict] = None,
    size: int = 480,
    per_letter_scores: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """The unfolded tetrahedron ("triforce"): the current best guess, drawn.

    Three corner triangles = the coloured side faces, laid out in the
    configured chirality order (slot 0 top, then clockwise); the central
    inverted triangle = the white base. Each face carries its letter, styled
    by confidence, plus a footer table of the filter's marginals.
    """
    colors = dict(TETRA_FACE_COLORS)
    colors.update(face_colors or {})
    top, bottom = 34, 104
    image = np.full((size, size, 3), 24, dtype=np.uint8)
    corners, base = _net_triangles(size, top, bottom)
    slots = _NET_SLOT_ORDER.get(chirality)
    if slots is None:
        raise ValueError(f"chirality must be 'cw' or 'ccw', got {chirality!r}")

    # White base face in the middle.
    _fill(image, base, (225, 225, 225))
    _outline(image, base)
    _centered_text(image, "base", base.mean(axis=0), 0.5, (90, 90, 90), 1)

    no_data = estimate.evidence.sum() <= 0
    for colour_index, class_name in enumerate(mask_classes):
        tri = corners[slots[colour_index]]
        _fill(image, tri, colors.get(class_name, (128, 128, 128)))
        _outline(image, tri)
        centroid = tri.mean(axis=0)
        outward = tri[0] - centroid  # toward the face's outer vertex
        glyph_at = centroid + 0.16 * outward
        tag_at = centroid - 0.30 * outward
        label_at = centroid + 0.62 * outward

        letters = [l for l in range(3) if estimate.best_perm[l] == colour_index]
        letter = letters[0] if letters else None
        if no_data or letter is None:
            _centered_text(image, "?", glyph_at, 1.6, (235, 235, 235), 3)
        else:
            confidence = estimate.confidence_of(letter)
            glyph = letter_names[letter]
            bright = (255, 255, 255) if estimate.locked else (185, 185, 185)
            _centered_text(image, glyph, glyph_at, 2.2, bright, 5)
            tag = f"{confidence:.2f}"
            if estimate.inferred[letter]:
                tag += " inferred"
            elif not estimate.locked:
                tag += " ?"
            _centered_text(image, tag, tag_at, 0.45, bright, 1)
        _centered_text(image, class_name, label_at, 0.45, (245, 245, 245), 1)

    # Header. Staleness rides in the same line (and recolours it) so it can
    # never collide with the state text.
    state = "LOCKED" if estimate.locked else ("NO DATA" if no_data else "UNCERTAIN")
    stale = np.isfinite(estimate.age) and estimate.age > 1.0
    header = f"TETRA  {state}  p={estimate.p_best:.2f}  chir={chirality}"
    if stale:
        header += f"  stale {estimate.age:.0f}s"
    header_color = (
        (90, 170, 255)
        if stale
        else ((120, 235, 120) if estimate.locked else (120, 200, 255))
    )
    cv2.putText(
        image,
        header,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        header_color,
        2,
        cv2.LINE_AA,
    )

    # Footer: marginals table (letters x colours) + per-letter evidence.
    y0 = size - bottom + 34
    col_x = [int(size * (0.36 + 0.20 * i)) for i in range(len(mask_classes))]
    for i, class_name in enumerate(mask_classes):
        cv2.putText(
            image,
            class_name,
            (col_x[i] - 18, y0 - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colors.get(class_name, (200, 200, 200)),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "ev  score",
        (18, y0 - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    for letter in range(3):
        y = y0 + 20 * letter
        score_text = (
            f"{per_letter_scores[letter]:.2f}" if per_letter_scores is not None else "-"
        )
        cv2.putText(
            image,
            f"{letter_names[letter]} {estimate.evidence[letter]:5.1f} {score_text}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        for colour in range(len(mask_classes)):
            value = estimate.marginals[letter, colour]
            shade = int(60 + 175 * value)
            cv2.putText(
                image,
                f"{value:.2f}",
                (col_x[colour] - 18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (shade, shade, shade),
                2 if value > 0.5 else 1,
                cv2.LINE_AA,
            )
    return image
