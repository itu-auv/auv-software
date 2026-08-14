#!/usr/bin/env python3
"""ViTPose pipeline utilities — ONE module, four sections.

House rule (Ufuk, 2026-08-10): **no new utils file per feature** — shared
vitpose code becomes a section here. Outside on purpose: vitpose_inference.py
(imports torch, which keeps the process node torch-free) and
slim_checkpoint.py (standalone CLI). Everything here is ROS-free so it can be
exercised offline.

Sections (``grep -n "^# ===" vitpose_utils.py``):

  1 CONFIG              per-object YAML loading, checkpoint paths, shared
                        validation, RateGate
  2 PLANAR POSE FUSION  FusedPlanarPoseEstimator — used by gate_pose
  3 TETRA ASSOCIATION   letter_face_membership, LetterFaceFilter,
                        render_tetra_net — used by tetra_unfold
  4 CROP TRACKING       CropTracker — used by the `model` bbox provider
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
# One YAML per object under config/vitpose/<object>.yaml: `object` +
# `detection` required, `model` + `process` optional (no `model:` =
# detect-only).


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


def load_object_config(name_or_path: str, ns: str = "taluy") -> dict:
    """Load one object config. all_object_configs() loads every YAML in the
    dir, so raising here takes down the whole process node — keep the
    required set minimal (`object`, `detection`, `camera`). The `camera`
    token is expanded into every camera-derived key (apply_camera)."""
    path = resolve_config_path(name_or_path)
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    for section in ("object", "detection", "camera"):
        if section not in config:
            raise ValueError(f"{path}: missing required key '{section}'")
    config["_path"] = path
    return apply_camera(config, config["camera"], ns)


def is_detect_only(config: dict) -> bool:
    """True for a config with no `model:` section: boxes out, no joint model."""
    return not config.get("model")


def validate_detect_only(object_name: str, provider_cfg: dict) -> None:
    """Reject detect-only misconfigurations that would otherwise fail
    silently. Shared by the real node and the sim twin. `provider_cfg` is the
    raw `detection.bbox_provider` mapping (still containing `type`)."""
    provider_cfg = provider_cfg or {}
    provider_type = provider_cfg.get("type", "full_frame")
    if provider_type != "model":
        raise ValueError(
            f"{object_name}: detect-only (no `model:` section) needs "
            f"bbox_provider type 'model', got '{provider_type}' — an "
            "objectness detector is the only thing left to run."
        )
    if provider_cfg.get("tracker"):
        raise ValueError(
            f"{object_name}: detect-only cannot use a `tracker:` block. "
            "CropTracker is propagated by the JOINT model's output via "
            "provider.feedback(); with no joint model the crop would "
            "silently freeze at the first seed forever. Use "
            "`detection.rate` to control the cost instead."
        )
    if not provider_cfg.get("publish_topic"):
        raise ValueError(
            f"{object_name}: detect-only needs "
            "`bbox_provider.publish_topic` — it is the only output."
        )


def runtime_detect_rate(detection_cfg: dict):
    """Frame-rate cap for a pose config running in mode 'detect': with no
    joint model there is no tracker to amortize the objectness pass, so every
    processed frame costs a full detector run (~27 ms). `detection.detect_rate`
    wins; default is the provider's `search_rate` (5 Hz). Non-model providers
    are cheap — `detection.rate` stands. Shared real + sim so the sim twin
    heartbeats at the rate the real node would."""
    provider = detection_cfg.get("bbox_provider") or {}
    if provider.get("type", "full_frame") != "model":
        return detection_cfg.get("rate")
    rate = detection_cfg.get("detect_rate")
    if rate is None:
        rate = provider.get("search_rate", 5.0)
    return rate


class RateGate:
    """`detection.rate` (Hz): due(now_sec) is True at most once per period;
    falsy rate = no gating. Shared by the real node and the sim twin."""

    def __init__(self, rate):
        self._min_period = 1.0 / float(rate) if rate else 0.0
        self._last: Optional[float] = None

    def due(self, now: float) -> bool:
        if self._min_period <= 0.0:
            return True
        if self._last is None or now - self._last >= self._min_period:
            self._last = now
            return True
        return False


def apply_camera(config: dict, camera: str, ns: str) -> dict:
    """Point a config at a camera by the repo's standard camera layout: image
    `/{ns}/cameras/cam_<camera>/image_raw`, calibration `cameras/cam_<camera>`,
    frame `{ns}/base_link/<camera>_camera_optical_link`, VitposeResult on
    `/vitpose_result_<camera>` — so `front`/`bottom`/`torpedo` work, and so
    does any camera published in that layout (e.g. a webcam masquerading as
    cam_webcam). Called by the loader for the YAML's own `camera:` and again
    for the `~camera` bench-test override. Mutates and returns `config`; the
    bbox publish_topic is left alone (it names the object).
    """
    image_topic = f"/{ns}/cameras/cam_{camera}/image_raw"
    result_topic = f"/vitpose_result_{camera}"
    config["camera"] = camera
    config["detection"]["image_topic"] = image_topic
    config["detection"]["result_topic"] = result_topic
    process = config.get("process")
    if process:
        process["image_topic"] = image_topic
        process["result_topic"] = result_topic
        process["camera"] = {
            "frame": f"{ns}/base_link/{camera}_camera_optical_link",
            "calibration_ns": f"cameras/cam_{camera}",
        }
    return config


def all_object_configs(ns: str = "taluy") -> dict:
    """{object name: config} for every YAML in the config dir."""
    configs = {}
    for path in sorted(glob.glob(os.path.join(config_dir(), "*.yaml"))):
        config = load_object_config(path, ns)
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
# Verified vs pseudo-GT on gate_joint_1000 val (dream:~/gate_fusion_verify):
# clean = label-noise floor; under occlusion (only the 4 near-collinear top
# kps survive) kp-only 4.7 deg / 10 cm median, worst 30 deg -> fused
# 2 deg / 4 cm, worst 6 — the aperture side edges carry the missing pitch.
#
# RAPiD-style outer/inner loop over SE(3):
#   0:     IPPE-RANSAC zeroes confidently-wrong keypoints. Huber only dampens
#          a bad point — a large outlier still drags the init into a wrong
#          basin the local LM cannot escape.
#   outer: project the boundary polygon; search along each sample's image
#          normal for the 0.5 mask crossing (range shrinks per iteration).
#   inner: Huber LM over (rvec, tvec): score-weighted kp reprojection (soft
#          keypoints contribute only perpendicular-to-slide) +
#          sharpness-weighted crossing residuals.
#
# Two elements verification proved necessary:
#   1. Two-sided LEVEL TEST per crossing (mask ~1 one side, ~0 the other):
#      without it an occluder edge near the true boundary captures the dense
#      term and fusion LOSES to kp-only. (Leave-one-edge-out consensus tried
#      and rejected — with few keypoints every edge is load-bearing.)
#   2. IPPE returns zero candidates when the surviving kps are near-collinear
#      — exactly the occlusion case — hence the ITERATIVE fallback seeded
#      from a prior pose.

FUSED_POSE_DEFAULTS = dict(
    score_gate=0.10,  # kps below this are dropped entirely
    # Needs >= 5 gated kps (4 leave nothing to vote with). If < 4 survive the
    # filter, the solve abstains and the frame publishes NOTHING — downstream
    # Kalman-filters, so a dropped frame is cheap and a bad one is not.
    ransac=True,
    ransac_reproj_px=8.0,  # inlier bound, px
    ransac_iterations=126,  # subset cap; C(9,4)=126 = exhaustive for the gate
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

    def _kp_residuals(self, kps, ids, rvec, tvec, K, D):
        """Per-keypoint image residual under a pose, respecting soft slides.

        For a soft keypoint (known slide-along-edge error mode) the residual
        is only the component perpendicular to its projected slide segment —
        the same notion refine() optimizes, so RANSAC and LM agree on what
        counts as an error.
        """
        proj = project_points(self.model_points[ids], rvec, tvec, K, D)
        res = np.zeros(len(ids))
        for j, kid in enumerate(ids):
            d = proj[j] - kps[kid]
            if kid in self.soft:
                a, b = self.soft[kid]
                seg = project_points(np.stack([a, b]), rvec, tvec, K, D)
                t = seg[1] - seg[0]
                t /= max(np.linalg.norm(t), 1e-9)
                res[j] = abs(np.array([-t[1], t[0]]) @ d)
            else:
                res[j] = np.linalg.norm(d)
        return res

    def ransac_filter(self, kps, scores, K, D):
        """IPPE-RANSAC over the gated keypoints: (scores, n_outliers) with
        outliers zeroed (a copy; input never mutated). Below 5 gated points a
        4-point solve fits anything, so the filter passes through."""
        cfg = self.cfg
        ids = np.nonzero(scores >= cfg["score_gate"])[0]
        if len(ids) < 5:
            return scores, 0

        subsets = list(itertools.combinations(range(len(ids)), 4))
        if len(subsets) > cfg["ransac_iterations"]:
            rng = np.random.default_rng(0)  # deterministic across frames
            subsets = [
                subsets[i]
                for i in rng.choice(
                    len(subsets), cfg["ransac_iterations"], replace=False
                )
            ]

        thresh = cfg["ransac_reproj_px"]
        best_inliers = None
        best_score = (-1, np.inf)  # (count, mean residual): max count, min res
        for subset in subsets:
            sub_ids = ids[list(subset)]
            try:
                cands = solve_ippe(self.model_points[sub_ids], kps[sub_ids], K, D)
            except cv2.error:
                continue
            for rvec, tvec, _err in cands:
                if tvec[2] <= 0:
                    continue
                res = self._kp_residuals(kps, ids, rvec, tvec, K, D)
                inliers = res < thresh
                count = int(inliers.sum())
                mean_res = float(res[inliers].mean()) if count else np.inf
                if (count, -mean_res) > (best_score[0], -best_score[1]):
                    best_score = (count, mean_res)
                    best_inliers = inliers
            # Full consensus cannot be beaten: a clean frame costs one subset.
            if best_score[0] == len(ids):
                break

        if best_inliers is None:
            return scores, 0
        n_out = int((~best_inliers).sum())
        if n_out == 0:
            return scores, 0
        filtered = np.array(scores, dtype=float, copy=True)
        filtered[ids[~best_inliers]] = 0.0
        return filtered, n_out

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
        """ransac_filter + init_pose + refine in one call.

        Returns refine()'s dict (plus "n_outliers") or None. The RANSAC stage
        zeroes outlier keypoint scores, so init AND refine both run on the
        consensus set only; the dense mask term is untouched (its own level
        test guards it against contamination).
        """
        n_outliers = 0
        if self.cfg["ransac"]:
            scores, n_outliers = self.ransac_filter(kps, scores, K, D)
        init = self.init_pose(kps, scores, K, D, prior=prior)
        if init is None:
            return None
        rvec0, tvec0, _ = init
        result = self.refine(kps, scores, mask_prob, K, D, rvec0, tvec0)
        result["n_outliers"] = n_outliers
        return result

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
# The mission payload is the association letter <-> face colour
# (gate_tetra_overview.md §3). Letter keypoints are per-image glyph centres,
# not fixed 3D points — no pose to solve, only an association to keep
# deciding.
#
#   1. PER FRAME: keypoint-in-mask lookup on the soft mask probabilities,
#      nearest-mask fallback just outside. Letters under the score gate
#      contribute nothing — an averted-face letter has no training
#      supervision, so its prediction is garbage and the score is the only
#      defence.
#   2. OVER TIME: log-posterior over the 6 bijections {A,B,C}->{r,g,b} with
#      exponential forgetting. One-letter-per-face is then structural, two
#      seen letters pin the third ("inferred"), p_best is a real lock
#      criterion, and the clamp keeps a wrong lock reversible.
#
# Chirality is a config constant used only to lay out the drawn net — never
# estimated, never part of the association.

TETRA_DEFAULTS = dict(
    min_score=0.35,  # letters below this contribute no evidence at all
    inside_min=0.25,  # mask prob at the kp that counts as "in this face"
    max_mask_distance_px=40.0,  # nearest-mask fallback reach
    proximity_weight=0.6,  # fallback evidence is worth less than a hit
    eps=0.04,  # membership floor -> bounds one frame's influence
    # Memory: the arrangement is static, so forgetting exists only to keep a
    # wrong belief reversible (model errors are time-correlated). Swept on
    # real evidence (dream:~/tetra_unfold_check): tau=10/clip=20 matches the
    # pure accumulator's accuracy (100%/98% at 1/3 / 1/2 frames lying, vs
    # tau=3's 98%/84%) and still shakes off 30 wrong frames in ~1.2 s.
    tau=10.0,  # s, evidence forgetting time constant
    gain=0.25,  # per-frame log-likelihood gain
    logp_clip=20.0,  # log-posterior clamp (reversibility)
    # Lock criteria by sweep: 0.99/10 vs 0.9/5 costs one clean frame and cuts
    # wrong first-locks 6.8% -> 1.2% at 35% corrupted frames.
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

    Returns m (L, C) membership rows, w (L,) evidence weights (0 = ignore),
    and per-letter sources "in" | "near" | "none" | "low". The nearest-mask
    distance transforms are lazy — ~1 ms/class and most frames never need
    them.
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
        # "Inferred" = rests on the permutation constraint, not observations;
        # only meaningful once the filter as a whole has evidence.
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


# =============================================================================
# SECTION 4 — CROP TRACKING  (carry the crop between sparse objectness seeds)
# =============================================================================
#
# Objectness is 82% of the pipeline's GPU time (21.7 vs 4.8 ms forward on a
# 4060 Ti; ViT-S at 640x480 = 1200 tokens vs the joint ViT-B's 192) — on an
# AGX Orin the difference between ~7 and ~30 Hz. So it only seeds, and the
# crop is carried from the model's own output for zero extra compute.
# Measured on the ITU pool clip: carried-box output IoU 0.724 vs VitTrack's
# 0.606 (4.4 ms/frame), and FLAT in the seed interval — the residual is
# one-frame lag, not drift. The clip is 2.5 Hz, a worst case; re-validate on
# high-rate footage. Health checks matter more than the propagation: a crop
# that quietly stopped containing the object produces confident nonsense, and
# the top-down model has no "I am lost" signal of its own.
#
# `propagate_from` — "the masks are the silhouette" is a tetra fact, not a
# general one:
#   mask       union of the mask planes (tetra: 3 faces = the solid's
#              outline). The measured mode; default.
#   keypoints  bbox of the confident keypoints. Gate's one mask class is the
#              aperture MEMBRANE — inside the frame, missing feet and pinger
#              pole — so a mask box would starve the pole out of the crop and
#              trip the border check on every close-range frame. Gate's 9
#              keypoints ARE its silhouette.
# Decoded keypoints exist for all K even when nothing supervised them (the
# averted-letter case), hence propagate_min_score / propagate_ids.

CROP_TRACKER_DEFAULTS = dict(
    seed_period=2.0,  # s; the detector re-runs at least this often
    propagate_from="mask",  # mask | keypoints (see above)
    margin=0.18,  # padding around the extent, fraction of its size
    min_mask_pixels=200,  # mask mode: a silhouette smaller than this is not an object
    propagate_min_score=0.2,  # keypoints mode: score for a kp to bound the box
    propagate_ids=None,  # keypoints mode: which kps may bound it (None = all)
    min_keypoints=4,  # keypoints mode: fewer usable kps than this = re-seed
    border_slack_px=4.0,  # extent this close to the CROP edge = outgrown it
    max_area_ratio=2.5,  # frame-to-frame box area jump that must be a mistake
    min_confidence=0.0,  # mean score over confidence_ids below this = re-seed
    confidence_ids=None,  # which keypoints to judge on (None = all)
    # The border test must use the crop the model actually SAW (box grown to
    # the model aspect then padded 1.25x — box2cs), not the box: a correct
    # silhouette routinely extends past the box that produced it.
    crop_pad=1.25,
    crop_aspect=192.0 / 256.0,  # model input W/H
)


class CropTracker:
    """Carry the joint model's crop between objectness seeds.

    Usage per frame, from the bbox provider:

        if tracker.needs_seed(now):
            box, score = detector.predict(image)   # the expensive path
            tracker.seed(box, now)
        box = tracker.box                          # feed this to the model
        ...
        tracker.update(kps, scores, mask_probs, threshold, image_shape)

    `update` both propagates (next box from the segmentation) and validates
    (health checks); a failed check clears the box so `needs_seed` fires next
    frame. `last_reason` carries why, for logging.
    """

    def __init__(self, **config):
        self.cfg = dict(CROP_TRACKER_DEFAULTS)
        unknown = set(config) - set(CROP_TRACKER_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown CropTracker config: {sorted(unknown)}")
        self.cfg.update(config)
        if self.cfg["propagate_from"] not in ("mask", "keypoints"):
            raise ValueError(
                f"CropTracker propagate_from must be 'mask' or 'keypoints', "
                f"got {self.cfg['propagate_from']!r}"
            )
        ids = self.cfg["confidence_ids"]
        self._confidence_ids = None if ids is None else [int(i) for i in ids]
        ids = self.cfg["propagate_ids"]
        self._propagate_ids = None if ids is None else [int(i) for i in ids]
        self.reset()

    def reset(self):
        self.box: Optional[List[float]] = None
        self._seed_time: Optional[float] = None
        self.last_reason = "no box yet"
        self.seeds = 0
        self.carried = 0

    # ------------------------------------------------------------- seeding

    def needs_seed(self, now: float) -> bool:
        if self.box is None:
            return True
        if self._seed_time is None:
            return True
        return (now - self._seed_time) >= self.cfg["seed_period"]

    def seed(self, box, now: float) -> None:
        """Install a detector box (None = detector found nothing)."""
        self.box = None if box is None else [float(v) for v in box]
        self._seed_time = now
        if self.box is None:
            self.last_reason = "detector found nothing"
        else:
            self.seeds += 1
            self.last_reason = "seeded"

    # ------------------------------------------------------------- carrying

    def update(self, kps, scores, mask_probs, mask_threshold, image_shape) -> bool:
        """Propagate + validate from one frame's model output.

        Returns True if the crop survives into the next frame, False if a
        health check dropped it (the next frame will re-seed).
        """
        if self.box is None:
            return False
        confidence = self._mean_confidence(scores)
        if confidence is not None and confidence < self.cfg["min_confidence"]:
            return self._drop(f"keypoint confidence {confidence:.2f} too low")

        if self.cfg["propagate_from"] == "mask":
            extent = self._mask_extent(mask_probs, mask_threshold)
        else:
            extent = self._keypoint_extent(kps, scores)
        if extent is None:
            return False  # the extent helper already dropped with a reason
        x0, y0, x1, y1 = extent

        # Both propagation sources live inside the crop the model was given,
        # so an extent pressed against the CROP border (not the box's) means
        # the object likely continues past it — the one failure a
        # self-propagating box cannot see its way out of.
        slack = self.cfg["border_slack_px"]
        bw, bh = self.box[2], self.box[3]
        cx0, cy0, cx1, cy1 = self._crop_rect(self.box)
        if (
            x0 <= cx0 + slack
            or y0 <= cy0 + slack
            or x1 >= cx1 - slack
            or y1 >= cy1 - slack
        ):
            return self._drop(
                f"{self.cfg['propagate_from']} extent touches the crop border"
            )

        width, height = x1 - x0 + 1.0, y1 - y0 + 1.0
        mx, my = self.cfg["margin"] * width, self.cfg["margin"] * height
        img_h, img_w = image_shape[:2]
        nx0 = max(x0 - mx, 0.0)
        ny0 = max(y0 - my, 0.0)
        nx1 = min(x1 + mx, img_w - 1.0)
        ny1 = min(y1 + my, img_h - 1.0)
        new_box = [nx0, ny0, nx1 - nx0 + 1.0, ny1 - ny0 + 1.0]

        ratio = (new_box[2] * new_box[3]) / max(bw * bh, 1.0)
        limit = self.cfg["max_area_ratio"]
        if ratio > limit or ratio < 1.0 / limit:
            return self._drop(f"box area jumped {ratio:.1f}x")

        self.box = new_box
        self.carried += 1
        self.last_reason = "carried"
        return True

    # ------------------------------------------------------------- internals

    def _mask_extent(self, mask_probs, mask_threshold):
        """(x0, y0, x1, y1) of the union of the thresholded mask planes."""
        if mask_probs is None or not len(mask_probs):
            self._drop("no masks to propagate from")
            return None
        silhouette = (np.asarray(mask_probs) >= mask_threshold).any(axis=0)
        pixels = int(silhouette.sum())
        if pixels < self.cfg["min_mask_pixels"]:
            self._drop(f"silhouette {pixels} px below minimum")
            return None
        ys, xs = np.nonzero(silhouette)
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def _keypoint_extent(self, kps, scores):
        """(x0, y0, x1, y1) spanned by the usable keypoints."""
        if kps is None or scores is None or not len(kps):
            self._drop("no keypoints to propagate from")
            return None
        points = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        usable = values >= self.cfg["propagate_min_score"]
        if self._propagate_ids is not None:
            allowed = np.zeros_like(usable)
            for i in self._propagate_ids:
                if 0 <= i < len(allowed):
                    allowed[i] = True
            usable &= allowed
        count = int(usable.sum())
        if count < self.cfg["min_keypoints"]:
            self._drop(f"only {count} usable keypoints")
            return None
        chosen = points[usable]
        return (
            float(chosen[:, 0].min()),
            float(chosen[:, 1].min()),
            float(chosen[:, 0].max()),
            float(chosen[:, 1].max()),
        )

    def _crop_rect(self, box) -> Tuple[float, float, float, float]:
        """(x0, y0, x1, y1) the model actually sees for `box` — mirrors
        vitpose_inference.box2cs (kept local so this module stays torch-free)."""
        x, y, w, h = box
        cx, cy = x + w * 0.5, y + h * 0.5
        aspect = self.cfg["crop_aspect"]
        if w > aspect * h:
            h = w / aspect
        elif w < aspect * h:
            w = h * aspect
        half_w = w * self.cfg["crop_pad"] * 0.5
        half_h = h * self.cfg["crop_pad"] * 0.5
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    def _mean_confidence(self, scores) -> Optional[float]:
        if scores is None or self.cfg["min_confidence"] <= 0.0:
            return None
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if self._confidence_ids is not None:
            ids = [i for i in self._confidence_ids if 0 <= i < len(values)]
            if not ids:
                return None
            values = values[ids]
        return float(values.mean()) if len(values) else None

    def _drop(self, reason: str) -> bool:
        self.box = None
        self.last_reason = reason
        return False
