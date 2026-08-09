"""Fused planar pose estimation: keypoint PnP + dense mask-edge refinement.

ROS-free (numpy / cv2 / scipy only). Consumed by vitpose ops (gate_pose).

Design + empirical verification: VITPOSE_PLAN.md §5 and the offline harness
at dream:~/gate_fusion_verify (2026-08-09). Verified against pseudo-GT on the
gate_joint_1000 val split:
  - clean images: fused == kp-only == label-noise floor (~0.5 deg / 1 cm);
  - occlusion workload (only the 4 near-collinear top kps survive, bottom of
    the mask corrupted): kp-only 4.7 deg / 10 cm median, worst 30 deg —
    fused 2 deg / 4 cm median, worst 6 deg, nearly flat in the occluded
    fraction (the aperture side edges carry the pitch information).

Formulation (RAPiD-style outer/inner loop over SE(3)):
  outer: project the known planar boundary polygon under the current pose;
         for each boundary sample, search along its image normal in the mask
         probability map for the 0.5 crossing -> fixed 1D targets.
  inner: scipy least_squares over (rvec, tvec), Huber loss:
         - keypoint reprojection residuals, score-weighted; "soft" keypoints
           (the gate post midpoints, which slide along their edge) contribute
           only the residual component perpendicular to their projected
           slide segment;
         - mask residuals: normal-projected distance to the crossing targets,
           sharpness-weighted.
  The normal-search range shrinks per outer iteration (coarse-to-fine).

Two robustness elements that verification proved necessary:
  1. Two-sided LEVEL TEST per crossing: a genuine boundary edge separates
     confident mask (P ~ 1) from confident background (P ~ 0); a
     contamination boundary (occluder cut, uncertain region) has a mid-level
     far side. Crossings failing the test carry zero weight. Without this,
     an occluder edge near the true boundary captures the dense term and
     fusion underperforms kp-only. (Leave-one-edge-out consensus was tried
     and REJECTED: with few keypoints every edge is load-bearing, so the
     leave-out pose drifts and good edges get dropped.)
  2. IPPE degenerates outright (zero candidates) when the surviving
     keypoints are near-collinear — exactly the occlusion case. estimate()
     therefore falls back to ITERATIVE PnP seeded from a prior pose (last
     accepted pose, or a canonical face-on guess).
"""

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

DEFAULTS = dict(
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


def _bilinear(mask, xy):
    """Sample mask (H, W) at xy (N, 2) pixel coords; outside -> 0."""
    return map_coordinates(
        mask, [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=0.0
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
        config: overrides for DEFAULTS.
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
        self.cfg = dict(DEFAULTS)
        unknown = set(config) - set(DEFAULTS)
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
            vals = _bilinear(mask, line) - 0.5
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
            sides = _bilinear(
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
