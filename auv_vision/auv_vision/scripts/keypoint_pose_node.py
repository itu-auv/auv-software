#!/usr/bin/env python3

import os
import sys
import threading
from collections import deque

import cv2
import numpy as np
import rospy
import yaml

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from auv_msgs.msg import KeypointResult
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)
import tf.transformations as tft

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import SHAPE_FACTORIES, transform_to_odom_and_publish


# ─────────────────────────────────────────────── debug viz constants

# Keypoint skeleton, 0-based indices into the per-frame keypoint array (id-1).
# 0..7 = bolt ring, 8 = face centre, 9 = arrow tip, 10 = handle end. The handle
# triplet (9-8-10) draws the arrow tip ── centre ── handle end line.
_SKELETON = [(i, (i + 1) % 8) for i in range(8)] + [(9, 8), (8, 10)]

COLOR_INLIER = (0, 220, 0)  # green — PnP inlier
COLOR_OUTLIER = (0, 0, 255)  # red — PnP outlier (reproj err > thresh)
COLOR_LOWCONF = (140, 140, 140)  # gray — below confidence gate
COLOR_HANDLE = (255, 180, 0)  # cyan/blue — handle-line keypoints (not in PnP)
COLOR_UNCLASS = (0, 200, 255)  # orange — high-conf, no PnP classification
COLOR_SKELETON = (0, 160, 0)
COLOR_BBOX = (0, 200, 255)
COLOR_STATE = (255, 255, 255)
COLOR_ACCEPT = (0, 220, 0)
COLOR_REJECT = (0, 0, 255)
COLOR_DIST = (255, 255, 0)

# Special single-letter labels for the handle keypoints.
_KP_LABELS = {9: "C", 10: "A", 11: "E"}


@dataclass
class Pose:
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3, 1)


@dataclass
class OutputFrame:
    child_frame_id: str
    offset: np.ndarray  # (3,) in model frame


@dataclass
class HandleLine:
    """Keypoints outside the PnP set that lie on the valve handle line.

    Collinearity is checked directly in the image: the keypoint pixels are fit
    with a 2D line and the max perpendicular residual must stay below
    `max_line_error_px` (pixels). This gate is independent of the PnP attitude —
    a straight handle projects to a straight line regardless of how the pose
    wobbles. The 3D handle direction (which overrides rotation about model X)
    is then taken from the same keypoints back-projected onto the valve plane.
    `arrow_id` fixes the line's sign — the axis points from the face centre
    toward the arrow keypoint. All listed keypoints must be present and
    high-confidence or the frame is skipped."""

    keypoint_ids: List[int]
    arrow_id: int
    output_axis: str = "z"
    confidence_threshold: float = 0.5
    parallel_eps_rad: float = 0.087
    max_line_error_px: float = 5.0


@dataclass
class KeypointObjectConfig:
    name: str
    object_names: List[str]
    cameras: List[str]
    pose_keypoints: Dict[int, np.ndarray]  # keypoint id -> (3,) point in model frame
    outputs: List[OutputFrame]
    min_keypoints: int
    solver: str
    max_distance: float
    reprojection_error_threshold: float
    confidence_threshold: float
    refine_iterative: bool
    handle_line: Optional[HandleLine] = None


@dataclass
class PnPOutcome:
    """Result of a single PnP attempt, plus the debug data needed to draw it.

    `success` means a usable pose was produced; on failure `reason` says why and
    the inlier arrays may still be populated (e.g. a solve that landed but had
    too few inliers) so the debug overlay can show what happened."""

    success: bool
    reason: str = ""
    pose: Optional[Pose] = None
    ids_used: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    inlier_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


@dataclass
class OutputDebug:
    child_frame_id: str
    published: bool
    reason: str
    base_link_xyz: Optional[Tuple[float, float, float]] = None


@dataclass
class KeypointBatch:
    """All keypoints from one frame that match a single estimator's
    object_names. Unfiltered — each consumer applies its own conf check."""

    estimator: "PnPEstimator"
    ids: np.ndarray  # (M,)
    pixels: np.ndarray  # (M, 2)
    confidences: np.ndarray  # (M,)


class PnPEstimator:
    _SOLVER_FLAGS = {
        "iterative": cv2.SOLVEPNP_ITERATIVE,
        "p3p": cv2.SOLVEPNP_P3P,
        "ap3p": cv2.SOLVEPNP_AP3P,
        "epnp": cv2.SOLVEPNP_EPNP,
        "ippe": cv2.SOLVEPNP_IPPE,
        "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE,
        "sqpnp": cv2.SOLVEPNP_SQPNP,
        "dls": cv2.SOLVEPNP_DLS,
        "upnp": cv2.SOLVEPNP_UPNP,
    }

    def __init__(self, config: KeypointObjectConfig):
        self.config = config
        sorted_ids = sorted(config.pose_keypoints)
        self._pose_ids = np.array(sorted_ids, dtype=np.int32)
        self._model_points = np.stack(
            [config.pose_keypoints[kid] for kid in sorted_ids]
        ).astype(np.float64)
        self._index_by_id = {int(kid): i for i, kid in enumerate(sorted_ids)}
        self._solver_flag = self._SOLVER_FLAGS.get(
            config.solver, cv2.SOLVEPNP_ITERATIVE
        )
        # Last good handle observation (camera frame): the flange normal and the
        # in-plane handle direction, cached so we can transport the roll onto the
        # current frame while the handle keypoints are occluded.
        self._handle_normal: Optional[np.ndarray] = None
        self._handle_dir: Optional[np.ndarray] = None
        self._handle_time: Optional[rospy.Time] = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def has_handle_line(self) -> bool:
        return self.config.handle_line is not None

    def estimate_pose(
        self, batch: KeypointBatch, K: np.ndarray, D: np.ndarray
    ) -> PnPOutcome:
        cfg = self.config
        in_model = np.isin(batch.ids, self._pose_ids)
        high_conf = batch.confidences >= cfg.confidence_threshold
        mask = in_model & high_conf
        n_used = int(mask.sum())
        if n_used < cfg.min_keypoints:
            return PnPOutcome(
                success=False,
                reason=f"too few kps ({n_used}/{cfg.min_keypoints})",
            )

        ids_used = batch.ids[mask]
        indices = np.array([self._index_by_id[int(k)] for k in ids_used])
        image_pts = batch.pixels[mask].reshape(-1, 1, 2).astype(np.float64)
        model_pts = self._model_points[indices].reshape(-1, 1, 3).astype(np.float64)

        reproj_thresh = cfg.reprojection_error_threshold

        # solvePnPGeneric exposes BOTH coplanar IPPE candidates; plain solvePnP
        # only ever hands back one. We work with the lowest-reprojection-error
        # candidate — no disambiguation beyond that ordering. (solvePnPRansac is
        # unusable here: it forces EPnP internally, which fails on coplanar
        # subsets — hence manual rejection.)
        try:
            n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                model_pts, image_pts, K, D, flags=self._solver_flag
            )
        except cv2.error as e:
            rospy.logwarn_throttle(
                2.0, f"PnP '{cfg.name}': solvePnPGeneric failed: {e}"
            )
            return PnPOutcome(success=False, reason="solvePnPGeneric error")
        if not n_sols:
            return PnPOutcome(success=False, reason="no PnP solution")

        # Pick the candidate with the lowest mean reprojection error.
        best = None
        for rvec, tvec in zip(rvecs, tvecs):
            proj, _ = cv2.projectPoints(model_pts, rvec, tvec, K, D)
            errs = np.linalg.norm(
                proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1
            )
            if best is None or errs.mean() < best[0]:
                best = (float(errs.mean()), rvec, tvec)

        _, rvec, tvec = best
        proj, _ = cv2.projectPoints(model_pts, rvec, tvec, K, D)
        errors = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
        inliers = errors <= reproj_thresh
        n_inliers = int(inliers.sum())

        rospy.loginfo_throttle(
            2.0,
            f"PnP '{cfg.name}': {n_sols} cand, {n_inliers}/{len(model_pts)} "
            f"inliers, max reproj err: {errors.max():.2f}px "
            f"(thresh {reproj_thresh:.1f}px, n_used={n_used})",
        )

        if n_inliers < cfg.min_keypoints:
            return PnPOutcome(
                success=False,
                reason=f"{n_inliers}/{len(model_pts)} inliers < {cfg.min_keypoints}",
                ids_used=ids_used,
                inlier_mask=inliers,
                rvec=rvec,
                tvec=tvec,
            )

        rvec, tvec = self._polish(rvec, tvec, model_pts, image_pts, inliers, K, D)
        if rvec is None:
            return PnPOutcome(
                success=False,
                reason="polish failed",
                ids_used=ids_used,
                inlier_mask=inliers,
            )

        R, _ = cv2.Rodrigues(rvec)
        return PnPOutcome(
            success=True,
            pose=Pose(R=R, t=tvec),
            ids_used=ids_used,
            inlier_mask=inliers,
            rvec=rvec,
            tvec=tvec,
        )

    def _polish(self, rvec, tvec, model_pts, image_pts, inliers, K, D):
        cfg = self.config
        n_inliers = int(inliers.sum())
        n_total = len(model_pts)

        # ITERATIVE (Levenberg–Marquardt) is the only PnP flag that honours
        # useExtrinsicGuess; every other solver ignores it and re-solves
        # from scratch — so the guess only buys you anything with ITERATIVE.
        if cfg.refine_iterative:
            ok, rvec, tvec = cv2.solvePnP(
                model_pts[inliers],
                image_pts[inliers],
                K,
                D,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        elif n_inliers < n_total:
            ok, rvec, tvec = cv2.solvePnP(
                model_pts[inliers],
                image_pts[inliers],
                K,
                D,
                flags=self._solver_flag,
            )
        else:
            ok = True

        return (rvec, tvec) if ok else (None, None)

    @staticmethod
    def _ray_plane_intersection(pixel, K, D, normal, plane_point, parallel_eps_rad):
        """Back-project a pixel onto the valve plane (camera frame).

        Returns the 3D intersection of the pixel ray with the plane through
        `plane_point` with normal `normal`, or None when the ray is nearly
        in-plane (valve seen edge-on) or the hit is behind the camera.
        """
        undist = cv2.undistortPoints(pixel.reshape(1, 1, 2).astype(np.float64), K, D)
        ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])

        # |normal · ray| / |ray| = sin(angle between ray and plane); skip
        # when the valve is seen edge-on (denom ≈ 0 → ray nearly in-plane).
        denom = float(normal @ ray)
        ray_norm = float(np.linalg.norm(ray))
        if abs(denom) < np.sin(parallel_eps_rad) * ray_norm:
            return None

        t_param = float(normal @ plane_point) / denom
        if t_param <= 0.0:
            return None
        return t_param * ray

    def apply_handle_line(
        self, pose: Pose, batch: KeypointBatch, K: np.ndarray, D: np.ndarray
    ) -> Optional[Pose]:
        """Replace rotation about model X with the direction of the line fit
        through the handle keypoints. Returns None when too few keypoints are
        available, the fit residual exceeds the threshold, or the geometry is
        degenerate."""
        hl = self.config.handle_line
        normal = pose.R[:, 0]
        center = pose.t.flatten()

        # Gather present, high-confidence handle keypoint pixels, in the order
        # of hl.keypoint_ids so the arrow stays identifiable downstream.
        pixels: List[np.ndarray] = []
        for kid in hl.keypoint_ids:
            matches = np.where(batch.ids == kid)[0]
            if len(matches) == 0:
                break
            i = matches[0]
            if batch.confidences[i] < hl.confidence_threshold:
                break
            pixels.append(batch.pixels[i].astype(np.float64))

        # Require every listed handle keypoint (over-determined + meaningful gate).
        if len(pixels) < len(hl.keypoint_ids):
            return None
        pixels = np.stack(pixels)  # (N, 2)

        # ── Collinearity gate, in image pixels (independent of PnP attitude).
        # Undistort to pixel coords (P=K) so the residual is a true pixel error.
        und = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        c2 = und - und.mean(axis=0)
        _, _, vt2 = np.linalg.svd(c2)
        dir2 = vt2[0]
        perp2 = c2 - np.outer(c2 @ dir2, dir2)
        max_err = float(np.linalg.norm(perp2, axis=1).max())
        if max_err > hl.max_line_error_px:
            rospy.logwarn_throttle(
                2.0,
                f"'{self.config.name}': handle line fit residual "
                f"{max_err:.1f}px > {hl.max_line_error_px:.1f}px, skipping",
            )
            return None

        # ── Handle direction in 3D: back-project onto the valve plane and fit a
        # line through the in-plane points (only the direction is used, not the
        # residual — that was already gated above in clean pixel space).
        plane_pts: List[np.ndarray] = []
        for px in pixels:
            pt = self._ray_plane_intersection(
                px, K, D, normal, center, hl.parallel_eps_rad
            )
            if pt is None:  # valve seen edge-on → direction undefined
                return None
            plane_pts.append(pt)
        pts = np.stack(plane_pts)
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]

        # Force exactly in-plane, then orient: point from face centre to arrow.
        direction = direction - (normal @ direction) * normal
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction = direction / norm
        arrow_pt = plane_pts[hl.keypoint_ids.index(hl.arrow_id)]
        if float(direction @ (arrow_pt - center)) < 0.0:
            direction = -direction

        return Pose(R=self._assemble_frame(normal, direction, hl.output_axis), t=pose.t)

    @staticmethod
    def _assemble_frame(
        normal: np.ndarray, direction: np.ndarray, output_axis: str
    ) -> np.ndarray:
        """Build a rotation matrix with X = flange normal and the handle
        `direction` placed on the configured in-plane axis."""
        if output_axis == "z":
            x_axis, z_axis = normal, direction
            y_axis = np.cross(z_axis, x_axis)
        else:  # "y", validated at config load
            x_axis, y_axis = normal, direction
            z_axis = np.cross(x_axis, y_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    @staticmethod
    def _min_arc_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Minimal (shortest-arc) rotation matrix taking unit vector a to b.

        Rodrigues' formula about the axis a×b. Used to transport a cached
        handle direction as the flange normal tilts, with no twist about the
        normal — the "good enough" assumption when the camera doesn't roll
        about the valve axis between the anchor frame and now."""
        v = np.cross(a, b)
        c = float(a @ b)
        s = float(np.linalg.norm(v))
        if s < 1e-9:  # already (anti)parallel
            return np.eye(3) if c > 0 else -np.eye(3)
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    def resolve_orientation(
        self, pnp_pose: Pose, batch: KeypointBatch, K, D, stamp
    ) -> Tuple[Optional[Pose], str]:
        """Apply the handle-line roll, with an occlusion fallback that transports
        the last good handle direction onto the current frame's flange normal.

        Returns (pose, source) where source is:
          - "live"           : handle keypoints present this frame (re-anchored)
          - "transported Xs"  : handle occluded; cached direction carried onto the
                                current normal (X s since the last live handle)
          - "none"           : handle occluded and no anchor yet → pose is None

        Both the cached direction and the current normal live in the camera
        frame and come from label-free geometry (the flange normal is invariant
        to the 8-fold bolt relabeling, and nothing here touches localization),
        so this survives both bolt label swaps and DVL/odom drift."""
        hl = self.config.handle_line
        axis_col = 2 if hl.output_axis == "z" else 1

        live = self.apply_handle_line(pnp_pose, batch, K, D)
        if live is not None:
            # Anchor: cache the flange normal + handle direction from this frame.
            self._handle_normal = live.R[:, 0].copy()
            self._handle_dir = live.R[:, axis_col].copy()
            self._handle_time = stamp
            return live, "live"

        # Handle occluded → transport the cached direction onto the live normal.
        if self._handle_normal is None or self._handle_time is None:
            return None, "none"

        age = (stamp - self._handle_time).to_sec()

        # PnP's normal + translation are invariant to the bolt relabeling, so
        # they're trustworthy even though PnP's in-plane (roll) axes are not.
        n_curr = pnp_pose.R[:, 0]
        n_curr = n_curr / np.linalg.norm(n_curr)
        Rm = self._min_arc_rotation(self._handle_normal, n_curr)
        direction = Rm @ self._handle_dir
        # Re-seat exactly in-plane against the current normal, then normalize.
        direction = direction - (n_curr @ direction) * n_curr
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None, "none"
        direction = direction / norm

        R = self._assemble_frame(n_curr, direction, hl.output_axis)
        return Pose(R=R, t=pnp_pose.t), f"transported {age:.1f}s"


class CameraHandler:
    def __init__(
        self,
        name: str,
        camera_frame: str,
        camera_info_topic: str,
        keypoint_topic: str,
        image_topic: str,
        estimators: List[PnPEstimator],
        tf_buffer: tf2_ros.Buffer,
        publisher: rospy.Publisher,
        base_link_frame: str,
        axis_length: float,
    ):
        self.name = name
        self.camera_frame = camera_frame
        self.estimators = estimators
        self.tf_buffer = tf_buffer
        self.publisher = publisher
        self.base_link_frame = base_link_frame
        self.axis_length = axis_length
        self._calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._bridge = CvBridge()
        # Small ring of recent frames so the debug overlay can be drawn on the
        # exact frame the keypoints came from (matched by header stamp).
        self._image_lock = threading.Lock()
        self._image_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=60)

        self._estimator_by_object_name: Dict[str, PnPEstimator] = {}
        for est in estimators:
            for alias in est.config.object_names:
                if alias in self._estimator_by_object_name:
                    raise ValueError(
                        f"object_name '{alias}' on camera '{name}' is claimed "
                        f"by both '{self._estimator_by_object_name[alias].name}' "
                        f"and '{est.name}'"
                    )
                self._estimator_by_object_name[alias] = est

        self._debug_image_pub = rospy.Publisher(
            f"keypoint_pose_image_{name}/compressed", CompressedImage, queue_size=1
        )

        rospy.Subscriber(camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(
            image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24
        )
        rospy.Subscriber(
            keypoint_topic, KeypointResult, self._keypoint_cb, queue_size=1
        )

    def _info_cb(self, msg: CameraInfo):
        K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.D, dtype=np.float64)
        self._calibration = (K, D)

    def _image_cb(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[{self.name}] image decode failed: {e}")
            return
        with self._image_lock:
            self._image_buf.append((msg.header.stamp.to_sec(), img))

    def _nearest_image(self, stamp) -> Optional[np.ndarray]:
        target = stamp.to_sec()
        with self._image_lock:
            if not self._image_buf:
                return None
            t, img = min(self._image_buf, key=lambda e: abs(e[0] - target))
        return img

    def _keypoint_cb(self, msg: KeypointResult):
        if self._calibration is None:
            return
        K, D = self._calibration

        # Per-estimator debug accumulator for the overlay.
        debug: Dict[str, Dict] = {}

        for batch in self._group(msg):
            est = batch.estimator
            outcome = est.estimate_pose(batch, K, D)

            final_pose: Optional[Pose] = outcome.pose
            accepted = False
            reason = outcome.reason
            handle_source = ""
            outputs_dbg: List[OutputDebug] = []

            if not outcome.success:
                rospy.logwarn_throttle(
                    5.0,
                    f"[{self.name}] PnP failed for '{est.name}' "
                    f"({len(batch.ids)} keypoints): {reason}",
                )
            else:
                publish_pose = outcome.pose
                if est.has_handle_line:
                    publish_pose, handle_source = est.resolve_orientation(
                        outcome.pose, batch, K, D, msg.header.stamp
                    )

                if publish_pose is None:
                    reason = "handle line missing (no handle anchor)"
                    rospy.loginfo_throttle(
                        2.0,
                        f"[{self.name}] '{est.name}': {reason}, skipping publish",
                    )
                else:
                    final_pose = publish_pose
                    outputs_dbg = self._publish_outputs(
                        est, publish_pose, msg.header.stamp
                    )
                    accepted = any(o.published for o in outputs_dbg)
                    if not accepted and outputs_dbg:
                        reason = outputs_dbg[0].reason

            debug[est.name] = {
                "batch": batch,
                "outcome": outcome,
                "final_pose": final_pose if accepted else outcome.pose,
                "accepted": accepted,
                "reason": reason,
                "handle_source": handle_source,
                "outputs": outputs_dbg,
            }

        self._publish_debug_image(msg, debug, K, D)

    def _group(self, msg: KeypointResult) -> List[KeypointBatch]:
        by_estimator: Dict[
            str,
            Tuple[PnPEstimator, List[int], List[Tuple[float, float]], List[float]],
        ] = {}
        for kp in msg.keypoints:
            est = self._estimator_by_object_name.get(kp.object)
            if est is None:
                continue
            if est.name not in by_estimator:
                by_estimator[est.name] = (est, [], [], [])
            _, ids, pts, confs = by_estimator[est.name]
            ids.append(kp.id)
            pts.append((kp.x, kp.y))
            confs.append(kp.confidence)

        return [
            KeypointBatch(
                estimator=est,
                ids=np.array(ids, dtype=np.int32),
                pixels=np.array(pts, dtype=np.float64),
                confidences=np.array(confs, dtype=np.float64),
            )
            for est, ids, pts, confs in by_estimator.values()
        ]

    def _to_base_link(self, x, y, z, stamp) -> Optional[Tuple[float, float, float]]:
        """Express a camera-frame point in base_link, for the distance readout."""
        ps = PointStamped()
        ps.header.frame_id = self.camera_frame
        ps.header.stamp = stamp
        ps.point.x, ps.point.y, ps.point.z = x, y, z
        try:
            out = self.tf_buffer.transform(
                ps, self.base_link_frame, rospy.Duration(0.2)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"[{self.name}] base_link transform: {e}")
            return None
        return (out.point.x, out.point.y, out.point.z)

    def _publish_outputs(
        self, est: PnPEstimator, pose: Pose, stamp
    ) -> List[OutputDebug]:
        M = np.eye(4)
        M[:3, :3] = pose.R
        quat = tft.quaternion_from_matrix(M)

        results: List[OutputDebug] = []
        for output in est.config.outputs:
            child_pos = pose.R @ output.offset.reshape(3, 1) + pose.t
            x, y, z = (
                float(child_pos[0, 0]),
                float(child_pos[1, 0]),
                float(child_pos[2, 0]),
            )
            base_xyz = self._to_base_link(x, y, z, stamp)

            if z <= 0.0:
                rospy.logdebug(f"Behind-camera: {output.child_frame_id} z={z:.2f}")
                results.append(
                    OutputDebug(output.child_frame_id, False, "behind camera", base_xyz)
                )
                continue

            distance = float(np.linalg.norm(child_pos))
            if distance > est.config.max_distance:
                rospy.logdebug(
                    f"Too far: {output.child_frame_id} "
                    f"d={distance:.2f} > {est.config.max_distance}"
                )
                results.append(
                    OutputDebug(
                        output.child_frame_id,
                        False,
                        f"too far ({distance:.1f}m)",
                        base_xyz,
                    )
                )
                continue

            transform_to_odom_and_publish(
                self.camera_frame,
                output.child_frame_id,
                x,
                y,
                z,
                stamp,
                self.tf_buffer,
                self.publisher,
                rotation_quat=quat,
            )
            results.append(OutputDebug(output.child_frame_id, True, "", base_xyz))
        return results

    # ─────────────────────────────────────────────── debug overlay

    def _publish_debug_image(self, msg: KeypointResult, debug: Dict, K, D):
        if self._debug_image_pub.get_num_connections() == 0:
            return
        vis = self._nearest_image(msg.header.stamp)
        if vis is None:
            return
        vis = vis.copy()

        # Producer state + tracker bbox (carried through the message).
        cv2.putText(
            vis,
            f"state: {msg.state}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_STATE,
            2,
        )
        if len(msg.bbox) == 4:
            bx, by, bw, bh = msg.bbox
            cv2.rectangle(
                vis,
                (int(bx), int(by)),
                (int(bx + bw), int(by + bh)),
                COLOR_BBOX,
                2,
            )
            cv2.putText(
                vis,
                "VITTrack",
                (int(bx), int(by) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_BBOX,
                1,
            )

        y_text = 52
        for est in self.estimators:
            d = debug.get(est.name)
            if d is None:
                continue
            y_text = self._draw_estimator(vis, est, d, K, D, y_text)

        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        ok, encoded = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        out.data = encoded.tobytes()
        self._debug_image_pub.publish(out)

    def _draw_estimator(self, vis, est, d, K, D, y_text):
        batch: KeypointBatch = d["batch"]
        outcome: PnPOutcome = d["outcome"]
        conf_thresh = est.config.confidence_threshold

        # id -> inlier? for the model points that went through PnP.
        inlier_by_id: Dict[int, bool] = {}
        if outcome.ids_used.size and outcome.inlier_mask.size:
            for kid, inl in zip(outcome.ids_used, outcome.inlier_mask):
                inlier_by_id[int(kid)] = bool(inl)

        handle_ids = set()
        if est.has_handle_line:
            handle_ids = set(est.config.handle_line.keypoint_ids)

        # id -> (x, y, conf) for skeleton + point drawing.
        pt_by_id: Dict[int, Tuple[float, float, float]] = {}
        for kid, (px, py), conf in zip(batch.ids, batch.pixels, batch.confidences):
            pt_by_id[int(kid)] = (float(px), float(py), float(conf))

        # Skeleton (only between present, high-conf keypoints).
        for a, b in _SKELETON:
            ia, ib = a + 1, b + 1
            if ia in pt_by_id and ib in pt_by_id:
                if pt_by_id[ia][2] >= conf_thresh and pt_by_id[ib][2] >= conf_thresh:
                    cv2.line(
                        vis,
                        (int(pt_by_id[ia][0]), int(pt_by_id[ia][1])),
                        (int(pt_by_id[ib][0]), int(pt_by_id[ib][1])),
                        COLOR_SKELETON,
                        2,
                    )

        # Keypoints, coloured by role.
        for kid, (px, py, conf) in pt_by_id.items():
            pt = (int(px), int(py))
            if conf < conf_thresh:
                color, filled = COLOR_LOWCONF, False
            elif kid in inlier_by_id:
                color = COLOR_INLIER if inlier_by_id[kid] else COLOR_OUTLIER
                filled = True
            elif kid in handle_ids:
                color, filled = COLOR_HANDLE, True
            else:
                color, filled = COLOR_UNCLASS, True
            cv2.circle(vis, pt, 5 if filled else 4, color, -1 if filled else 1)
            label = _KP_LABELS.get(kid, str(kid))
            cv2.putText(
                vis,
                f"{label} ({conf:.2f})",
                (pt[0] + 6, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        # Pose axis (drawn from whatever pose we have, even if later rejected).
        final_pose: Optional[Pose] = d["final_pose"]
        if final_pose is not None:
            rvec, _ = cv2.Rodrigues(final_pose.R)
            try:
                cv2.drawFrameAxes(vis, K, D, rvec, final_pose.t, self.axis_length, 2)
            except cv2.error as e:
                rospy.logwarn_throttle(5.0, f"[{self.name}] drawFrameAxes: {e}")

        # Text block: inliers, accept/reject, per-output base_link distance.
        n_in = int(outcome.inlier_mask.sum()) if outcome.inlier_mask.size else 0
        n_tot = int(outcome.ids_used.size)
        cv2.putText(
            vis,
            f"{est.name}  inliers: {n_in}/{n_tot}",
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_STATE,
            1,
        )
        y_text += 20

        status = "ACCEPTED" if d["accepted"] else "REJECTED"
        status_color = COLOR_ACCEPT if d["accepted"] else COLOR_REJECT
        handle_src = d.get("handle_source", "")
        if d["accepted"]:
            status_txt = (
                status if not handle_src else f"{status} (handle: {handle_src})"
            )
        else:
            status_txt = f"{status}: {d['reason']}"
        cv2.putText(
            vis,
            status_txt,
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2,
        )
        y_text += 22

        for od in d["outputs"]:
            if od.base_link_xyz is None:
                continue
            dx, dy, dz = od.base_link_xyz
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            cv2.putText(
                vis,
                f"{od.child_frame_id}  base_link x={dx:.2f} y={dy:.2f} "
                f"z={dz:.2f} (d={dist:.2f}m)",
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_DIST,
                1,
            )
            y_text += 18

        return y_text + 6


def _build_pose_keypoints(model_spec) -> Dict[int, np.ndarray]:
    result: Dict[int, np.ndarray] = {}
    for part in model_spec:
        sub_points = SHAPE_FACTORIES[part["shape"]](part["args"])
        sub_offset = np.array(part.get("offset", [0, 0, 0]), dtype=np.float64)
        ids = part["ids"]
        if len(ids) != len(sub_points):
            raise ValueError(
                f"shape '{part['shape']}' produced {len(sub_points)} points "
                f"but `ids` lists {len(ids)} entries"
            )
        for kid, pt in zip(ids, sub_points):
            kid = int(kid)
            if kid in result:
                raise ValueError(f"duplicate keypoint id {kid} in pose model")
            result[kid] = np.asarray(pt, dtype=np.float64) + sub_offset
    return result


def load_config(config_path: str) -> Tuple[List[KeypointObjectConfig], Dict]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    configs: List[KeypointObjectConfig] = []
    for obj_cfg in config["objects"]:
        pose_keypoints = _build_pose_keypoints(obj_cfg["model"])
        outputs = [
            OutputFrame(
                child_frame_id=out["child_frame_id"],
                offset=np.array(out["offset"], dtype=np.float64),
            )
            for out in obj_cfg["outputs"]
        ]

        handle_line = None
        if "handle_line" in obj_cfg:
            hl_cfg = obj_cfg["handle_line"]
            output_axis = hl_cfg.get("output_axis", "z")
            if output_axis not in ("y", "z"):
                raise ValueError(
                    f"handle_line.output_axis for '{obj_cfg['name']}' "
                    f"must be 'y' or 'z', got '{output_axis}'"
                )
            keypoint_ids = [int(k) for k in hl_cfg["keypoint_ids"]]
            if len(keypoint_ids) < 3:
                raise ValueError(
                    f"handle_line.keypoint_ids for '{obj_cfg['name']}' must list "
                    f"at least 3 keypoints, got {keypoint_ids}"
                )
            clash = [k for k in keypoint_ids if k in pose_keypoints]
            if clash:
                raise ValueError(
                    f"handle_line.keypoint_ids {clash} for '{obj_cfg['name']}' "
                    f"clash with pose model ids — handle points must not be "
                    f"used by PnP"
                )
            arrow_id = int(hl_cfg["arrow_id"])
            if arrow_id not in keypoint_ids:
                raise ValueError(
                    f"handle_line.arrow_id {arrow_id} for '{obj_cfg['name']}' "
                    f"must be one of keypoint_ids {keypoint_ids}"
                )
            handle_line = HandleLine(
                keypoint_ids=keypoint_ids,
                arrow_id=arrow_id,
                output_axis=output_axis,
                confidence_threshold=float(hl_cfg.get("confidence_threshold", 0.5)),
                parallel_eps_rad=float(hl_cfg.get("parallel_eps_rad", 0.087)),
                max_line_error_px=float(hl_cfg.get("max_line_error_px", 5.0)),
            )

        configs.append(
            KeypointObjectConfig(
                name=obj_cfg["name"],
                object_names=obj_cfg.get("object_names", [obj_cfg["name"]]),
                cameras=obj_cfg["cameras"],
                pose_keypoints=pose_keypoints,
                outputs=outputs,
                min_keypoints=obj_cfg.get("min_keypoints", 6),
                solver=obj_cfg.get("solver", "iterative"),
                max_distance=obj_cfg.get("max_distance", 30.0),
                reprojection_error_threshold=obj_cfg.get(
                    "reprojection_error_threshold", 5.0
                ),
                confidence_threshold=obj_cfg.get("confidence_threshold", 0.3),
                refine_iterative=bool(obj_cfg.get("refine_iterative", False)),
                handle_line=handle_line,
            )
        )

    return configs, config["cameras"]


class KeypointPoseNode:
    def __init__(self):
        rospy.init_node("keypoint_pose_node")

        default_config = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "config",
                "valve_keypoints.yaml",
            )
        )
        config_path = rospy.get_param("~config_file", default_config)
        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link")
        self.axis_length = float(rospy.get_param("~axis_length", 0.15))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publisher = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )

        configs, camera_configs = load_config(config_path)
        estimators = [PnPEstimator(c) for c in configs]

        estimators_by_camera: Dict[str, List[PnPEstimator]] = {
            cam: [] for cam in camera_configs
        }
        for est in estimators:
            for cam_name in est.config.cameras:
                if cam_name in estimators_by_camera:
                    estimators_by_camera[cam_name].append(est)

        self.handlers: Dict[str, CameraHandler] = {}
        for cam_name, cam_cfg in camera_configs.items():
            cam_estimators = estimators_by_camera.get(cam_name, [])
            if not cam_estimators:
                continue
            self.handlers[cam_name] = CameraHandler(
                name=cam_name,
                camera_frame=cam_cfg["frame"],
                camera_info_topic=cam_cfg["camera_info_topic"],
                keypoint_topic=cam_cfg["keypoint_topic"],
                image_topic=cam_cfg["image_topic"],
                estimators=cam_estimators,
                tf_buffer=self.tf_buffer,
                publisher=self.publisher,
                base_link_frame=self.base_link_frame,
                axis_length=self.axis_length,
            )

        total_pts = sum(len(e.config.pose_keypoints) for e in estimators)
        total_outputs = sum(len(e.config.outputs) for e in estimators)
        rospy.loginfo(
            f"KeypointPoseNode started — {len(estimators)} objects, "
            f"{total_pts} model points, {total_outputs} output frames, "
            f"{len(self.handlers)} cameras"
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = KeypointPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
