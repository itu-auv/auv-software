#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np
import rospy
import yaml

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from auv_msgs.msg import KeypointResult
from geometry_msgs.msg import TransformStamped, PoseArray, Pose as GeomPose
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64MultiArray

import tf2_ros
import tf.transformations as tft

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import SHAPE_FACTORIES, transform_to_odom_and_publish


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
class EstimationResult:
    """Best pose plus every candidate the solver returned (for debug logging).

    `candidates` is sorted ascending by mean reprojection error, so index 0 is
    the published solution. Each entry is (mean_reproj_err_px, R(3x3), t(3x1))."""

    pose: Pose
    candidates: List[Tuple[float, np.ndarray, np.ndarray]]


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

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def has_handle_line(self) -> bool:
        return self.config.handle_line is not None

    def estimate_pose(
        self, batch: KeypointBatch, K: np.ndarray, D: np.ndarray
    ) -> Optional[EstimationResult]:
        cfg = self.config
        in_model = np.isin(batch.ids, self._pose_ids)
        high_conf = batch.confidences >= cfg.confidence_threshold
        mask = in_model & high_conf
        n_used = int(mask.sum())
        if n_used < cfg.min_keypoints:
            return None

        ids_used = batch.ids[mask]
        indices = np.array([self._index_by_id[int(k)] for k in ids_used])
        image_pts = batch.pixels[mask].reshape(-1, 1, 2).astype(np.float64)
        model_pts = self._model_points[indices].reshape(-1, 1, 3).astype(np.float64)

        reproj_thresh = cfg.reprojection_error_threshold

        # solvePnPGeneric exposes BOTH coplanar IPPE candidates; plain solvePnP
        # only ever hands back one. We keep both for offline analysis, then work
        # with the lowest-reprojection-error candidate. No disambiguation beyond
        # that ordering. (solvePnPRansac is unusable here: it forces EPnP
        # internally, which fails on coplanar subsets — hence manual rejection.)
        try:
            n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                model_pts, image_pts, K, D, flags=self._solver_flag
            )
        except cv2.error as e:
            rospy.logwarn_throttle(
                2.0, f"PnP '{cfg.name}': solvePnPGeneric failed: {e}"
            )
            return None
        if not n_sols:
            return None

        # Mean reprojection error per candidate over the points used in the solve.
        candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for rvec, tvec in zip(rvecs, tvecs):
            proj, _ = cv2.projectPoints(model_pts, rvec, tvec, K, D)
            errs = np.linalg.norm(
                proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1
            )
            candidates.append((float(errs.mean()), rvec, tvec))
        candidates.sort(key=lambda c: c[0])

        # Best candidate drives the published pose + per-point outlier rejection.
        _, rvec, tvec = candidates[0]
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
            return None

        rvec, tvec = self._polish(rvec, tvec, model_pts, image_pts, inliers, K, D)
        if rvec is None:
            return None

        R, _ = cv2.Rodrigues(rvec)
        # Convert candidates to R/t for debug publishing (order preserved).
        cand_poses = [(err, cv2.Rodrigues(rv)[0], tv) for err, rv, tv in candidates]
        return EstimationResult(pose=Pose(R=R, t=tvec), candidates=cand_poses)

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

        if hl.output_axis == "z":
            x_axis, z_axis = normal, direction
            y_axis = np.cross(z_axis, x_axis)
        else:  # "y", validated at config load
            x_axis, y_axis = normal, direction
            z_axis = np.cross(x_axis, y_axis)

        return Pose(R=np.column_stack([x_axis, y_axis, z_axis]), t=pose.t)


class CameraHandler:
    def __init__(
        self,
        name: str,
        camera_frame: str,
        camera_info_topic: str,
        keypoint_topic: str,
        estimators: List[PnPEstimator],
        tf_buffer: tf2_ros.Buffer,
        publisher: rospy.Publisher,
    ):
        self.name = name
        self.camera_frame = camera_frame
        self.estimators = estimators
        self.tf_buffer = tf_buffer
        self.publisher = publisher
        self._calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None

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

        # Per-object debug topics carrying every PnP candidate (both IPPE
        # solutions), for offline bag analysis of the planar two-fold ambiguity.
        # Poses are in the camera optical frame; reproj errors are index-aligned
        # with the PoseArray and sorted ascending (index 0 = published pose).
        self._debug_pub_poses: Dict[str, rospy.Publisher] = {}
        self._debug_pub_errors: Dict[str, rospy.Publisher] = {}
        for est in estimators:
            base = f"keypoint_pose_debug/{est.name}"
            self._debug_pub_poses[est.name] = rospy.Publisher(
                base + "/candidate_poses", PoseArray, queue_size=1
            )
            self._debug_pub_errors[est.name] = rospy.Publisher(
                base + "/candidate_reproj_errors", Float64MultiArray, queue_size=1
            )

        rospy.Subscriber(camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(
            keypoint_topic, KeypointResult, self._keypoint_cb, queue_size=1
        )

    def _info_cb(self, msg: CameraInfo):
        K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.D, dtype=np.float64)
        self._calibration = (K, D)

    def _keypoint_cb(self, msg: KeypointResult):
        if self._calibration is None:
            return
        K, D = self._calibration

        for batch in self._group(msg):
            est = batch.estimator
            result = est.estimate_pose(batch, K, D)
            if result is None:
                rospy.logwarn_throttle(
                    5.0,
                    f"[{self.name}] PnP failed for '{est.name}' "
                    f"({len(batch.ids)} keypoints)",
                )
                continue

            self._publish_candidates_debug(est, result.candidates, msg.header.stamp)
            pose = result.pose

            if est.has_handle_line:
                pose = est.apply_handle_line(pose, batch, K, D)
                if pose is None:
                    rospy.loginfo_throttle(
                        2.0,
                        f"[{self.name}] '{est.name}': handle line missing "
                        f"or degenerate, skipping publish",
                    )
                    continue

            self._publish_outputs(est, pose, msg.header.stamp)

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

    def _publish_candidates_debug(self, est: PnPEstimator, candidates, stamp):
        """Publish all PnP candidates (camera frame) + their reproj errors."""
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.camera_frame
        errors = Float64MultiArray()

        for err, R, t in candidates:
            M = np.eye(4)
            M[:3, :3] = R
            quat = tft.quaternion_from_matrix(M)
            p = GeomPose()
            p.position.x = float(t[0, 0])
            p.position.y = float(t[1, 0])
            p.position.z = float(t[2, 0])
            p.orientation.x = float(quat[0])
            p.orientation.y = float(quat[1])
            p.orientation.z = float(quat[2])
            p.orientation.w = float(quat[3])
            pose_array.poses.append(p)
            errors.data.append(err)

        self._debug_pub_poses[est.name].publish(pose_array)
        self._debug_pub_errors[est.name].publish(errors)

    def _publish_outputs(self, est: PnPEstimator, pose: Pose, stamp):
        M = np.eye(4)
        M[:3, :3] = pose.R
        quat = tft.quaternion_from_matrix(M)

        for output in est.config.outputs:
            child_pos = pose.R @ output.offset.reshape(3, 1) + pose.t
            x, y, z = (
                float(child_pos[0, 0]),
                float(child_pos[1, 0]),
                float(child_pos[2, 0]),
            )

            if z <= 0.0:
                rospy.logdebug(f"Behind-camera: {output.child_frame_id} z={z:.2f}")
                continue

            distance = float(np.linalg.norm(child_pos))
            if distance > est.config.max_distance:
                rospy.logdebug(
                    f"Too far: {output.child_frame_id} "
                    f"d={distance:.2f} > {est.config.max_distance}"
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
                estimators=cam_estimators,
                tf_buffer=self.tf_buffer,
                publisher=self.publisher,
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
