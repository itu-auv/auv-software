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
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo

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
class StateKeypoint:
    """A keypoint outside the PnP set whose direction in the model's YZ plane
    overrides the rotation about model X (e.g. valve handle angle)."""

    keypoint_id: int
    output_axis: str = "z"
    confidence_threshold: float = 0.5
    parallel_eps_rad: float = 0.087


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
    state_keypoint: Optional[StateKeypoint] = None


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
    def has_state_keypoint(self) -> bool:
        return self.config.state_keypoint is not None

    def estimate_pose(
        self, batch: KeypointBatch, K: np.ndarray, D: np.ndarray
    ) -> Optional[Pose]:
        cfg = self.config
        in_model = np.isin(batch.ids, self._pose_ids)
        high_conf = batch.confidences >= cfg.confidence_threshold
        mask = in_model & high_conf
        if int(mask.sum()) < cfg.min_keypoints:
            return None

        ids_used = batch.ids[mask]
        indices = np.array([self._index_by_id[int(k)] for k in ids_used])
        image_pts = batch.pixels[mask].reshape(-1, 1, 2).astype(np.float64)
        model_pts = self._model_points[indices].reshape(-1, 1, 3).astype(np.float64)

        # solvePnPRansac always uses EPnP internally regardless of the
        # `flags` argument, and EPnP fails on coplanar subsets — so we do
        # manual reprojection-based outlier rejection instead.
        ok, rvec, tvec = cv2.solvePnP(
            model_pts, image_pts, K, D, flags=self._solver_flag
        )
        if not ok:
            return None

        proj, _ = cv2.projectPoints(model_pts, rvec, tvec, K, D)
        errors = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
        inliers = errors <= cfg.reprojection_error_threshold
        n_inliers = int(inliers.sum())

        rospy.loginfo_throttle(
            2.0,
            f"PnP '{cfg.name}': {n_inliers}/{len(model_pts)} inliers, "
            f"max reproj err: {errors.max():.2f}px",
        )

        if n_inliers < cfg.min_keypoints:
            return None

        rvec, tvec = self._polish(rvec, tvec, model_pts, image_pts, inliers, K, D)
        if rvec is None:
            return None

        R, _ = cv2.Rodrigues(rvec)
        return Pose(R=R, t=tvec)

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

    def apply_state_keypoint(
        self, pose: Pose, batch: KeypointBatch, K: np.ndarray, D: np.ndarray
    ) -> Optional[Pose]:
        """Replace rotation about model X with the in-plane direction toward
        the state keypoint. Returns None when missing/low-conf/degenerate."""
        sk = self.config.state_keypoint
        matches = np.where(batch.ids == sk.keypoint_id)[0]
        if len(matches) == 0:
            return None
        i = matches[0]
        if batch.confidences[i] < sk.confidence_threshold:
            return None

        pixel = batch.pixels[i].reshape(1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pixel, K, D)
        ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])

        normal = pose.R[:, 0]
        center = pose.t.flatten()

        # |normal · ray| / |ray| = sin(angle between ray and plane); skip
        # when the valve is seen edge-on (denom ≈ 0 → ray nearly in-plane).
        denom = float(normal @ ray)
        ray_norm = float(np.linalg.norm(ray))
        if abs(denom) < np.sin(sk.parallel_eps_rad) * ray_norm:
            return None

        t_param = float(normal @ center) / denom
        if t_param <= 0.0:
            return None

        kp_camera = t_param * ray
        raw = kp_camera - center
        in_plane = raw - (normal @ raw) * normal
        norm = float(np.linalg.norm(in_plane))
        if norm < 1e-6:
            return None
        direction = in_plane / norm

        if sk.output_axis == "z":
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
            pose = est.estimate_pose(batch, K, D)
            if pose is None:
                rospy.logwarn_throttle(
                    5.0,
                    f"[{self.name}] PnP failed for '{est.name}' "
                    f"({len(batch.ids)} keypoints)",
                )
                continue

            if est.has_state_keypoint:
                pose = est.apply_state_keypoint(pose, batch, K, D)
                if pose is None:
                    rospy.loginfo_throttle(
                        2.0,
                        f"[{self.name}] '{est.name}': state keypoint missing "
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

        state_kp = None
        if "state_keypoint" in obj_cfg:
            sk_cfg = obj_cfg["state_keypoint"]
            output_axis = sk_cfg.get("output_axis", "z")
            if output_axis not in ("y", "z"):
                raise ValueError(
                    f"state_keypoint.output_axis for '{obj_cfg['name']}' "
                    f"must be 'y' or 'z', got '{output_axis}'"
                )
            sk_id = int(sk_cfg["keypoint_id"])
            if sk_id in pose_keypoints:
                raise ValueError(
                    f"state_keypoint.keypoint_id {sk_id} for "
                    f"'{obj_cfg['name']}' clashes with a pose model id"
                )
            state_kp = StateKeypoint(
                keypoint_id=sk_id,
                output_axis=output_axis,
                confidence_threshold=float(sk_cfg.get("confidence_threshold", 0.5)),
                parallel_eps_rad=float(sk_cfg.get("parallel_eps_rad", 0.087)),
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
                state_keypoint=state_kp,
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
