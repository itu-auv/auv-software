#!/usr/bin/env python3

"""
Keypoint-based PnP pose estimation node.

Subscribes to KeypointResult messages (from sim_keypoint_node or a real
keypoint detector), solves PnP to recover 6-DOF object poses, and publishes
TransformStamped messages on object_transform_updates — the same topic the
camera_detection_node uses, so the downstream object map TF server and SMACH
layer see no difference.
"""

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

# Add scripts directory to path for utils imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils.detection_utils import SHAPE_FACTORIES, transform_to_odom_and_publish


# ---------------------------------------------------------------------------
# Object model
# ---------------------------------------------------------------------------


@dataclass
class OutputFrame:
    child_frame_id: str
    offset: np.ndarray  # (3,) local offset from model origin


@dataclass
class OrientationRefinement:
    """Override the PnP-derived rotation around the model X axis using a
    2D keypoint that lies in the model's YZ plane.

    Use case (valve): the bolt ring fixes the face plane, but the rotation
    around the normal that PnP returns is anchored to bolt #1 — irrelevant
    for gripper alignment. The handle tip keypoint, projected onto the
    face plane, gives the direction the gripper actually wants to align to.

    Algorithm:
      1. Back-project the refinement keypoint pixel into a camera-frame ray.
      2. Intersect that ray with the face plane (passes through the
         PnP-derived center, normal = R[:, 0] in camera frame).
      3. Project the (center → intersection) vector onto the face plane.
      4. Build a new rotation: X = face normal, output_axis = that vector,
         third axis from the cross product.

    Returns None (and the caller skips publishing) when the keypoint is
    missing, low-confidence, or the geometry is degenerate (view ray
    nearly parallel to the face, or handle tip lands on the center).
    """

    keypoint_id: int
    output_axis: str = "z"  # "y" or "z"
    confidence_threshold: float = 0.5
    parallel_eps_rad: float = 0.087  # ~5°: skip if view ray near edge-on


@dataclass
class KeypointObject:
    name: str
    object_names: List[str]
    cameras: List[str]
    id_start: int
    model_points: np.ndarray  # (N, 3) 3D points in model frame
    outputs: List[OutputFrame]
    min_keypoints: int
    solver: str
    max_distance: float
    reprojection_error_threshold: float
    confidence_threshold: float
    refinement: Optional[OrientationRefinement] = None

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

    def solve(
        self,
        keypoint_ids: np.ndarray,
        image_points: np.ndarray,
        confidences: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run PnP on matched keypoints. Returns (rvec, tvec) or None."""

        # Filter by confidence and valid model index together so the
        # min_keypoints gate reflects actually usable keypoints.
        mask = confidences >= self.confidence_threshold
        keypoint_ids = keypoint_ids[mask]
        image_points = image_points[mask]

        indices = keypoint_ids - self.id_start
        valid = (indices >= 0) & (indices < len(self.model_points))
        if valid.sum() < self.min_keypoints:
            return None

        indices = indices[valid]
        image_pts = image_points[valid].reshape(-1, 1, 2).astype(np.float64)
        model_pts = self.model_points[indices].reshape(-1, 1, 3).astype(np.float64)

        solver_flag = self._SOLVER_FLAGS.get(self.solver, cv2.SOLVEPNP_ITERATIVE)

        # solvePnPRansac always uses EPnP internally for its RANSAC
        # kernel (solvepnp.cpp:256), ignoring the requested solver.
        # EPnP produces garbage poses for coplanar point subsets, so
        # every RANSAC iteration finds zero inliers and returns false.
        # Instead: solve with all points, reject outliers by reprojection
        # error, re-solve with the inlier set.
        success, rvec, tvec = cv2.solvePnP(
            model_pts,
            image_pts,
            camera_matrix,
            dist_coeffs,
            flags=solver_flag,
        )
        if not success:
            return None

        # Outlier rejection: project model points, drop those with
        # reprojection error above threshold, re-solve if any were dropped.
        proj, _ = cv2.projectPoints(model_pts, rvec, tvec, camera_matrix, dist_coeffs)
        errors = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
        inlier_mask = errors <= self.reprojection_error_threshold
        n_inliers = inlier_mask.sum()

        rospy.loginfo_throttle(
            2.0,
            f"PnP '{self.name}': {n_inliers}/{len(model_pts)} inliers, "
            f"max reproj err: {errors.max():.2f}px",
        )

        if n_inliers < self.min_keypoints:
            return None

        if n_inliers < len(model_pts):
            success, rvec, tvec = cv2.solvePnP(
                model_pts[inlier_mask],
                image_pts[inlier_mask],
                camera_matrix,
                dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=solver_flag,
            )
            if not success:
                return None

        return rvec, tvec

    def apply_orientation_refinement(
        self,
        R: np.ndarray,
        tvec: np.ndarray,
        keypoint_ids: np.ndarray,
        image_points: np.ndarray,
        confidences: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Replace the rotation around model X with one anchored to an
        in-plane refinement keypoint (e.g. valve handle tip → +Z axis).

        Returns the new (3, 3) rotation matrix, or None if the refinement
        keypoint is missing / low-confidence / geometrically degenerate.
        Callers should skip publishing the frame on None.
        """
        ref = self.refinement
        if ref is None:
            return R

        matches = np.where(keypoint_ids == ref.keypoint_id)[0]
        if len(matches) == 0:
            return None
        i = matches[0]
        if confidences[i] < ref.confidence_threshold:
            return None

        # Back-project pixel → camera-frame ray (cv2.undistortPoints with no
        # P returns normalized image coordinates, equivalent to K^-1·[u,v,1]).
        pixel = np.array([[image_points[i]]], dtype=np.float64)  # (1, 1, 2)
        undist = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
        d = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])

        # Plane: passes through tvec, normal is R[:, 0] (= valve face normal
        # in camera frame, since the model's X axis is the valve normal).
        n = R[:, 0]
        c = tvec.flatten()

        # Edge-on guard. n is unit; |n·d|/|d| = sin(angle between ray and
        # plane). Skip when that angle is below parallel_eps_rad.
        denom = float(n @ d)
        ray_norm = float(np.linalg.norm(d))
        if abs(denom) < np.sin(ref.parallel_eps_rad) * ray_norm:
            return None

        t = float(n @ c) / denom
        if t <= 0.0:
            return None  # intersection behind camera

        handle_camera = t * d

        # Direction from face center to handle tip, projected onto the
        # face plane (already in-plane in theory, but project for numerics).
        raw = handle_camera - c
        in_plane = raw - (n @ raw) * n
        norm = float(np.linalg.norm(in_plane))
        if norm < 1e-6:
            return None  # handle on top of center → angle ill-defined
        h = in_plane / norm

        # Build new rotation: X stays as the face normal, the chosen
        # output axis points along h, third axis from cross product.
        if ref.output_axis == "z":
            x_axis, z_axis = n, h
            y_axis = np.cross(z_axis, x_axis)
        elif ref.output_axis == "y":
            x_axis, y_axis = n, h
            z_axis = np.cross(x_axis, y_axis)
        else:
            return None  # validated at config load, defensive

        return np.column_stack([x_axis, y_axis, z_axis])


# ---------------------------------------------------------------------------
# Per-camera handler
# ---------------------------------------------------------------------------


class KeypointPoseCamera:
    def __init__(
        self,
        name: str,
        camera_frame: str,
        camera_info_topic: str,
        keypoint_topic: str,
        objects: List[KeypointObject],
        tf_buffer: tf2_ros.Buffer,
        publisher: rospy.Publisher,
    ):
        self.name = name
        self.camera_frame = camera_frame
        self.objects = objects
        self.tf_buffer = tf_buffer
        self.publisher = publisher

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        # Build lookup: keypoint object name -> KeypointObject
        # Multiple object_names can map to the same KeypointObject
        self.object_name_to_obj: Dict[str, KeypointObject] = {}
        for obj in objects:
            for alias in obj.object_names:
                self.object_name_to_obj[alias] = obj

        rospy.Subscriber(camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(
            keypoint_topic, KeypointResult, self._keypoint_cb, queue_size=1
        )

    def _info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D, dtype=np.float64)

    def _keypoint_cb(self, msg: KeypointResult):
        if self.camera_matrix is None:
            return

        stamp = msg.header.stamp

        # Group keypoints by KeypointObject (not by kp.object string).
        # Multiple kp.object names can map to the same KeypointObject for
        # joint PnP solving (e.g. several sub-parts of a single rigid body).
        groups: Dict[
            str,
            Tuple[KeypointObject, List[int], List[Tuple[float, float]], List[float]],
        ] = {}
        for kp in msg.keypoints:
            obj = self.object_name_to_obj.get(kp.object)
            if obj is None:
                continue
            if obj.name not in groups:
                groups[obj.name] = (obj, [], [], [])
            _, ids, pts, confs = groups[obj.name]
            ids.append(kp.id)
            pts.append((kp.x, kp.y))
            confs.append(kp.confidence)

        # Solve PnP per object
        for obj_name, (obj, ids, pts, confs) in groups.items():
            ids_arr = np.array(ids, dtype=np.int32)
            pts_arr = np.array(pts, dtype=np.float64)
            confs_arr = np.array(confs, dtype=np.float64)

            result = obj.solve(
                ids_arr,
                pts_arr,
                confs_arr,
                self.camera_matrix,
                self.dist_coeffs,
            )

            if result is None:
                rospy.logwarn_throttle(
                    5.0,
                    f"[{self.name}] PnP failed for '{obj_name}' "
                    f"({len(ids)} keypoints)",
                )
                continue

            rvec, tvec = result
            R, _ = cv2.Rodrigues(rvec)

            # Optional: replace the rotation around the model X axis using
            # an in-plane keypoint (e.g. valve handle tip). When configured
            # but the refinement keypoint is missing/low-conf/degenerate,
            # apply_orientation_refinement returns None and we skip the
            # whole publish for this object — better than publishing a
            # frame whose yaw is meaningless for the gripper.
            if obj.refinement is not None:
                R_refined = obj.apply_orientation_refinement(
                    R,
                    tvec,
                    ids_arr,
                    pts_arr,
                    confs_arr,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
                if R_refined is None:
                    rospy.loginfo_throttle(
                        2.0,
                        f"[{self.name}] '{obj_name}': refinement keypoint "
                        f"id={obj.refinement.keypoint_id} missing or "
                        f"degenerate, skipping publish",
                    )
                    continue
                R = R_refined

            # Quaternion of the model frame, expressed in the camera frame.
            # The output offset is pure translation (see OutputFrame), so
            # every output frame inherits this same orientation.
            M = np.eye(4)
            M[:3, :3] = R
            quat_camera = tft.quaternion_from_matrix(M)  # (x, y, z, w)

            # For each output frame, compose the model-origin pose with the
            # output's local offset to get the child position in camera frame.
            for output in obj.outputs:
                child_pos_camera = R @ output.offset.reshape(3, 1) + tvec

                if child_pos_camera[2, 0] <= 0.0:
                    rospy.logdebug(
                        f"Behind-camera: {output.child_frame_id} "
                        f"z={child_pos_camera[2, 0]:.2f}"
                    )
                    continue

                distance = np.linalg.norm(child_pos_camera)
                if distance > obj.max_distance:
                    rospy.logdebug(
                        f"Too far: {output.child_frame_id} "
                        f"distance={distance:.2f} > {obj.max_distance}"
                    )
                    continue

                transform_to_odom_and_publish(
                    self.camera_frame,
                    output.child_frame_id,
                    child_pos_camera[0, 0],
                    child_pos_camera[1, 0],
                    child_pos_camera[2, 0],
                    stamp,
                    self.tf_buffer,
                    self.publisher,
                    rotation_quat=quat_camera,
                )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _build_model_points(model_spec) -> np.ndarray:
    """Build Nx3 model points array from YAML model spec (list of shape parts)."""
    all_points = []
    for part in model_spec:
        sub_points = SHAPE_FACTORIES[part["shape"]](part["args"])
        sub_offset = np.array(part.get("offset", [0, 0, 0]), dtype=np.float64)
        all_points.extend(p + sub_offset for p in sub_points)
    return np.array(all_points, dtype=np.float64)


def load_config(config_path: str) -> Tuple[List[KeypointObject], Dict]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    camera_configs = config["cameras"]

    objects = []
    for obj_cfg in config["objects"]:
        model_points = _build_model_points(obj_cfg["model"])
        outputs = [
            OutputFrame(
                child_frame_id=out["child_frame_id"],
                offset=np.array(out["offset"], dtype=np.float64),
            )
            for out in obj_cfg["outputs"]
        ]
        # object_names: list of keypoint object names that belong to this
        # PnP object. Falls back to [name] if not specified.
        object_names = obj_cfg.get("object_names", [obj_cfg["name"]])

        refinement = None
        if "orientation_refinement" in obj_cfg:
            ref_cfg = obj_cfg["orientation_refinement"]
            output_axis = ref_cfg.get("output_axis", "z")
            if output_axis not in ("y", "z"):
                raise ValueError(
                    f"orientation_refinement.output_axis for "
                    f"'{obj_cfg['name']}' must be 'y' or 'z', got "
                    f"'{output_axis}'"
                )
            refinement = OrientationRefinement(
                keypoint_id=int(ref_cfg["keypoint_id"]),
                output_axis=output_axis,
                confidence_threshold=float(ref_cfg.get("confidence_threshold", 0.5)),
                parallel_eps_rad=float(ref_cfg.get("parallel_eps_rad", 0.087)),
            )

        objects.append(
            KeypointObject(
                name=obj_cfg["name"],
                object_names=object_names,
                cameras=obj_cfg["cameras"],
                id_start=obj_cfg["id_start"],
                model_points=model_points,
                outputs=outputs,
                min_keypoints=obj_cfg.get("min_keypoints", 6),
                solver=obj_cfg.get("solver", "iterative"),
                max_distance=obj_cfg.get("max_distance", 30.0),
                reprojection_error_threshold=obj_cfg.get(
                    "reprojection_error_threshold", 5.0
                ),
                confidence_threshold=obj_cfg.get("confidence_threshold", 0.3),
                refinement=refinement,
            )
        )

    return objects, camera_configs


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class KeypointPoseNode:
    def __init__(self):
        rospy.init_node("keypoint_pose_node")

        default_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "config",
            "keypoint_objects.yaml",
        )
        config_path = rospy.get_param("~config_file", default_config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publisher = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )

        objects, camera_configs = load_config(config_path)

        # Build per-camera object lists
        objects_by_camera: Dict[str, List[KeypointObject]] = {
            cam: [] for cam in camera_configs
        }
        for obj in objects:
            for cam_name in obj.cameras:
                if cam_name in objects_by_camera:
                    objects_by_camera[cam_name].append(obj)

        self.cameras: Dict[str, KeypointPoseCamera] = {}
        for cam_name, cam_cfg in camera_configs.items():
            cam_objects = objects_by_camera.get(cam_name, [])
            if not cam_objects:
                continue
            self.cameras[cam_name] = KeypointPoseCamera(
                name=cam_name,
                camera_frame=cam_cfg["frame"],
                camera_info_topic=cam_cfg["camera_info_topic"],
                keypoint_topic=cam_cfg["keypoint_topic"],
                objects=cam_objects,
                tf_buffer=self.tf_buffer,
                publisher=self.publisher,
            )

        total_points = sum(len(o.model_points) for o in objects)
        total_outputs = sum(len(o.outputs) for o in objects)
        rospy.loginfo(
            f"KeypointPoseNode started — {len(objects)} objects, "
            f"{total_points} model points, {total_outputs} output frames, "
            f"{len(self.cameras)} cameras"
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = KeypointPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
