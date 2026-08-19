#!/usr/bin/env python3
"""gate_pose op — 6-DOF gate pose from the keypoints alone.

Runs FusedPlanarPoseEstimator (vitpose_utils SECTION 2) on its keypoint-only
path — the aperture mask is deliberately NOT consumed (no dense refinement,
no mask-IoU validation); `aperture_polygon` only serves the overlay and the
range gate. Publishes the configured frames through the object map TF
server. The prior (last accepted pose, else a canonical face-on guess)
serves the IPPE flip guard and the degenerate-init fallback. Rejected or absent solves publish
nothing: the object map server Kalman-filters, so dropped frames are cheap
and wrong frames are not.
"""

import threading

import cv2
import numpy as np
import rospy
import tf.transformations
import tf2_ros

from utils.vitpose_utils import FusedPlanarPoseEstimator, project_points

_REQUIRED = ("model_points", "aperture_polygon")


def create_op(params, ctx):
    return GatePoseOp(params, ctx)


class GatePoseOp:
    name = "gate_pose"

    def __init__(self, params, ctx):
        self.ctx = ctx
        for key in _REQUIRED:
            if key not in params:
                raise ValueError(f"gate_pose: missing required param '{key}'")

        self.min_keypoints = int(params.get("min_keypoints", 4))
        self.max_distance = float(params.get("max_distance", 30.0))
        self.prior_timeout = float(params.get("prior_timeout", 5.0))
        self.prior_distance = float(params.get("prior_distance", 3.0))
        self.robot_frame = str(params.get("robot_frame", "taluy/base_link"))
        self.orientation_reference_frame = str(
            params.get("orientation_reference_frame", "odom")
        )
        self.orientation_lookup_timeout = float(
            params.get("orientation_lookup_timeout", 0.5)
        )
        self.outputs = [
            dict(
                child_frame=str(out["child_frame"]),
                offset_xyz=np.asarray(
                    out.get("offset_xyz", (0.0, 0.0, 0.0)), dtype=np.float64
                ),
            )
            for out in params.get("outputs", [])
        ]
        if not self.outputs:
            raise ValueError("gate_pose: 'outputs' must list at least one frame")

        # Overlay axes of the first output frame (gate_link), 0.35 m long.
        origin = self.outputs[0]["offset_xyz"]
        self._axes_obj = np.stack(
            [
                origin,
                origin + (0.35, 0.0, 0.0),
                origin + (0.0, 0.35, 0.0),
                origin + (0.0, 0.0, 0.35),
            ]
        )

        plane_normal = params.get("plane_normal", (0.0, 1.0, 0.0))
        self.estimator = FusedPlanarPoseEstimator(
            model_points=params["model_points"],
            boundary_polygon=params["aperture_polygon"],
            plane_normal=plane_normal,
            soft_slide_segments=params.get("slide_segments"),
            **(params.get("refine") or {}),
        )
        self.num_kps = len(self.estimator.model_points)

        # Canonical face-on prior: object upright, plane normal toward the
        # camera, boundary centroid on the optical axis at prior_distance.
        r0 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
        centroid = self.estimator.boundary_polygon.mean(axis=0)
        self._canonical_prior = (
            cv2.Rodrigues(r0)[0].ravel(),
            np.array([0.0, 0.0, self.prior_distance]) - r0 @ centroid,
        )
        self._last_accepted = None  # (rvec, tvec, stamp)
        self._viz_lock = threading.Lock()
        self._viz = None  # (polygon (M,2), accepted, status_text, axes (4,2))

        rospy.loginfo(
            f"gate_pose op ready for '{ctx.object_name}': "
            f"{self.num_kps} model points, outputs "
            f"{[o['child_frame'] for o in self.outputs]}"
        )

    # ------------------------------------------------------------- helpers

    def _prior(self, stamp):
        if self._last_accepted is not None:
            rvec, tvec, t_last = self._last_accepted
            if (stamp - t_last).to_sec() <= self.prior_timeout:
                return rvec, tvec
        return self._canonical_prior

    def _set_viz(self, polygon, accepted, text, axes=None):
        with self._viz_lock:
            self._viz = (polygon, accepted, text, axes)

    def _abstain(self, reason):
        self._set_viz(None, False, f"gate_pose: {reason}")
        rospy.logdebug_throttle(2.0, f"gate_pose abstains: {reason}")

    @staticmethod
    def _transform_matrix(transform):
        rotation = transform.transform.rotation
        matrix = tf.transformations.quaternion_matrix(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        translation = transform.transform.translation
        matrix[:3, 3] = (translation.x, translation.y, translation.z)
        return matrix

    def _fix_orientation_towards_robot(self, R, gate_xyz, stamp):
        """Make local -Y point toward the robot in the horizontal plane.

        Planar PnP can return the gate with its normal reversed. Resolve that
        ambiguity before publishing gate_link, using the same dot-product
        convention formerly applied by pinger_teknofest_trajectory_publisher.
        """
        try:
            reference_from_camera = self.ctx.tf_buffer.lookup_transform(
                self.orientation_reference_frame,
                self.ctx.camera_frame,
                stamp,
                rospy.Duration(self.orientation_lookup_timeout),
            )
            reference_from_robot = self.ctx.tf_buffer.lookup_transform(
                self.orientation_reference_frame,
                self.robot_frame,
                stamp,
                rospy.Duration(self.orientation_lookup_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                5.0,
                "gate_pose: cannot fix gate orientation in %s: %s",
                self.orientation_reference_frame,
                exc,
            )
            return None

        reference_from_camera_matrix = self._transform_matrix(
            reference_from_camera
        )
        gate_position = (
            reference_from_camera_matrix[:3, :3] @ gate_xyz
            + reference_from_camera_matrix[:3, 3]
        )
        gate_rotation = reference_from_camera_matrix[:3, :3] @ R
        robot_position = self._transform_matrix(reference_from_robot)[:3, 3]

        gate_to_robot = robot_position[:2] - gate_position[:2]
        closer_direction = -gate_rotation[:2, 1]
        if float(np.dot(closer_direction, gate_to_robot)) < 0.0:
            # Local-Z half turn flips local X/Y while preserving upright Z.
            R = R @ np.diag((-1.0, -1.0, 1.0))
            rospy.logdebug_throttle(
                2.0,
                "gate_pose: fixed gate_link orientation by 180 degrees "
                "around local Z",
            )
        return R

    # ------------------------------------------------------------- process

    def process(self, frame):
        calib = self.ctx.calibration()
        if calib is None:
            rospy.logwarn_throttle(10.0, "gate_pose: no camera calibration yet")
            return
        K, D = calib

        # Rebuild dense arrays by keypoint id; missing ids scored 0.
        kps = np.zeros((self.num_kps, 2), dtype=np.float64)
        scores = np.zeros(self.num_kps, dtype=np.float64)
        for kp_id, px, s in zip(frame.ids, frame.pixels, frame.scores):
            if 0 <= kp_id < self.num_kps:
                kps[kp_id] = px
                scores[kp_id] = s
        n_confident = int((scores >= self.estimator.cfg["score_gate"]).sum())
        if n_confident < self.min_keypoints:
            return self._abstain(f"{n_confident} confident kps < {self.min_keypoints}")

        result = self.estimator.estimate(
            kps, scores, None, K, D, prior=self._prior(frame.stamp)
        )
        if result is None:
            return self._abstain("no PnP solution")
        rvec, tvec = result["rvec"], result["tvec"]
        R, _ = cv2.Rodrigues(rvec)

        # --- validation gates -------------------------------------------
        centroid_cam = R @ self.estimator.boundary_polygon.mean(axis=0) + tvec
        distance = float(np.linalg.norm(centroid_cam))
        polygon = project_points(self.estimator.boundary_polygon, rvec, tvec, K, D)
        axes = project_points(self._axes_obj, rvec, tvec, K, D)
        status = f"kp={n_confident} out={result.get('n_outliers', 0)} d={distance:.1f}m"

        if centroid_cam[2] <= 0:
            return self._abstain(f"behind camera ({status})")
        if distance > self.max_distance:
            self._set_viz(polygon, False, f"gate_pose: too far ({status})", axes)
            return

        # Resolve the planar-PnP normal ambiguity before gate_link enters the
        # object map; all downstream users then see one orientation convention.
        gate_origin = R @ self.outputs[0]["offset_xyz"] + tvec
        output_R = self._fix_orientation_towards_robot(R, gate_origin, frame.stamp)
        if output_R is None:
            return self._abstain("orientation reference unavailable")

        # The overlay axes represent the orientation that is actually
        # published, while the polygon remains the raw PnP reprojection.
        output_rvec = cv2.Rodrigues(output_R)[0].ravel()
        axes = project_points(self._axes_obj, output_rvec, tvec, K, D)

        # --- accept: publish outputs ------------------------------------
        # Keep the raw PnP pose as its next-frame prior. The corrected output
        # rotation does not preserve model-point correspondences.
        self._last_accepted = (rvec, tvec, frame.stamp)
        quat = tf.transformations.quaternion_from_matrix(
            np.block([[output_R, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]])
        )
        for out in self.outputs:
            xyz = output_R @ out["offset_xyz"] + tvec
            self.ctx.publish_tf(
                out["child_frame"], xyz, frame.stamp, rotation_quat=quat
            )
        self._set_viz(polygon, True, f"gate_pose: OK {status}", axes)

    # ------------------------------------------------------------- overlay

    def draw(self, image_bgr, frame):
        with self._viz_lock:
            viz = self._viz
        if viz is None:
            return
        polygon, accepted, text, axes = viz
        color = (0, 220, 0) if accepted else (0, 0, 235)
        if polygon is not None:
            cv2.polylines(
                image_bgr,
                [np.round(polygon).astype(np.int32)],
                True,
                color,
                2,
            )
        if axes is not None:
            # gate_link axes: X red, Y green (aperture normal), Z blue.
            pts = np.round(axes).astype(np.int32)
            for tip, axis_color in zip(
                pts[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))
            ):
                cv2.line(image_bgr, tuple(pts[0]), tuple(tip), axis_color, 2)
        cv2.putText(
            image_bgr,
            text,
            (10, image_bgr.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
