#!/usr/bin/env python3
"""sim_vitpose_node: ground-truth stand-in for vitpose_detection_node.

Same node name, topics, services and semantics — the shell is shared code
(VitposeNodeBase) — so SMACH written against the real pipeline runs
unmodified; yildiz.launch sim:=true does exactly that. Interface-shaped
config comes from the same config/vitpose/<object>.yaml files; only the
ground truth (Gazebo model, keypoint positions, mask polygons, face
bindings) comes from sim_vitpose_objects.yaml. No checkpoint or torch is
ever touched.

Fidelity: all K keypoints published, confidence 1.0 visible / 0.0 not (out
of frame, behind camera, or `face:`-bound to a back-facing mask — the
averted-letter case); masks exact 0/255, back-face culled, empty if any
vertex is behind the camera; nothing visible = no VitposeResult while the
bbox topic still heartbeats (also through the camera_info/TF/gazebo
warm-up). NOT emulated: keypoint noise, image-space L/R flips, objectness
misses (incl. the 4:3-crop blindness on 16:9 cameras — sim sees the full
frame), crop tracking. Sim is the ideal detector; test degradation on real
footage.

Projection machinery modeled after sim_bbox_node.py, including the
stabilized front-camera optical frame.
"""

import os
import sys
import threading

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import rospkg
import tf2_ros
import tf.transformations as tft
import yaml

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

from auv_msgs.msg import Keypoint, VitposeResult
from auv_msgs.srv import SetStringResponse

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.vitpose_utils import (  # noqa: E402
    RateGate,
    is_detect_only,
    load_object_config,
    runtime_detect_rate,
    validate_detect_only,
)
from vitpose_detection_node import (  # noqa: E402
    VitposeNodeBase,
    detection_array,
)


# ─────────────────────────────────────────────── transform helpers


def pose_to_matrix(pose: Pose) -> np.ndarray:
    p, o = pose.position, pose.orientation
    return tft.translation_matrix((p.x, p.y, p.z)) @ tft.quaternion_matrix(
        (o.x, o.y, o.z, o.w)
    )


def transform_to_matrix(transform: Transform) -> np.ndarray:
    t, r = transform.translation, transform.rotation
    return tft.translation_matrix((t.x, t.y, t.z)) @ tft.quaternion_matrix(
        (r.x, r.y, r.z, r.w)
    )


def invert_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


class GazeboInterface:
    """Stamp-matched /gazebo/model_states snapshots (as in sim_bbox_node)."""

    _BUFFER_SIZE = 100

    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        self.model_matrices: Dict[str, np.ndarray] = {}
        self._pose_buffer: deque = deque(maxlen=self._BUFFER_SIZE)
        self._state_lock = threading.Lock()
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1
        )

    def _model_states_cb(self, msg: ModelStates):
        matrices = {
            name: pose_to_matrix(pose) for name, pose in zip(msg.name, msg.pose)
        }
        with self._state_lock:
            self._pose_buffer.append((rospy.Time.now(), matrices))
            self.model_matrices = matrices

    def snapshot_at_time(self, stamp: rospy.Time) -> Dict[str, np.ndarray]:
        with self._state_lock:
            if not self._pose_buffer:
                return self.model_matrices
            _, matrices = min(
                tuple(self._pose_buffer),
                key=lambda entry: abs((entry[0] - stamp).to_sec()),
            )
        return matrices


# ─────────────────────────────────────────────── sim geometry config


@dataclass
class SimMask:
    name: str
    polygon_h: np.ndarray  # (N, 4) homogeneous object-frame corners
    cull_backface: bool
    normal: np.ndarray  # (3,) outward unit normal (from polygon winding)
    centroid_h: np.ndarray  # (4,) polygon centroid


@dataclass
class SimKeypoint:
    position_h: np.ndarray  # (4,)
    face: Optional[str]  # mask name whose visibility gates this keypoint


@dataclass
class SimGeometry:
    gazebo_model: str
    camera: str
    keypoints: List[SimKeypoint]
    masks: Dict[str, SimMask]
    bbox_points_h: np.ndarray  # (N, 4) extent points for detect-only boxes


def _homogeneous(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.column_stack((pts, np.ones(len(pts))))


def _build_mask(name, spec) -> SimMask:
    polygon = np.asarray(spec["polygon"], dtype=np.float64)
    if len(polygon) < 3:
        raise ValueError(f"mask '{name}': polygon needs >= 3 vertices")
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError(f"mask '{name}': degenerate polygon")
    return SimMask(
        name=name,
        polygon_h=_homogeneous(polygon),
        cull_backface=bool(spec.get("cull_backface", False)),
        normal=normal / norm,
        centroid_h=np.append(polygon.mean(axis=0), 1.0),
    )


def sim_config_path() -> str:
    return os.path.join(
        rospkg.RosPack().get_path("auv_vision"), "config", "sim_vitpose_objects.yaml"
    )


def load_sim_config(path: str, ns: str) -> Tuple[Dict, Dict[str, SimGeometry]]:
    """(cameras, {object name: SimGeometry}) with {ns} substituted."""
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)

    cameras = {
        name: {key: value.replace("{ns}", ns) for key, value in cfg.items()}
        for name, cfg in config["cameras"].items()
    }

    objects: Dict[str, SimGeometry] = {}
    for name, spec in config["objects"].items():
        camera = spec["camera"]
        if camera not in cameras:
            raise ValueError(f"sim object '{name}': unknown camera '{camera}'")
        masks = {
            mask_name: _build_mask(mask_name, mask_spec)
            for mask_name, mask_spec in (spec.get("masks") or {}).items()
        }
        keypoints = []
        for entry in spec.get("keypoints") or []:
            if isinstance(entry, dict):
                face = entry.get("face")
                if face is not None and face not in masks:
                    raise ValueError(
                        f"sim object '{name}': keypoint face '{face}' has no mask"
                    )
                keypoints.append(SimKeypoint(_homogeneous(entry["position"])[0], face))
            else:
                keypoints.append(SimKeypoint(_homogeneous(entry)[0], None))

        bbox_points = spec.get("bbox_points")
        if bbox_points is not None:
            bbox_points_h = _homogeneous(bbox_points)
        else:
            # Default extent = every keypoint plus every mask corner.
            parts = [kp.position_h for kp in keypoints]
            parts += [row for mask in masks.values() for row in mask.polygon_h]
            if not parts:
                raise ValueError(
                    f"sim object '{name}': needs keypoints, masks or bbox_points"
                )
            bbox_points_h = np.stack(parts)

        objects[name] = SimGeometry(
            gazebo_model=spec["gazebo_model"],
            camera=camera,
            keypoints=keypoints,
            masks=masks,
            bbox_points_h=bbox_points_h,
        )
    return cameras, objects


# ─────────────────────────────────────────────── active pipeline (swappable)


class _SimPipeline:
    """The real _Pipeline's twin: one object config + its sim geometry.
    Runs the same detect-only validation — sim may be the only thing
    exercising a config before pool day."""

    def __init__(
        self, config, sim_cameras, sim_objects, image_cb, info_cb, load_pose=True
    ):
        self.config = config
        self.object_name = config["object"]
        detection_cfg = config["detection"]
        static_detect_only = is_detect_only(config)
        self.detect_only = static_detect_only or not load_pose

        if self.object_name not in sim_objects:
            raise ValueError(
                f"'{self.object_name}' has no entry in sim_vitpose_objects.yaml "
                f"(available: {sorted(sim_objects)})"
            )
        self.geometry = sim_objects[self.object_name]
        camera_cfg = sim_cameras[self.geometry.camera]
        self.optical_frame = camera_cfg["optical_frame"]

        provider_cfg = dict(detection_cfg.get("bbox_provider") or {})
        rate = detection_cfg.get("rate")
        if static_detect_only:
            validate_detect_only(self.object_name, provider_cfg)
        elif not load_pose:
            # Mode 'detect' on a pose config: mirror the real node's output
            # rate so SMACH sees representative heartbeats.
            rate = runtime_detect_rate(detection_cfg)
        if self.detect_only:
            self.keypoint_names: List[str] = []
            self.mask_classes: List[str] = []
            self.mask_threshold = 0.5
        else:
            model_cfg = config["model"]
            self.keypoint_names = list(model_cfg.get("keypoint_names") or [])
            self.mask_classes = list(model_cfg.get("mask_classes") or [])
            self.mask_threshold = float(model_cfg.get("mask_threshold") or 0.5)
            # The real node checks names against checkpoint K/C; here the sim
            # geometry plays the checkpoint's role.
            if len(self.keypoint_names) != len(self.geometry.keypoints):
                raise ValueError(
                    f"{self.object_name}: {len(self.keypoint_names)} "
                    f"keypoint_names but sim geometry has "
                    f"K={len(self.geometry.keypoints)}"
                )
            if set(self.mask_classes) != set(self.geometry.masks):
                raise ValueError(
                    f"{self.object_name}: mask_classes {self.mask_classes} != "
                    f"sim geometry masks {sorted(self.geometry.masks)}"
                )

        self.class_id = int(provider_cfg.get("class_id", 0))
        self._rate = RateGate(rate)

        # Per-camera state filled by callbacks.
        self.intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.image_w = 0
        self.image_h = 0
        self.base_to_camera: Optional[np.ndarray] = None

        self.result_pub = (
            None
            if self.detect_only
            else rospy.Publisher(
                detection_cfg["result_topic"], VitposeResult, queue_size=1
            )
        )
        bbox_topic = provider_cfg.get("publish_topic")
        self.bbox_pub = (
            rospy.Publisher(bbox_topic, Detection2DArray, queue_size=1)
            if bbox_topic
            else None
        )
        self.info_sub = rospy.Subscriber(
            camera_cfg["camera_info_topic"],
            CameraInfo,
            lambda msg, pipeline=self: info_cb(msg, pipeline),
            queue_size=1,
        )
        self.image_sub = rospy.Subscriber(
            detection_cfg["image_topic"],
            Image,
            lambda msg, pipeline=self: image_cb(msg, pipeline),
            queue_size=1,
            buff_size=2**24,
        )

    def due(self, header) -> bool:
        """Rate gate — the same RateGate the real node's pipeline uses."""
        return self._rate.due(header.stamp.to_sec() or rospy.get_time())

    def shutdown(self):
        self.image_sub.unregister()
        self.info_sub.unregister()
        if self.result_pub is not None:
            self.result_pub.unregister()
        if self.bbox_pub is not None:
            self.bbox_pub.unregister()


# ─────────────────────────────────────────────── node


class SimVitposeNode(VitposeNodeBase):
    log_name = "sim_vitpose_node"

    def __init__(self):
        rospy.init_node("vitpose_detection_node")

        ns = rospy.get_param("~namespace", rospy.get_namespace().strip("/") or "taluy")
        self._ns = ns
        self._sim_cameras, self._sim_objects = load_sim_config(
            rospy.get_param("~sim_config", sim_config_path()), ns
        )
        self.base_frame = f"{ns}/base_link"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.gazebo = GazeboInterface(robot_name=ns)
        self._missing_model_warned = False
        self._init_shell()

    # ------------------------------------------------------------- pipeline

    def _build_pipeline(self, name_or_path, load_pose=True) -> _SimPipeline:
        config = load_object_config(name_or_path, self._ns, variant=self._variant)
        return _SimPipeline(
            config,
            self._sim_cameras,
            self._sim_objects,
            self._image_cb,
            self._info_cb,
            load_pose=load_pose,
        )

    def _swap_config(self, name_or_path):
        """Build synchronously and install with only a brief lock scope.

        Sim is deliberately stricter than the real node: a bad config fails
        the service call here instead of later in a log. The old pipeline is
        shut down after releasing _pipeline_lock so an in-flight image
        callback cannot deadlock the service while checking pipeline activity.
        """
        try:
            new_pipeline = self._build_pipeline(
                name_or_path, load_pose=(self.mode == "pose")
            )
        except Exception as exc:
            message = f"set_config('{name_or_path}') failed: {exc}"
            rospy.logerr(message)
            return SetStringResponse(success=False, message=message)
        with self._pipeline_lock:
            old, self._pipeline = self._pipeline, new_pipeline
            self._config_name = name_or_path
        if new_pipeline.detect_only and self.mode == "pose":
            # A detect-only config caps the mode.
            self.mode = "detect"
            rospy.logwarn(
                f"'{new_pipeline.object_name}' is detect-only: "
                "mode clamped to 'detect'"
            )
        self._dispose_pipeline(old)
        self._publish_status(None)
        self._missing_model_warned = False
        rospy.loginfo(
            f"sim_vitpose_node switched to object '{new_pipeline.object_name}'"
        )
        return SetStringResponse(
            success=True, message=f"switched to '{new_pipeline.object_name}'"
        )

    # ------------------------------------------------------------- callbacks

    def _info_cb(self, msg: CameraInfo, pipeline):
        pipeline.intrinsics = (msg.K[0], msg.K[4], msg.K[2], msg.K[5])
        pipeline.image_w = msg.width
        pipeline.image_h = msg.height

    def _get_base_to_camera(self, pipeline) -> Optional[np.ndarray]:
        if pipeline.base_to_camera is not None:
            return pipeline.base_to_camera
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                pipeline.optical_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(2.0),
            )
            pipeline.base_to_camera = transform_to_matrix(tf_msg.transform)
            rospy.loginfo(
                f"[{pipeline.object_name}] cached static TF: "
                f"{self.base_frame} -> {pipeline.optical_frame}"
            )
            return pipeline.base_to_camera
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                5.0, f"[{pipeline.object_name}] static TF not yet available: {exc}"
            )
            return None

    def _image_cb(self, msg: Image, pipeline):
        if not self._pipeline_active(pipeline):
            return
        if not pipeline.due(msg.header):
            return
        # Warm-up early-outs still heartbeat: "looking, can't work yet" must
        # stay distinguishable from "node is dead".
        if pipeline.intrinsics is None:
            self._publish_bbox(pipeline, None, msg.header)
            return
        base_to_camera = self._get_base_to_camera(pipeline)
        if base_to_camera is None:
            self._publish_bbox(pipeline, None, msg.header)
            return

        matrices = self.gazebo.snapshot_at_time(msg.header.stamp)
        robot_matrix = matrices.get(self.gazebo.robot_name)
        model_matrix = matrices.get(pipeline.geometry.gazebo_model)
        if robot_matrix is None:
            self._publish_bbox(pipeline, None, msg.header)
            return
        if model_matrix is None:
            if not self._missing_model_warned:
                self._missing_model_warned = True
                rospy.logwarn(
                    f"[{pipeline.object_name}] gazebo model "
                    f"'{pipeline.geometry.gazebo_model}' not in the world — "
                    "publishing 'nothing detected'"
                )
            self._publish_bbox(pipeline, None, msg.header)
            return

        full_tf = base_to_camera @ invert_rigid_transform(robot_matrix) @ model_matrix

        if pipeline.detect_only:
            bbox = self._project_bbox(pipeline, full_tf)
            self._publish_bbox(pipeline, bbox, msg.header)
            return

        keypoints, mask_planes, bbox = self._project_object(pipeline, full_tf)
        self._publish_bbox(pipeline, bbox, msg.header)
        if bbox is None:
            return  # nothing visible -> no result, like the real pipeline
        result = VitposeResult()
        result.header = msg.header
        result.object = pipeline.object_name
        result.bbox = [float(v) for v in bbox]
        result.keypoint_names = pipeline.keypoint_names
        result.mask_classes = pipeline.mask_classes
        result.mask_threshold = pipeline.mask_threshold
        result.keypoints = keypoints
        for plane in mask_planes:
            image = self.bridge.cv2_to_imgmsg(plane, encoding="mono8")
            image.header = msg.header
            result.masks.append(image)
        pipeline.result_pub.publish(result)

    # ------------------------------------------------------------- projection

    @staticmethod
    def _project_points_h(points_h, full_tf, intrinsics):
        """(us, vs, depths) for homogeneous object-frame points."""
        fx, fy, cx, cy = intrinsics
        cam = points_h @ full_tf.T
        depths = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            us = fx * cam[:, 0] / depths + cx
            vs = fy * cam[:, 1] / depths + cy
        return us, vs, depths

    def _face_visible(self, mask: SimMask, full_tf) -> bool:
        normal_cam = full_tf[:3, :3] @ mask.normal
        centroid_cam = (full_tf @ mask.centroid_h)[:3]
        return float(normal_cam @ centroid_cam) < 0.0

    def _project_object(self, pipeline, full_tf):
        """(Keypoint list, mask planes, bbox) for a pose object.

        All K keypoints are returned (confidence 1.0 visible / 0.0 not); all C
        mask planes are returned (all-zero when culled or off-screen); bbox is
        the extent of everything visible, or None when nothing is.
        """
        geometry = pipeline.geometry
        intrinsics = pipeline.intrinsics
        w_img, h_img = pipeline.image_w, pipeline.image_h

        face_visible = {
            name: (not mask.cull_backface) or self._face_visible(mask, full_tf)
            for name, mask in geometry.masks.items()
        }

        extents: List[Tuple[float, float]] = []  # visible (u, v) samples

        keypoints: List[Keypoint] = []
        points_h = np.stack([kp.position_h for kp in geometry.keypoints])
        us, vs, depths = self._project_points_h(points_h, full_tf, intrinsics)
        for i, kp_def in enumerate(geometry.keypoints):
            kp = Keypoint()
            kp.id = i
            visible = (
                depths[i] > 0.0
                and 0.0 <= us[i] < w_img
                and 0.0 <= vs[i] < h_img
                and (kp_def.face is None or face_visible[kp_def.face])
            )
            if depths[i] > 0.0:
                kp.x, kp.y = float(us[i]), float(vs[i])
            else:
                kp.x, kp.y = -1.0, -1.0
            kp.confidence = 1.0 if visible else 0.0
            if visible:
                extents.append((kp.x, kp.y))
            keypoints.append(kp)

        mask_planes: List[np.ndarray] = []
        for class_name in pipeline.mask_classes:
            mask = geometry.masks[class_name]
            plane = np.zeros((h_img, w_img), dtype=np.uint8)
            if face_visible[class_name]:
                mus, mvs, mdepths = self._project_points_h(
                    mask.polygon_h, full_tf, intrinsics
                )
                if np.all(mdepths > 0.0):
                    polygon = np.rint(np.column_stack((mus, mvs))).astype(np.int32)
                    cv2.fillConvexPoly(plane, polygon, 255)
                    ys, xs = np.nonzero(plane)
                    if len(xs):
                        extents.append((float(xs.min()), float(ys.min())))
                        extents.append((float(xs.max()), float(ys.max())))
            mask_planes.append(plane)

        if not extents:
            return keypoints, mask_planes, None
        arr = np.asarray(extents)
        x0 = float(np.clip(arr[:, 0].min(), 0, w_img - 1))
        y0 = float(np.clip(arr[:, 1].min(), 0, h_img - 1))
        x1 = float(np.clip(arr[:, 0].max(), 0, w_img - 1))
        y1 = float(np.clip(arr[:, 1].max(), 0, h_img - 1))
        return keypoints, mask_planes, (x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0)

    def _project_bbox(self, pipeline, full_tf):
        """AABB of the configured extent points (detect-only), clipped to the
        image, or None."""
        us, vs, depths = self._project_points_h(
            pipeline.geometry.bbox_points_h, full_tf, pipeline.intrinsics
        )
        if np.any(depths <= 0.0):
            return None
        u0, u1 = float(us.min()), float(us.max())
        v0, v1 = float(vs.min()), float(vs.max())
        # Cull only on zero overlap, like sim_bbox_node's project_object.
        if u1 < 0 or u0 >= pipeline.image_w or v1 < 0 or v0 >= pipeline.image_h:
            return None
        # Clip to the frame — a real detector cannot report out-of-image
        # extent — and use _project_object's inclusive +1 width convention.
        u0 = float(np.clip(u0, 0.0, pipeline.image_w - 1.0))
        u1 = float(np.clip(u1, 0.0, pipeline.image_w - 1.0))
        v0 = float(np.clip(v0, 0.0, pipeline.image_h - 1.0))
        v1 = float(np.clip(v1, 0.0, pipeline.image_h - 1.0))
        return (u0, v0, u1 - u0 + 1.0, v1 - v0 + 1.0)

    # ------------------------------------------------------------- publishing

    def _publish_bbox(self, pipeline, bbox, header):
        if pipeline.bbox_pub is None:
            return
        pipeline.bbox_pub.publish(detection_array(bbox, 1.0, pipeline.class_id, header))


if __name__ == "__main__":
    try:
        SimVitposeNode().run()
    except rospy.ROSInterruptException:
        pass
