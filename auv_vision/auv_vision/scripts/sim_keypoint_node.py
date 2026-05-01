#!/usr/bin/env python3

"""
Simulation ground truth keypoint node.

Replaces ViTPose inference in simulation by publishing KeypointResult messages
derived from Gazebo ground truth model poses.  Projects known 3D keypoints
(e.g. gate sign corners) into each camera's image plane using pinhole geometry.

Topics published (per camera):
  /keypoint_result_<cam>   (KeypointResult)
  /keypoint_image_<cam>    (annotated Image — only when subscribed)
"""

import os
import threading

import cv2
import numpy as np
import rospy
import yaml

from collections import deque
from cv_bridge import CvBridge
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from auv_msgs.msg import Keypoint, KeypointResult
from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates

import rospkg
import tf2_ros
import tf.transformations as tft


# ---------------------------------------------------------------------------
# Shape factories — same as sim_bbox_node
# ---------------------------------------------------------------------------


def rect(half_extents: Tuple[float, float, float]) -> List[np.ndarray]:
    axes = [row for row in np.diag(half_extents) if np.any(row)]
    if len(axes) != 2:
        raise ValueError("Exactly two non-zero half-extents required")
    return [sx * axes[0] + sy * axes[1] for sx in [1, -1] for sy in [1, -1]]


def circle(radii: Tuple[float, float, float], n: int = 16) -> List[np.ndarray]:
    axes = [row for row in np.diag(radii) if np.any(row)]
    if len(axes) != 2:
        raise ValueError("Exactly two non-zero radii required")
    return [
        axes[0] * np.cos(t) + axes[1] * np.sin(t)
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


def box(half_x: float, half_y: float, half_z: float) -> List[np.ndarray]:
    return [
        np.array([sx * half_x, sy * half_y, sz * half_z])
        for sx in [1, -1]
        for sy in [1, -1]
        for sz in [1, -1]
    ]


def cylinder(half_extents: Tuple[float, float, float], n: int = 8) -> List[np.ndarray]:
    e = np.array(half_extents)
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        if e[j] == e[k]:
            a0, a1, h = np.zeros(3), np.zeros(3), np.zeros(3)
            a0[j], a1[k], h[i] = e[j], e[k], e[i]
            return [
                a0 * np.cos(t) + a1 * np.sin(t) + s * h
                for s in [1, -1]
                for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
            ]
    raise ValueError("Exactly two equal half-extents (radii) required")


SHAPE_FACTORIES = {
    "rect": lambda args: rect(tuple(args)),
    "circle": lambda args: circle(tuple(args)),
    "box": lambda args: box(*args),
    "cylinder": lambda args: cylinder(tuple(args)),
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class KeypointDef:
    """A single 3D keypoint in the object's local frame."""

    id: int
    position_h: np.ndarray  # (4,) homogeneous


@dataclass
class SimKeypointObject:
    name: str
    cameras: List[str]
    gazebo_model: str
    keypoints: List[KeypointDef]
    match_prefix: bool = False


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gazebo interface (identical to sim_bbox_node)
# ---------------------------------------------------------------------------


class GazeboInterface:
    _BUFFER_SIZE = 100

    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        self.model_poses: Dict[str, Pose] = {}
        self.model_matrices: Dict[str, np.ndarray] = {}
        self._pose_buffer: deque = deque(maxlen=self._BUFFER_SIZE)
        self._state_lock = threading.Lock()

        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1
        )

    def _model_states_cb(self, msg: ModelStates):
        poses = dict(zip(msg.name, msg.pose))
        matrices = {name: pose_to_matrix(pose) for name, pose in poses.items()}
        with self._state_lock:
            self._pose_buffer.append((rospy.Time.now(), poses, matrices))
            self.model_poses = poses
            self.model_matrices = matrices

    def snapshot_at_time(
        self, stamp: rospy.Time
    ) -> Tuple[Dict[str, Pose], Dict[str, np.ndarray]]:
        with self._state_lock:
            if not self._pose_buffer:
                return self.model_poses, self.model_matrices
            snapshot = min(
                tuple(self._pose_buffer),
                key=lambda entry: abs((entry[0] - stamp).to_sec()),
            )
        _, poses, matrices = snapshot
        return poses, matrices


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project_keypoints(
    keypoints: List[KeypointDef],
    full_tf: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    object_name: str,
) -> List[Keypoint]:
    """Project 3D keypoints into pixel space, returning only visible ones."""
    fx, fy, cx, cy = intrinsics
    results = []

    for kp in keypoints:
        pt_cam = full_tf @ kp.position_h
        if pt_cam[2] <= 0.0:
            continue

        u = fx * pt_cam[0] / pt_cam[2] + cx
        v = fy * pt_cam[1] / pt_cam[2] + cy

        if u < 0 or u >= image_w or v < 0 or v >= image_h:
            continue

        msg = Keypoint()
        msg.id = kp.id
        msg.x = float(u)
        msg.y = float(v)
        msg.confidence = 1.0
        msg.object = object_name
        results.append(msg)

    return results


# ---------------------------------------------------------------------------
# Per-camera handler
# ---------------------------------------------------------------------------

# Distinct colors for keypoint IDs (BGR)
_KP_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 0),
    (0, 128, 255),
]


class SimKeypointCamera:
    def __init__(
        self,
        name: str,
        image_topic: str,
        camera_info_topic: str,
        optical_frame: str,
        base_frame: str,
        result_topic: str,
        image_out_topic: str,
        tf_buffer: tf2_ros.Buffer,
        gazebo: GazeboInterface,
        objects: List[SimKeypointObject],
    ):
        self.name = name
        self.optical_frame = optical_frame
        self.base_frame = base_frame
        self.tf_buffer = tf_buffer
        self.bridge = CvBridge()
        self.gazebo = gazebo
        self.objects = objects

        self.intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.image_w = 0
        self.image_h = 0

        self._base_to_camera: Optional[np.ndarray] = None
        self._debug_thread: Optional[threading.Thread] = None
        self._debug_lock = threading.Lock()

        self.result_pub = rospy.Publisher(result_topic, KeypointResult, queue_size=1)
        self.image_pub = rospy.Publisher(image_out_topic, Image, queue_size=1)

        rospy.Subscriber(camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(
            image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24
        )

    def _info_cb(self, msg: CameraInfo):
        self.intrinsics = (msg.K[0], msg.K[4], msg.K[2], msg.K[5])
        self.image_w = msg.width
        self.image_h = msg.height

    def _image_cb(self, msg: Image):
        if self.intrinsics is None:
            return

        base_to_camera = self._get_base_to_camera()
        if base_to_camera is None:
            return

        model_poses, model_matrices = self.gazebo.snapshot_at_time(msg.header.stamp)

        robot_matrix = model_matrices.get(self.gazebo.robot_name)
        if robot_matrix is None:
            return

        world_to_camera = base_to_camera @ invert_rigid_transform(robot_matrix)

        all_keypoints: List[Keypoint] = []

        for obj in self.objects:
            if obj.match_prefix:
                model_names = [
                    name for name in model_poses if name.startswith(obj.gazebo_model)
                ]
            else:
                model_names = [obj.gazebo_model]

            for model_name in model_names:
                model_matrix = model_matrices.get(model_name)
                if model_matrix is None:
                    continue
                full_tf = world_to_camera @ model_matrix

                kps = project_keypoints(
                    obj.keypoints,
                    full_tf,
                    self.intrinsics,
                    self.image_w,
                    self.image_h,
                    obj.name,
                )
                all_keypoints.extend(kps)

        result_msg = KeypointResult()
        result_msg.header.stamp = msg.header.stamp
        result_msg.keypoints = all_keypoints
        self.result_pub.publish(result_msg)

        if self.image_pub.get_num_connections() > 0:
            self._start_debug_publish(msg, all_keypoints, msg.header.stamp)

    def _start_debug_publish(
        self,
        msg: Image,
        keypoints: List[Keypoint],
        stamp: rospy.Time,
    ):
        with self._debug_lock:
            if self._debug_thread is not None and self._debug_thread.is_alive():
                return
            self._debug_thread = threading.Thread(
                target=self._publish_debug_image,
                args=(msg, keypoints, stamp),
                daemon=True,
            )
            self._debug_thread.start()

    def _publish_debug_image(
        self,
        msg: Image,
        keypoints: List[Keypoint],
        stamp: rospy.Time,
    ):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            for kp in keypoints:
                color = _KP_COLORS[kp.id % len(_KP_COLORS)]
                center = (int(kp.x), int(kp.y))
                cv2.circle(cv_image, center, 5, color, -1)
                cv2.circle(cv_image, center, 7, (255, 255, 255), 1)
                cv2.putText(
                    cv_image,
                    str(kp.id),
                    (center[0] + 10, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            out_msg.header.stamp = stamp
            self.image_pub.publish(out_msg)
        except Exception as e:
            rospy.logwarn_throttle(5, f"[{self.name}] annotation error: {e}")

    def _get_base_to_camera(self) -> Optional[np.ndarray]:
        if self._base_to_camera is not None:
            return self._base_to_camera

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.optical_frame, self.base_frame, rospy.Time(0), rospy.Duration(2.0)
            )
            self._base_to_camera = transform_to_matrix(tf_msg.transform)
            rospy.loginfo(
                f"[{self.name}] cached static TF: {self.base_frame} -> {self.optical_frame}"
            )
            return self._base_to_camera
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5, f"[{self.name}] static TF not yet available: {e}")
            return None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _build_keypoints(spec, offset: np.ndarray, id_start: int) -> List[KeypointDef]:
    """Build KeypointDefs from a YAML keypoints spec."""
    if isinstance(spec, dict):
        points = SHAPE_FACTORIES[spec["shape"]](spec["args"])
    else:
        # Compound: list of {shape, args, offset}
        points = []
        for part in spec:
            sub_points = SHAPE_FACTORIES[part["shape"]](part["args"])
            sub_offset = np.array(part.get("offset", [0, 0, 0]), dtype=np.float64)
            points.extend(p + sub_offset for p in sub_points)

    keypoints = []
    for i, pt in enumerate(points):
        local = pt + offset
        position_h = np.array([local[0], local[1], local[2], 1.0], dtype=np.float64)
        keypoints.append(KeypointDef(id=id_start + i, position_h=position_h))
    return keypoints


def load_config(config_path: str, ns: str) -> Tuple[List[SimKeypointObject], Dict]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    camera_configs = {}
    for cam_name, cam_cfg in config["cameras"].items():
        camera_configs[cam_name] = {
            k: v.replace("{ns}", ns) for k, v in cam_cfg.items()
        }

    all_objects: List[SimKeypointObject] = []
    for obj_cfg in config["objects"]:
        offset = np.array(obj_cfg["offset"], dtype=np.float64)
        keypoints = _build_keypoints(obj_cfg["keypoints"], offset, obj_cfg["id_start"])
        all_objects.append(
            SimKeypointObject(
                name=obj_cfg["name"],
                cameras=obj_cfg["cameras"],
                gazebo_model=obj_cfg["gazebo_model"],
                keypoints=keypoints,
                match_prefix=obj_cfg.get("match_prefix", False),
            )
        )

    return all_objects, camera_configs


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class SimKeypointNode:
    def __init__(self):
        rospy.init_node("sim_keypoint_node")

        ns = rospy.get_param("~namespace", "taluy")
        default_config = os.path.join(
            rospkg.RosPack().get_path("auv_vision"),
            "config",
            "sim_keypoint_objects.yaml",
        )
        config_path = rospy.get_param("~config", default_config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        base_frame = f"{ns}/base_link"

        self.gazebo = GazeboInterface(robot_name=ns)

        all_objects, camera_configs = load_config(config_path, ns)

        objects_by_camera: Dict[str, List[SimKeypointObject]] = {
            cam: [] for cam in camera_configs
        }
        for obj in all_objects:
            for cam in obj.cameras:
                objects_by_camera[cam].append(obj)

        self.cameras: Dict[str, SimKeypointCamera] = {}
        for cam_name, cam_cfg in camera_configs.items():
            self.cameras[cam_name] = SimKeypointCamera(
                name=cam_name,
                image_topic=cam_cfg["image_topic"],
                camera_info_topic=cam_cfg["camera_info_topic"],
                optical_frame=cam_cfg["optical_frame"],
                base_frame=base_frame,
                result_topic=cam_cfg["result_topic"],
                image_out_topic=cam_cfg["image_out_topic"],
                tf_buffer=self.tf_buffer,
                gazebo=self.gazebo,
                objects=objects_by_camera[cam_name],
            )

        rospy.loginfo(
            f"SimKeypointNode started — {sum(len(o.keypoints) for o in all_objects)} "
            f"keypoints across {len(all_objects)} objects, {len(self.cameras)} cameras"
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SimKeypointNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
