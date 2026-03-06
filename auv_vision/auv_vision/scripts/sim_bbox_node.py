#!/usr/bin/env python3

"""
Simulation ground truth bounding box node.

Replaces YOLO tracker inference in simulation by publishing YoloResult messages
derived from Gazebo ground truth model poses. Projects 3D object positions into
each camera's image plane using pinhole geometry and publishes pixel-space
bounding boxes on the same topics the real trackers use.

Topics published:
  /yolo_result_front, /yolo_result_bottom, /yolo_result_torpedo   (YoloResult)
  /yolo_image_front, /yolo_image_bottom, /yolo_image_torpedo       (annotated Image)
"""

import os
import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)
from geometry_msgs.msg import Pose, Pose2D, Transform
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates

import rospkg
import xml.etree.ElementTree as ET

import tf2_ros
import tf.transformations as tft


CLASS_NAMES = {
    "front": {
        0: "sawfish",
        1: "shark",
        2: "red_pipe",
        3: "white_pipe",
        4: "torpedo_map",
        6: "bin_whole",
        7: "octagon",
    },
    "bottom": {
        0: "bin_shark",
        1: "bin_sawfish",
        2: "octagon_table",
        3: "octagon_bin_pink",
        4: "octagon_bin_yellow",
    },
    "torpedo": {
        5: "torpedo_hole",
    },
    "bottom_seg": {
        0: "bottle",
    },
}


@dataclass
class SimObject:
    class_id: int
    camera: str
    gazebo_model: str
    offset: np.ndarray
    boundary: List[np.ndarray]


def yz_rect(half_y: float, half_z: float) -> List[np.ndarray]:
    hw = np.array([0.0, half_y, 0.0])
    hh = np.array([0.0, 0.0, half_z])
    return [sw * hw + sh * hh for sw, sh in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]


def yz_circle(radius: float, n: int = 16) -> List[np.ndarray]:
    return [
        np.array([0.0, radius * np.cos(t), radius * np.sin(t)])
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


def xz_rect(half_x: float, half_z: float) -> List[np.ndarray]:
    hw = np.array([half_x, 0.0, 0.0])
    hh = np.array([0.0, 0.0, half_z])
    return [sw * hw + sh * hh for sw, sh in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]


def xy_rect(half_x: float, half_y: float) -> List[np.ndarray]:
    hw = np.array([half_x, 0.0, 0.0])
    hh = np.array([0.0, half_y, 0.0])
    return [sw * hw + sh * hh for sw, sh in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]


def box(half_x: float, half_y: float, half_z: float) -> List[np.ndarray]:
    return [
        np.array([sx * half_x, sy * half_y, sz * half_z])
        for sx in [1, -1]
        for sy in [1, -1]
        for sz in [1, -1]
    ]


def z_cylinder(radius: float, half_height: float, n: int = 8) -> List[np.ndarray]:
    points = []
    for dz in [half_height, -half_height]:
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            points.append(np.array([radius * np.cos(t), radius * np.sin(t), dz]))
    return points


def load_dae_vertices(
    model_path: str, vertex_indices: Optional[List[int]] = None
) -> List[np.ndarray]:
    """Load mesh vertices from a COLLADA (.dae) file in auv_sim_description.

    Args:
        model_path: Path relative to the auv_sim_description package root,
            e.g. "models/robosub_bottle/meshes/bottle_final1.dae".
        vertex_indices: If provided, return only the vertices at these indices.
    """
    rp = rospkg.RosPack()
    dae_path = os.path.join(rp.get_path("auv_sim_description"), model_path)
    collada_ns = "http://www.collada.org/2005/11/COLLADASchema"
    ns = {"c": collada_ns}
    tree = ET.parse(dae_path)
    root = tree.getroot()

    # Find the positions float_array
    fa = None
    for el in root.iter(f"{{{collada_ns}}}float_array"):
        if "positions" in el.attrib.get("id", ""):
            fa = el
            break
    if fa is None:
        raise ValueError(f"No positions float_array found in {dae_path}")
    raw = [float(x) for x in fa.text.split()]
    verts = np.array(raw).reshape(-1, 3)

    # Walk the visual scene node chain to accumulate transforms
    transform = np.eye(4)
    for node in root.findall(".//c:visual_scene//c:node", ns):
        mat_el = node.find("c:matrix", ns)
        if mat_el is not None:
            vals = [float(x) for x in mat_el.text.split()]
            mat = np.array(vals).reshape(4, 4)
            if not np.allclose(mat, np.eye(4)):
                transform = transform @ mat

    all_points = [(transform @ np.array([*v, 1.0]))[:3] for v in verts]
    if vertex_indices is not None:
        return [all_points[i] for i in vertex_indices]
    return all_points


def pose_to_matrix(pose: Pose) -> np.ndarray:
    T = tft.translation_matrix([pose.position.x, pose.position.y, pose.position.z])
    R = tft.quaternion_matrix(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    return T @ R


def transform_to_matrix(transform: Transform) -> np.ndarray:
    T = tft.translation_matrix(
        [transform.translation.x, transform.translation.y, transform.translation.z]
    )
    R = tft.quaternion_matrix(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )
    return T @ R


def project_object(
    points_world: List[np.ndarray],
    world_to_camera: np.ndarray,
    K: List[float],
    image_w: int,
    image_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Project 3D world-frame boundary points to a 2D axis-aligned bounding box.

    Transforms each point into the camera frame via pinhole projection and
    returns the AABB of the projected points.

    Args:
        points_world: list of (3,) boundary positions in world frame.
        world_to_camera: 4x4 world-to-camera-optical transform.
        K: camera intrinsic matrix as a flat 9-element list (row-major).
        image_w: image width in pixels.
        image_h: image height in pixels.

    Returns:
        (center_x, center_y, size_x, size_y) in pixels, or None if not visible.
    """
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]

    us, vs = [], []
    for point in points_world:
        p = (world_to_camera @ np.array([*point, 1.0]))[:3]
        if p[2] <= 0:
            return None
        us.append(fx * p[0] / p[2] + cx)
        vs.append(fy * p[1] / p[2] + cy)

    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    # Cull only if the bbox has zero overlap with the image
    if u_max < 0 or u_min >= image_w or v_max < 0 or v_min >= image_h:
        return None

    return (u_min + u_max) / 2, (v_min + v_max) / 2, u_max - u_min, v_max - v_min


def make_detection(
    class_id: int, cx: float, cy: float, sx: float, sy: float
) -> Detection2D:
    det = Detection2D()
    hyp = ObjectHypothesisWithPose()
    hyp.id = class_id
    hyp.score = 1.0
    det.results.append(hyp)
    det.bbox = BoundingBox2D()
    det.bbox.center = Pose2D(x=cx, y=cy, theta=0.0)
    det.bbox.size_x = sx
    det.bbox.size_y = sy
    return det


class GazeboInterface:
    """Caches Gazebo model states from a single topic subscription.

    All poses come from the same /gazebo/model_states message, so the robot
    and object poses are guaranteed to be from the same simulation instant.
    """

    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        self.model_poses: Dict[str, Pose] = {}

        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1
        )

    def _model_states_cb(self, msg: ModelStates):
        self.model_poses = dict(zip(msg.name, msg.pose))

    def get_model_matrix(self, model_name: str) -> Optional[np.ndarray]:
        pose = self.model_poses.get(model_name)
        if pose is None:
            return None
        return pose_to_matrix(pose)

    def get_robot_matrix(self) -> Optional[np.ndarray]:
        return self.get_model_matrix(self.robot_name)

    def object_boundary_in_world(
        self,
        model_name: str,
        offset: np.ndarray,
        boundary: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        model_matrix = self.get_model_matrix(model_name)
        if model_matrix is None:
            return None

        points = []
        for bp in boundary:
            local = offset + bp
            world = (model_matrix @ np.array([*local, 1.0]))[:3]
            points.append(world)
        return points


class SimCamera:
    """Handles projection and publishing for one simulated camera.

    Image-callback-driven: projection happens when a new camera image arrives,
    ensuring the bboxes are computed from the same simulation instant as the
    image content. The base_link → optical_link transform is static (from the
    URDF) and is looked up once via TF, then cached forever.
    """

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
        gazebo: "GazeboInterface",
        objects: List[SimObject],
    ):
        self.name = name
        self.optical_frame = optical_frame
        self.base_frame = base_frame
        self.tf_buffer = tf_buffer
        self.bridge = CvBridge()
        self.gazebo = gazebo
        self.objects = objects

        self.K: Optional[List[float]] = None
        self.image_w = 0
        self.image_h = 0

        # Static transform: base_link → camera optical link (cached on first use)
        self._base_to_camera: Optional[np.ndarray] = None

        self.result_pub = rospy.Publisher(result_topic, YoloResult, queue_size=1)
        self.image_pub = rospy.Publisher(image_out_topic, Image, queue_size=1)

        rospy.Subscriber(camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(
            image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24
        )

    def _info_cb(self, msg: CameraInfo):
        self.K = list(msg.K)
        self.image_w = msg.width
        self.image_h = msg.height

    def _image_cb(self, msg: Image):
        if self.K is None:
            return

        base_to_camera = self._get_base_to_camera()
        if base_to_camera is None:
            return

        robot_matrix = self.gazebo.get_robot_matrix()
        if robot_matrix is None:
            return

        robot_matrix_inv = np.linalg.inv(robot_matrix)
        world_to_camera = base_to_camera @ robot_matrix_inv

        stamp = msg.header.stamp
        detections: List[Detection2D] = []

        for obj in self.objects:
            points = self.gazebo.object_boundary_in_world(
                obj.gazebo_model, obj.offset, obj.boundary
            )
            if points is None:
                continue

            result = project_object(
                points, world_to_camera, self.K, self.image_w, self.image_h
            )
            if result is not None:
                detections.append(make_detection(obj.class_id, *result))

        # Publish YoloResult with the image's own timestamp
        result_msg = YoloResult()
        result_msg.header.stamp = stamp
        result_msg.detections = Detection2DArray()
        result_msg.detections.header.stamp = stamp
        result_msg.detections.detections = detections
        self.result_pub.publish(result_msg)

        # Publish annotated debug image
        if self.image_pub.get_num_connections() > 0:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self._draw_detections(cv_image, detections, self.name)
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
                f"[{self.name}] cached static TF: {self.base_frame} → {self.optical_frame}"
            )
            return self._base_to_camera
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5, f"[{self.name}] static TF not yet available: {e}")
            return None

    @staticmethod
    def _draw_detections(
        image: np.ndarray, detections: List[Detection2D], camera_name: str
    ):
        names = CLASS_NAMES.get(camera_name, {})
        for det in detections:
            if not det.results:
                continue
            cx, cy = int(det.bbox.center.x), int(det.bbox.center.y)
            w, h = int(det.bbox.size_x), int(det.bbox.size_y)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            label = names.get(det.results[0].id, str(det.results[0].id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"SIM {label}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )


class SimBboxNode:
    def __init__(self):
        rospy.init_node("sim_bbox_node")

        ns = rospy.get_param("~namespace", "taluy")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        base_frame = f"{ns}/base_link"

        self.gazebo = GazeboInterface(robot_name=ns)

        # Gate signs — 0.3048 m (1 ft) square, face in YZ plane
        gate_sign = yz_rect(0.3048 / 2, 0.3048 / 2)

        # Torpedo map panel — 0.6096 m (2 ft) square, face in YZ plane
        torp_map = yz_rect(0.3048, -0.3048)

        # Torpedo holes — 0.125 m diameter circles, 16-point circumference
        torp_hole = yz_circle(0.125 / 2)

        # Slalom pipes — vertical cylinders, radius 0.0127 m, height 0.9 m
        pipe_boundary = z_cylinder(radius=0.0127, half_height=0.45)

        # Bin — full structure for front camera, two squares for bottom camera
        # Face in XZ plane of model (normal along Y)
        # bin_whole: `-_-` cross-section — narrow base, wide overhang.
        # Base/walls (Y ≈ 0.005): X ±0.315, Z ±0.162 from center
        # Overhang   (Y ≈ 0.156): X ±0.457, Z ±0.305 from center
        dy_base = np.array([0.0, -0.076, 0.0])  # base Y offset from center
        dy_ovhg = np.array([0.0, 0.075, 0.0])  # overhang Y offset from center
        bin_whole = [dy_base + bp for bp in xz_rect(0.315, 0.162)] + [
            dy_ovhg + bp for bp in xz_rect(0.457, 0.305)
        ]
        bin_square = xz_rect(0.3048 / 2, 0.3048 / 2)  # each marker: ~1 ft square

        all_objects: List[SimObject] = [
            SimObject(
                class_id=0,
                camera="front",
                gazebo_model="robosub_gate",
                offset=np.array([0.0, -0.762, 1.356]),
                boundary=gate_sign,
            ),  # sawfish
            SimObject(
                class_id=1,
                camera="front",
                gazebo_model="robosub_gate",
                offset=np.array([0.0, 0.762, 1.356]),
                boundary=gate_sign,
            ),  # shark
            # Map panel (front camera, class 4)
            SimObject(
                class_id=4,
                camera="front",
                gazebo_model="robosub_torpedo",
                offset=np.array([-0.1166, 0.6472, -0.9490]),
                boundary=torp_map,
            ),  # torpedo_map
            # Holes (torpedo camera, class 5) — both published as ID 5,
            # the pose estimator sorts upper/lower by image Y position.
            SimObject(
                class_id=5,
                camera="torpedo",
                gazebo_model="robosub_torpedo",
                offset=np.array([-0.1166, 0.6089, -0.9687]),
                boundary=torp_hole,
            ),  # torpedo hole 1 (lower in mesh)
            SimObject(
                class_id=5,
                camera="torpedo",
                gazebo_model="robosub_torpedo",
                offset=np.array([-0.1166, 0.8619, -1.0787]),
                boundary=torp_hole,
            ),  # torpedo hole 2 (upper in mesh)
        ]

        # Each slalom model has: left_white (-1.5,0,0.45),
        # red (0,0,0.45), right_white (1.5,0,0.45) in model frame.
        slalom_pipes = [
            (3, np.array([-1.5, 0.0, 0.45])),  # left white
            (2, np.array([0.0, 0.0, 0.45])),  # red
            (3, np.array([1.5, 0.0, 0.45])),  # right white
        ]
        for i in range(1, 4):
            for class_id, offset in slalom_pipes:
                all_objects.append(
                    SimObject(
                        class_id=class_id,
                        camera="front",
                        gazebo_model=f"robosub_slalom_{i}",
                        offset=offset,
                        boundary=pipe_boundary,
                    )
                )

        all_objects.extend(
            [
                # Whole bin structure (front camera, class 6)
                SimObject(
                    class_id=6,
                    camera="front",
                    gazebo_model="robosub_bin",
                    offset=np.array([0.0, 0.0812, 0.9337]),
                    boundary=bin_whole,
                ),  # bin_whole
                # Bottom camera markers — class IDs 0/1 are reused
                # (pose estimator has per-camera id_tf_map)
                SimObject(
                    class_id=0,
                    camera="bottom",
                    gazebo_model="robosub_bin",
                    offset=np.array([-0.1525, 0.0144, 0.9337]),
                    boundary=bin_square,
                ),  # bin_shark (left square)
                SimObject(
                    class_id=1,
                    camera="bottom",
                    gazebo_model="robosub_bin",
                    offset=np.array([0.1525, 0.0144, 0.9337]),
                    boundary=bin_square,
                ),  # bin_sawfish (right square)
            ]
        )

        # Legs (Z < 0.54): XY ±0.32 × ±0.55
        # Basket level (Z ≥ 0.54): baskets only on ±Y (2 and 4),
        #   X extent = table width (±0.305), Y includes baskets (±0.55)
        # Center: (0, 0, 0.347), no rotation in world
        dz_legs = np.array(
            [0.0, 0.0, -0.077]
        )  # leg Z center relative to overall center
        dz_top = np.array(
            [0.0, 0.0, 0.270]
        )  # basket Z center relative to overall center
        oct_boundary = [dz_legs + bp for bp in box(0.32, 0.55, 0.27)] + [
            dz_top + bp for bp in box(0.305, 0.55, 0.076)
        ]
        all_objects.append(
            SimObject(
                class_id=7,
                camera="front",
                gazebo_model="robosub_octagon",
                offset=np.array([0.0, 0.0, 0.347]),
                boundary=oct_boundary,
            )
        )  # octagon
        # Octagon table — central flat panel (bottom camera, class 2)
        # octagon_middle: XY ±0.3048 m, Z ≈ 0.645, horizontal face
        all_objects.append(
            SimObject(
                class_id=2,
                camera="bottom",
                gazebo_model="robosub_octagon",
                offset=np.array([0.0, 0.0, 0.6446]),
                boundary=xy_rect(0.3048, 0.3048),
            )
        )  # octagon_table
        # Octagon bins — 2 baskets on ±Y, top-face corners (bottom camera, class 3)
        # basket_2 at (-0.021, +0.432), basket_4 at (-0.021, -0.432)
        # DAE nodes have 90° rotation: local X→world Y, local Y→world -X
        # After rotation: ±0.1715 in X (world), ±0.1175 in Y (world)
        oct_bin_boundary = xy_rect(0.1715, 0.1175)
        all_objects.append(
            SimObject(
                class_id=4,
                camera="bottom",
                gazebo_model="robosub_octagon",
                offset=np.array([-0.021, 0.432, 0.6865]),
                boundary=oct_bin_boundary,
            )
        )  # octagon_bin_yellow (+Y basket)
        all_objects.append(
            SimObject(
                class_id=3,
                camera="bottom",
                gazebo_model="robosub_octagon",
                offset=np.array([-0.021, -0.432, 0.6865]),
                boundary=oct_bin_boundary,
            )
        )  # octagon_bin_pink (-Y basket)

        # Hand-picked silhouette vertices from mesh (via dae_vertex_picker).
        # fmt: off
        bottle_vert_indices = [
            0, 1, 2, 3, 5, 7, 9, 11, 18, 26, 28, 30, 33, 40, 41, 45, 46,
            47, 48, 55, 56, 58, 67, 75, 90, 97, 98, 101, 111, 114, 115, 116,
            118, 119, 122, 126, 127, 128, 131, 134, 136, 142, 160, 163, 164,
            165, 166, 169, 171, 178, 180, 183, 189, 190, 192, 194, 205, 206,
            214, 218, 220, 222, 225, 226, 234, 235, 237, 241, 243, 245, 246,
            247, 249, 250,
        ]
        # fmt: on
        bottle_boundary = load_dae_vertices(
            "models/robosub_bottle/meshes/bottle_final1.dae",
            vertex_indices=bottle_vert_indices,
        )
        rospy.loginfo(f"Bottle: {len(bottle_boundary)} boundary vertices")
        all_objects.append(
            SimObject(
                class_id=0,
                camera="bottom_seg",
                gazebo_model="bottle_final1",
                offset=np.array([0.0, 0.0, 0.0]),
                boundary=bottle_boundary,
            )
        )  # bottle

        objects_by_camera: Dict[str, List[SimObject]] = {
            "front": [],
            "bottom": [],
            "torpedo": [],
            "bottom_seg": [],
        }
        for obj in all_objects:
            objects_by_camera[obj.camera].append(obj)

        self.cameras: Dict[str, SimCamera] = {
            "front": SimCamera(
                name="front",
                image_topic=f"/{ns}/cameras/cam_front/image_raw",
                camera_info_topic=f"/{ns}/cameras/cam_front/camera_info",
                optical_frame=f"{ns}/base_link/front_camera_optical_link",
                base_frame=base_frame,
                result_topic="/yolo_result_front",
                image_out_topic="/yolo_image_front",
                tf_buffer=self.tf_buffer,
                gazebo=self.gazebo,
                objects=objects_by_camera["front"],
            ),
            "bottom": SimCamera(
                name="bottom",
                image_topic=f"/{ns}/cameras/cam_bottom/image_raw",
                camera_info_topic=f"/{ns}/cameras/cam_bottom/camera_info",
                optical_frame=f"{ns}/base_link/bottom_camera_optical_link",
                base_frame=base_frame,
                result_topic="/yolo_result_bottom",
                image_out_topic="/yolo_image_bottom",
                tf_buffer=self.tf_buffer,
                gazebo=self.gazebo,
                objects=objects_by_camera["bottom"],
            ),
            "torpedo": SimCamera(
                name="torpedo",
                image_topic=f"/{ns}/cameras/cam_torpedo/image_raw",
                camera_info_topic=f"/{ns}/cameras/cam_torpedo/camera_info",
                optical_frame=f"{ns}/base_link/torpedo_camera_optical_link",
                base_frame=base_frame,
                result_topic="/yolo_result_torpedo",
                image_out_topic="/yolo_image_torpedo",
                tf_buffer=self.tf_buffer,
                gazebo=self.gazebo,
                objects=objects_by_camera["torpedo"],
            ),
            "bottom_seg": SimCamera(
                name="bottom_seg",
                image_topic=f"/{ns}/cameras/cam_bottom/image_raw",
                camera_info_topic=f"/{ns}/cameras/cam_bottom/camera_info",
                optical_frame=f"{ns}/base_link/bottom_camera_optical_link",
                base_frame=base_frame,
                result_topic="/yolo_result_seg",
                image_out_topic="/yolo_image_seg",
                tf_buffer=self.tf_buffer,
                gazebo=self.gazebo,
                objects=objects_by_camera["bottom_seg"],
            ),
        }

        rospy.loginfo(
            f"SimBboxNode started — {len(all_objects)} objects, {len(self.cameras)} cameras"
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SimBboxNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
