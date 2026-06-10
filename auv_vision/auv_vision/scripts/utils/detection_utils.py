#!/usr/bin/env python3

import math
import yaml
import numpy as np
import rospy
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
import auv_common_lib.vision.camera_calibrations as camera_calibrations
import tf2_ros
import tf2_geometry_msgs


class CameraCalibration:
    def __init__(self, namespace: str):
        self.calibration = camera_calibrations.CameraCalibrationFetcher(
            namespace, True
        ).get_camera_info()

    def calculate_angles(self, pixel_coordinates: tuple) -> tuple:
        fx = self.calibration.K[0]
        fy = self.calibration.K[4]
        cx = self.calibration.K[2]
        cy = self.calibration.K[5]
        norm_x = (pixel_coordinates[0] - cx) / fx
        norm_y = (pixel_coordinates[1] - cy) / fy
        angle_x = math.atan(norm_x)
        angle_y = math.atan(norm_y)
        return angle_x, angle_y

    def distance_from_height(self, real_height: float, measured_height: float) -> float:
        focal_length = self.calibration.K[4]
        distance = (real_height * focal_length) / measured_height
        return distance

    def distance_from_width(self, real_width: float, measured_width: float) -> float:
        focal_length = self.calibration.K[0]
        distance = (real_width * focal_length) / measured_width
        return distance


class Prop:
    def __init__(self, id: int, name: str, real_height: float, real_width: float):
        self.id = id
        self.name = name
        self.real_height = real_height
        self.real_width = real_width

    def estimate_distance(
        self,
        measured_height: float,
        measured_width: float,
        calibration: CameraCalibration,
    ):
        distance_from_height = None
        distance_from_width = None

        if self.real_height is not None:
            distance_from_height = calibration.distance_from_height(
                self.real_height, measured_height
            )

        if self.real_width is not None:
            distance_from_width = calibration.distance_from_width(
                self.real_width, measured_width
            )

        if distance_from_height is not None and distance_from_width is not None:
            return (distance_from_height + distance_from_width) * 0.5
        elif distance_from_height is not None:
            return distance_from_height
        elif distance_from_width is not None:
            return distance_from_width
        else:
            rospy.logerr(f"Could not estimate distance for prop {self.name}")
            return None

    # distance_from_height
    def estimate_distance_diagonal(
        self,
        measured_height: float,
        measured_width: float,
        calibration: CameraCalibration,
    ):
        if self.real_height is not None and self.real_width is not None:
            return calibration.distance_from_height(
                self.real_height, math.sqrt(measured_height**2 + measured_width**2)
            )
        else:
            rospy.logerr(f"Could not estimate distance for prop {self.name}")
            return None


def load_config(yaml_path: str) -> dict:
    """Load detection_objects.yaml and return parsed config dict."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Build Prop objects from YAML (props are keyed by link_name, dimensions only)
    props = {}
    for link_name, prop_data in config["props"].items():
        props[link_name] = Prop(
            id=0,  # ID is per-camera, not per-prop — see id_tf_map in camera config
            name=prop_data["name"],
            real_height=prop_data.get("real_height"),
            real_width=prop_data.get("real_width"),
        )
    config["props_objects"] = props

    return config


class DetectionIDMap:
    """Bidirectional mapping between YOLO detection IDs and link names.

    Forward:  detection_id -> link_name  (dict-compatible)
    Reverse:  link_name -> detection_id  (.id_of / .ids_of)
    """

    def __init__(self, raw_map: dict):
        self._id_to_name = {int(k): v for k, v in raw_map.items()}
        self._name_to_id = {v: int(k) for k, v in raw_map.items()}

    # --- reverse lookups (the elegant part) ---

    def id_of(self, link_name: str):
        """Return the detection ID for a given link name, or None."""
        return self._name_to_id.get(link_name)

    def ids_of(self, *link_names: str) -> list:
        """Return a list of detection IDs for the given link names (skips unknown)."""
        return [self._name_to_id[n] for n in link_names if n in self._name_to_id]

    # --- dict protocol (backwards-compatible) ---

    def __contains__(self, detection_id):
        return detection_id in self._id_to_name

    def __getitem__(self, detection_id):
        return self._id_to_name[detection_id]

    def get(self, detection_id, default=None):
        return self._id_to_name.get(detection_id, default)

    def keys(self):
        return self._id_to_name.keys()

    def items(self):
        return self._id_to_name.items()


def build_id_tf_map(camera_config: dict) -> DetectionIDMap:
    """Build a bidirectional DetectionIDMap from camera config's id_tf_map."""
    return DetectionIDMap(camera_config.get("id_tf_map", {}))


def check_inside_image(
    detection, image_width: int = 640, image_height: int = 480
) -> bool:
    """Check if a detection bounding box is fully inside the image."""
    center = detection.bbox.center
    half_size_x = detection.bbox.size_x * 0.5
    half_size_y = detection.bbox.size_y * 0.5
    deadzone = 5  # pixels
    if (
        center.x + half_size_x >= image_width - deadzone
        or center.x - half_size_x <= deadzone
    ):
        return False
    if (
        center.y + half_size_y >= image_height - deadzone
        or center.y - half_size_y <= deadzone
    ):
        return False
    return True


TORPEDO_MASK_REFERENCE_WIDTH = 800.0
TORPEDO_MASK_REFERENCE_HEIGHT = 448.0
TORPEDO_FORBIDDEN_SIDE_MASKS = (
    (
        (0.0, 0.0),
        (185.0, 0.0),
        (145.0, 45.0),
        (118.0, 95.0),
        (102.0, 155.0),
        (96.0, 225.0),
        (106.0, 290.0),
        (130.0, 360.0),
        (175.0, 447.0),
        (0.0, 447.0),
    ),
    (
        (799.0, 0.0),
        (615.0, 0.0),
        (655.0, 45.0),
        (682.0, 95.0),
        (698.0, 155.0),
        (704.0, 225.0),
        (694.0, 290.0),
        (670.0, 360.0),
        (625.0, 447.0),
        (799.0, 447.0),
    ),
)


BOTTOM_MASK_REFERENCE_WIDTH = 1920.0
BOTTOM_MASK_REFERENCE_HEIGHT = 1080.0
BOTTOM_FORBIDDEN_MASKS = (
    # Polygon 1 — left side AUV body / arms
    (
        (438.0, 29.0),
        (447.0, 98.0),
        (458.0, 123.0),
        (460.0, 175.0),
        (454.0, 202.0),
        (454.0, 226.0),
        (450.0, 261.0),
        (443.0, 277.0),
        (433.0, 294.0),
        (397.0, 333.0),
        (393.0, 367.0),
        (394.0, 408.0),
        (405.0, 423.0),
        (455.0, 442.0),
        (458.0, 467.0),
        (459.0, 517.0),
        (457.0, 536.0),
        (423.0, 536.0),
        (411.0, 574.0),
        (412.0, 592.0),
        (412.0, 625.0),
        (413.0, 653.0),
        (413.0, 663.0),
        (430.0, 690.0),
        (445.0, 722.0),
        (448.0, 750.0),
        (447.0, 777.0),
        (443.0, 807.0),
        (440.0, 854.0),
        (441.0, 877.0),
        (441.0, 899.0),
        (439.0, 920.0),
        (438.0, 945.0),
        (420.0, 975.0),
        (439.0, 979.0),
        (481.0, 979.0),
        (492.0, 999.0),
        (508.0, 1021.0),
        (495.0, 1046.0),
        (471.0, 1056.0),
        (470.0, 1075.0),
        (2.0, 1080.0),
        (2.0, 32.0),
        (439.0, 34.0),
    ),
    # Polygon 2 — right side DVL
    (
        (1875.0, 144.0),
        (1788.0, 177.0),
        (1732.0, 218.0),
        (1685.0, 262.0),
        (1645.0, 330.0),
        (1629.0, 367.0),
        (1607.0, 424.0),
        (1603.0, 504.0),
        (1594.0, 553.0),
        (1603.0, 591.0),
        (1614.0, 640.0),
        (1649.0, 713.0),
        (1700.0, 791.0),
        (1755.0, 846.0),
        (1848.0, 885.0),
        (1892.0, 908.0),
        (1920.0, 920.0),
        (1918.0, 151.0),
        (1920.0, 142.0),
        (1876.0, 145.0),
    ),
)


def _scale_polygon(
    points, image_width: int, image_height: int, ref_w: float, ref_h: float
):
    """Scale polygon points from reference resolution to actual image resolution."""
    scale_x = image_width / ref_w
    scale_y = image_height / ref_h
    return tuple((x * scale_x, y * scale_y) for x, y in points)


def _point_in_polygon(point, polygon) -> bool:
    """Return True when point is inside polygon, including polygon boundary."""
    point_x, point_y = point
    inside = False

    for index, current in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        x1, y1 = current
        x2, y2 = next_point

        cross_product = (point_y - y1) * (x2 - x1) - (point_x - x1) * (y2 - y1)
        if (
            abs(cross_product) < 1e-9
            and min(x1, x2) <= point_x <= max(x1, x2)
            and min(y1, y2) <= point_y <= max(y1, y2)
        ):
            return True

        if (y1 > point_y) != (y2 > point_y):
            intersection_x = (x2 - x1) * (point_y - y1) / (y2 - y1) + x1
            if point_x <= intersection_x:
                inside = not inside

    return inside


def check_inside_image_torpedo(
    detection,
    image_width: int = 640,
    image_height: int = 480,
    forbidden_side_masks=None,
) -> bool:
    """Check if a torpedo detection bbox is inside the usable image area."""
    center = detection.bbox.center
    half_size_x = detection.bbox.size_x * 0.5
    half_size_y = detection.bbox.size_y * 0.5
    deadzone = 5  # pixels
    if (
        center.x + half_size_x >= image_width - deadzone
        or center.x - half_size_x <= deadzone
    ):
        return False
    if (
        center.y + half_size_y >= image_height - deadzone
        or center.y - half_size_y <= deadzone
    ):
        return False

    bbox_corners = (
        (center.x - half_size_x, center.y - half_size_y),
        (center.x + half_size_x, center.y - half_size_y),
        (center.x + half_size_x, center.y + half_size_y),
        (center.x - half_size_x, center.y + half_size_y),
    )

    if forbidden_side_masks is None:
        forbidden_side_masks = tuple(
            _scale_polygon(
                mask,
                image_width,
                image_height,
                TORPEDO_MASK_REFERENCE_WIDTH,
                TORPEDO_MASK_REFERENCE_HEIGHT,
            )
            for mask in TORPEDO_FORBIDDEN_SIDE_MASKS
        )

    for forbidden_mask in forbidden_side_masks:
        if any(_point_in_polygon(corner, forbidden_mask) for corner in bbox_corners):
            return False

    return True


def check_inside_image_bottom(
    detection,
    image_width: int = 1920,
    image_height: int = 1080,
    forbidden_masks=None,
) -> bool:
    """Check if a bottom-camera detection bbox is inside the usable image area.

    Returns False if:
      - The bbox touches the image edges (within deadzone), OR
      - Any corner of the bbox falls inside a forbidden mask polygon.
    """
    center = detection.bbox.center
    half_size_x = detection.bbox.size_x * 0.5
    half_size_y = detection.bbox.size_y * 0.5
    deadzone = 5  # pixels

    # Edge check
    if (
        center.x + half_size_x >= image_width - deadzone
        or center.x - half_size_x <= deadzone
    ):
        return False
    if (
        center.y + half_size_y >= image_height - deadzone
        or center.y - half_size_y <= deadzone
    ):
        return False

    # Forbidden polygon check
    bbox_corners = (
        (center.x - half_size_x, center.y - half_size_y),
        (center.x + half_size_x, center.y - half_size_y),
        (center.x + half_size_x, center.y + half_size_y),
        (center.x - half_size_x, center.y + half_size_y),
    )

    if forbidden_masks is None:
        forbidden_masks = tuple(
            _scale_polygon(
                mask,
                image_width,
                image_height,
                BOTTOM_MASK_REFERENCE_WIDTH,
                BOTTOM_MASK_REFERENCE_HEIGHT,
            )
            for mask in BOTTOM_FORBIDDEN_MASKS
        )

    for forbidden_mask in forbidden_masks:
        if any(_point_in_polygon(corner, forbidden_mask) for corner in bbox_corners):
            return False

    return True


def calculate_angles_and_offsets(calibration, bbox_center, distance):
    """Calculate viewing angles and XY offsets from calibration, bbox center, and distance.

    Returns:
        (angles, offset_x, offset_y) where angles is (angle_x, angle_y)
    """
    angles = calibration.calculate_angles((bbox_center.x, bbox_center.y))
    offset_x = math.tan(angles[0]) * distance
    offset_y = math.tan(angles[1]) * distance
    return angles, offset_x, offset_y


def transform_to_odom_and_publish(
    camera_frame,
    child_frame_id,
    offset_x,
    offset_y,
    distance,
    detection_stamp,
    tf_buffer,
    publisher,
    rotation_quat=None,
):
    """Transform a detection from camera frame to odom and publish as TransformStamped.

    Uses detection_stamp for the header timestamp.

    Args:
        rotation_quat: Optional (x, y, z, w) in camera_frame.  When None
            (default, legacy YOLO-bbox path) the published rotation is the
            camera-frame identity quaternion — appropriate for a 2D bbox
            detection that carries no orientation.  When supplied (PnP /
            ArUco / any 6-DOF source), the orientation is transformed
            through TF along with the position so the published rotation
            is the actual pose in odom.
    """
    transform_stamped_msg = TransformStamped()
    transform_stamped_msg.header.stamp = detection_stamp
    transform_stamped_msg.header.frame_id = camera_frame
    transform_stamped_msg.child_frame_id = child_frame_id

    transform_stamped_msg.transform.translation = Vector3(offset_x, offset_y, distance)
    if rotation_quat is None:
        transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
    else:
        transform_stamped_msg.transform.rotation = Quaternion(
            float(rotation_quat[0]),
            float(rotation_quat[1]),
            float(rotation_quat[2]),
            float(rotation_quat[3]),
        )

    try:
        pose_stamped = PoseStamped()
        pose_stamped.header = transform_stamped_msg.header
        pose_stamped.pose.position = transform_stamped_msg.transform.translation
        pose_stamped.pose.orientation = transform_stamped_msg.transform.rotation

        transformed_pose_stamped = tf_buffer.transform(
            pose_stamped, "odom", rospy.Duration(4.0)
        )

        final_transform_stamped = TransformStamped()
        final_transform_stamped.header = transformed_pose_stamped.header
        # Use detection_stamp, not the transformed header stamp
        final_transform_stamped.header.stamp = detection_stamp
        final_transform_stamped.child_frame_id = child_frame_id
        final_transform_stamped.transform.translation = (
            transformed_pose_stamped.pose.position
        )
        if rotation_quat is None:
            # Legacy bbox behaviour: keep the camera-frame identity quat.
            final_transform_stamped.transform.rotation = (
                transform_stamped_msg.transform.rotation
            )
        else:
            # Real 6-DOF pose: use the rotation as transformed into odom.
            final_transform_stamped.transform.rotation = (
                transformed_pose_stamped.pose.orientation
            )

        publisher.publish(final_transform_stamped)
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        rospy.logwarn_throttle(5.0, f"Transform error for {child_frame_id}: {e}")


def calculate_intersection_with_plane(point1_odom, point2_odom, z_plane):
    """Calculate where a ray (point1→point2) intersects a horizontal plane at z_plane."""
    if point2_odom.point.z != point1_odom.point.z:
        t = (z_plane - point1_odom.point.z) / (
            point2_odom.point.z - point1_odom.point.z
        )
        if 0 <= t <= 1:
            x = point1_odom.point.x + t * (point2_odom.point.x - point1_odom.point.x)
            y = point1_odom.point.y + t * (point2_odom.point.y - point1_odom.point.y)
            return x, y, z_plane
        else:
            return None
    else:
        rospy.logwarn("The line segment is parallel to the ground plane.")
        return None


# ---------------------------------------------------------------------------
# Shape factories — shared by sim_bbox_node, sim_keypoint_node, keypoint_pose_node
# ---------------------------------------------------------------------------


def rect(half_extents):
    axes = [row for row in np.diag(half_extents) if np.any(row)]
    if len(axes) != 2:
        raise ValueError("Exactly two non-zero half-extents required")
    return [sx * axes[0] + sy * axes[1] for sx in [1, -1] for sy in [1, -1]]


def circle(radii, n=16):
    axes = [row for row in np.diag(radii) if np.any(row)]
    if len(axes) != 2:
        raise ValueError("Exactly two non-zero radii required")
    return [
        axes[0] * np.cos(t) + axes[1] * np.sin(t)
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


def box(half_x, half_y, half_z):
    return [
        np.array([sx * half_x, sy * half_y, sz * half_z])
        for sx in [1, -1]
        for sy in [1, -1]
        for sz in [1, -1]
    ]


def cylinder(half_extents, n=8):
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
    # Explicit list of 3D points — for objects whose keypoints don't fit a
    # parametric shape (e.g. valve: 8 bolts in a ring + face center).
    "points": lambda args: [np.array(p, dtype=np.float64) for p in args],
}


def build_pose_keypoints(model_spec) -> dict:
    """Expand a YAML model spec (list of shape parts) into {keypoint id: (3,)
    model point}. Shared by keypoint_pose_node (PnP model) and
    valve_keypoint_node (RANSAC consensus gate) so both read the same geometry
    from the same YAML."""
    result = {}
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
