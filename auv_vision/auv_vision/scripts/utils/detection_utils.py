#!/usr/bin/env python3

import math
import yaml
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


def build_id_tf_map(camera_config: dict) -> dict:
    """Build {detection_id: link_name} map from camera config's id_tf_map.
    
    YAML keys are strings, so we convert them to int.
    """
    raw_map = camera_config.get("id_tf_map", {})
    return {int(k): v for k, v in raw_map.items()}


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
):
    """Transform a detection from camera frame to odom and publish as TransformStamped.

    Uses detection_stamp for the header timestamp.
    """
    transform_stamped_msg = TransformStamped()
    transform_stamped_msg.header.stamp = detection_stamp
    transform_stamped_msg.header.frame_id = camera_frame
    transform_stamped_msg.child_frame_id = child_frame_id

    transform_stamped_msg.transform.translation = Vector3(
        offset_x, offset_y, distance
    )
    transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)

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
        final_transform_stamped.transform.rotation = (
            transform_stamped_msg.transform.rotation
        )

        publisher.publish(final_transform_stamped)
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        rospy.logerr(f"Transform error for {child_frame_id}: {e}")


def calculate_intersection_with_plane(point1_odom, point2_odom, z_plane):
    """Calculate where a ray (point1→point2) intersects a horizontal plane at z_plane."""
    if point2_odom.point.z != point1_odom.point.z:
        t = (z_plane - point1_odom.point.z) / (
            point2_odom.point.z - point1_odom.point.z
        )
        if 0 <= t <= 1:
            x = point1_odom.point.x + t * (
                point2_odom.point.x - point1_odom.point.x
            )
            y = point1_odom.point.y + t * (
                point2_odom.point.y - point1_odom.point.y
            )
            return x, y, z_plane
        else:
            return None
    else:
        rospy.logwarn("The line segment is parallel to the ground plane.")
        return None
