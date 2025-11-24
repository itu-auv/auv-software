import math
from typing import Tuple, Optional
import rospy
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs

def calculate_angles(pixel_coordinates: Tuple[float, float], calibration_k: list) -> Tuple[float, float]:
    """
    Calculates the angles from the camera center to the pixel coordinates.
    """
    fx = calibration_k[0]
    fy = calibration_k[4]
    cx = calibration_k[2]
    cy = calibration_k[5]
    norm_x = (pixel_coordinates[0] - cx) / fx
    norm_y = (pixel_coordinates[1] - cy) / fy
    angle_x = math.atan(norm_x)
    angle_y = math.atan(norm_y)
    return angle_x, angle_y

def distance_from_height(real_height: float, measured_height: float, focal_length_y: float) -> float:
    """
    Calculates distance based on object height.
    """
    return (real_height * focal_length_y) / measured_height

def distance_from_width(real_width: float, measured_width: float, focal_length_x: float) -> float:
    """
    Calculates distance based on object width.
    """
    return (real_width * focal_length_x) / measured_width

def estimate_distance(
    real_height: Optional[float],
    real_width: Optional[float],
    measured_height: float,
    measured_width: float,
    calibration_k: list
) -> Optional[float]:
    """
    Estimates distance to an object based on its known dimensions and measured pixel dimensions.
    """
    dist_height = None
    dist_width = None

    if real_height is not None:
        dist_height = distance_from_height(real_height, measured_height, calibration_k[4])

    if real_width is not None:
        dist_width = distance_from_width(real_width, measured_width, calibration_k[0])

    if dist_height is not None and dist_width is not None:
        return (dist_height + dist_width) * 0.5
    elif dist_height is not None:
        return dist_height
    elif dist_width is not None:
        return dist_width
    else:
        return None

def calculate_intersection_with_plane(point1_odom: PointStamped, point2_odom: PointStamped, z_plane: float) -> Optional[Tuple[float, float, float]]:
    """
    Calculates the intersection of a line segment defined by two points in the odom frame with a horizontal plane at z=z_plane.
    """
    # Calculate t where the z component is z_plane
    if point2_odom.point.z != point1_odom.point.z:
        t = (z_plane - point1_odom.point.z) / (
            point2_odom.point.z - point1_odom.point.z
        )

        # Check if the intersection point is within the segment [point1, point2]
        # In the original code, it was 0 <= t <= 1, but for ray casting from camera (point1) through pixel (point2),
        # we generally want t >= 0. Since point2 is arbitrary scaled (distance 500m), it acts as a direction.
        # But wait, the original code used 0 <= t <= 1.
        # If point2 is at distance 500m, then t will be very small if intersection is close.
        if 0 <= t <= 1:
            # Calculate intersection point
            x = point1_odom.point.x + t * (
                point2_odom.point.x - point1_odom.point.x
            )
            y = point1_odom.point.y + t * (
                point2_odom.point.y - point1_odom.point.y
            )
            return x, y, z_plane
        else:
            # Intersection is outside the defined segment (behind camera or too far)
            return None
    else:
        # rospy.logwarn("The line segment is parallel to the ground plane.")
        return None

def check_if_detection_is_inside_image(
    bbox_center_x: float,
    bbox_center_y: float,
    bbox_size_x: float,
    bbox_size_y: float,
    image_width: int = 640,
    image_height: int = 480
) -> bool:
    """
    Checks if the detection bounding box is fully inside the image (with a margin).
    """
    half_size_x = bbox_size_x * 0.5
    half_size_y = bbox_size_y * 0.5
    deadzone = 5  # pixels
    if (
        bbox_center_x + half_size_x >= image_width - deadzone
        or bbox_center_x - half_size_x <= deadzone
    ):
        return False
    if (
        bbox_center_y + half_size_y >= image_height - deadzone
        or bbox_center_y - half_size_y <= deadzone
    ):
        return False
    return True
