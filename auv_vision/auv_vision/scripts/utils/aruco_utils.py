#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
import tf2_ros

ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def get_aruco_dictionary(name):
    if name not in ARUCO_DICT_MAP:
        raise ValueError(
            f"Unknown ArUco dictionary: {name}. "
            f"Available: {list(ARUCO_DICT_MAP.keys())}"
        )
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[name])


_clahe_cache = {"clip": None, "tile": None, "obj": None}


def preprocess_image(cv_image, clahe_clip, clahe_tile):
    if _clahe_cache["clip"] != clahe_clip or _clahe_cache["tile"] != clahe_tile:
        _clahe_cache["clip"] = clahe_clip
        _clahe_cache["tile"] = clahe_tile
        _clahe_cache["obj"] = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile)
        )

    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    return _clahe_cache["obj"].apply(l_channel)


def build_marker_object_points(marker_size):
    """4x3 object points for a single marker centered at origin."""
    half = marker_size / 2.0
    return np.array(
        [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ],
        dtype=np.float32,
    )


def build_board_object_points(board_config, markers_config):
    """Build {aruco_id: 4x3 corner points} for a board's marker layout.

    Each marker's (x, y) in the layout is the top-left corner position.
    """
    obj_points = {}
    for marker_name, position in board_config["marker_layout"].items():
        marker = markers_config[marker_name]
        aruco_id = marker["aruco_id"]
        size = marker["size"]
        x, y = position["x"], position["y"]
        corners = np.array(
            [
                [x, y, 0],
                [x + size, y, 0],
                [x + size, y + size, 0],
                [x, y + size, 0],
            ],
            dtype=np.float32,
        )
        obj_points[aruco_id] = corners
    return obj_points


def solve_board_pose(
    obj_points_by_id,
    detected_ids,
    detected_corners,
    camera_matrix,
    dist_coeffs,
    ransac_iterations,
    ransac_reproj_error,
):
    """Match detected markers to board, solvePnPRansac + LM refinement.

    Returns (success, rvec, tvec, matched_ids, inliers).
    """
    obj_points_list = []
    img_points_list = []
    matched_ids = []

    for i, marker_id in enumerate(detected_ids.flatten()):
        if marker_id in obj_points_by_id:
            obj_points_list.append(obj_points_by_id[marker_id])
            img_points_list.append(detected_corners[i].reshape(4, 2))
            matched_ids.append(marker_id)

    if len(matched_ids) == 0:
        return False, None, None, [], None

    obj_points = np.vstack(obj_points_list).astype(np.float32)
    img_points = np.vstack(img_points_list).astype(np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_points,
        img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
        iterationsCount=ransac_iterations,
        reprojectionError=ransac_reproj_error,
        confidence=0.99,
    )

    if (
        not success
        or rvec is None
        or tvec is None
        or inliers is None
        or len(inliers) == 0
    ):
        return False, None, None, matched_ids, None

    inlier_indices = inliers.flatten()
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points[inlier_indices],
        img_points[inlier_indices],
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
    )

    return True, rvec, tvec, matched_ids, inliers


def solve_marker_pose(marker_obj_points, marker_corners, camera_matrix, dist_coeffs):
    """Estimate pose for a single marker. Returns (success, rvec, tvec)."""
    success, rvec, tvec = cv2.solvePnP(
        marker_obj_points,
        marker_corners.reshape(4, 2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return success, rvec, tvec


def rvec_tvec_to_quaternion(rvec, tvec, force_floor_orientation=False):
    """Convert solvePnP rvec to a quaternion.

    Applies 180-degree X rotation (ArUco -> ROS convention).
    If force_floor_orientation, keeps only yaw (for floor-mounted boards/markers).
    """
    rot_mat, _ = cv2.Rodrigues(rvec)
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rot_mat
    transform_matrix[0:3, 3] = tvec.flatten()

    rot_x_180 = tf.transformations.rotation_matrix(np.pi, [1, 0, 0])
    transform_matrix = transform_matrix @ rot_x_180

    quat = tf.transformations.quaternion_from_matrix(transform_matrix)

    if force_floor_orientation:
        euler = tf.transformations.euler_from_quaternion(quat)
        yaw = euler[2]
        quat = tf.transformations.quaternion_from_euler(np.pi, 0.0, yaw)

    return quat


def publish_pose_to_odom(
    camera_frame, child_frame_id, tvec, quaternion, stamp, tf_buffer, publisher
):
    """Transform an ArUco pose from camera frame to odom and publish as TransformStamped.

    Returns the transformed PoseStamped on success, None on failure.
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.stamp = stamp
    pose_stamped.header.frame_id = camera_frame
    pose_stamped.pose.position.x = float(tvec[0])
    pose_stamped.pose.position.y = float(tvec[1])
    pose_stamped.pose.position.z = float(tvec[2])
    pose_stamped.pose.orientation = Quaternion(
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )

    try:
        transformed = tf_buffer.transform(pose_stamped, "odom", rospy.Duration(0.1))

        odom_transform = TransformStamped()
        odom_transform.header = transformed.header
        odom_transform.header.stamp = stamp
        odom_transform.child_frame_id = child_frame_id
        odom_transform.transform.translation = transformed.pose.position
        odom_transform.transform.rotation = transformed.pose.orientation

        publisher.publish(odom_transform)
        return transformed
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        rospy.logwarn_throttle(5, f"Transform to odom failed for {child_frame_id}: {e}")
        return None
