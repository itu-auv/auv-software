#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stand Orientation Estimator Node
---------------------------------
Estimates the orientation (surface normal) of the valve stand (desk)
using depth data from RealSense D435 stereo camera.

Pipeline:
  1. Detect yellow panels of the stand from color image using HSV
  2. Find pixel coordinates of the detected region
  3. Extract corresponding 3D points from PointCloud2 mapping to these pixels
  4. Fit plane with RANSAC -> surface normal vector
  5. Publish valve_stand_link TF frame

Subscribe:
  - /taluy/camera/color/image_raw       (sensor_msgs/Image)
  - /taluy/camera/depth/color/points    (sensor_msgs/PointCloud2)

Publish:
  - valve_stand_link TF frame
  - stand_orientation/debug_image       (sensor_msgs/Image)
"""

import rospy
import numpy as np
import cv2
import struct
import math

from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from cv_bridge import CvBridge
import tf2_ros
import tf.transformations as tft


def pointcloud2_to_array(cloud_msg):
    """
    Convert PointCloud2 message to (H, W, 3) numpy array.
    Assumes organized point cloud (height > 1).
    """
    # Find field offsets
    field_map = {}
    for field in cloud_msg.fields:
        field_map[field.name] = field.offset

    if "x" not in field_map or "y" not in field_map or "z" not in field_map:
        return None

    point_step = cloud_msg.point_step
    row_step = cloud_msg.row_step
    data = cloud_msg.data

    h = cloud_msg.height
    w = cloud_msg.width

    # Organized cloud check
    if h <= 1:
        rospy.logwarn_throttle(
            5.0, "Point cloud is unorganized (height=1), " "will reshape by width only."
        )
        h = 1

    points = np.zeros((h, w, 3), dtype=np.float32)

    x_off = field_map["x"]
    y_off = field_map["y"]
    z_off = field_map["z"]

    for v in range(h):
        for u in range(w):
            idx = v * row_step + u * point_step
            x = struct.unpack_from("f", data, idx + x_off)[0]
            y = struct.unpack_from("f", data, idx + y_off)[0]
            z = struct.unpack_from("f", data, idx + z_off)[0]
            points[v, u] = [x, y, z]

    return points


def ransac_plane_fit(points_3d, max_iterations=200, distance_threshold=0.03):
    """
    Find the best fitting plane (normal vector) using RANSAC.

    Args:
        points_3d: (N, 3) numpy array - 3D points
        max_iterations: Number of RANSAC iterations
        distance_threshold: Inlier distance threshold (meters)

    Returns:
        (normal, centroid, inlier_ratio) or None
        normal = np.array([nx, ny, nz]) unit vector
        centroid = np.array([cx, cy, cz]) center point of the plane
        inlier_ratio = inlier ratio (0..1)
    """
    if len(points_3d) < 3:
        return None

    best_normal = None
    best_centroid = None
    best_inlier_count = 0
    n_points = len(points_3d)

    for _ in range(max_iterations):
        # Select 3 random points
        indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points_3d[indices]

        # Calculate plane normal (cross product)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            continue  # Degenerate triangle (collinear)

        normal = normal / norm

        # Tüm noktaların düzleme mesafesini hesapla
        diffs = points_3d - p1
        distances = np.abs(np.dot(diffs, normal))

        # Count inliers
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            # Recalculate normal from inlier points (more accurate)
            inlier_points = points_3d[inliers]
            centroid = np.mean(inlier_points, axis=0)
            # PCA with covariance matrix
            centered = inlier_points - centroid
            cov_matrix = np.dot(centered.T, centered) / len(inlier_points)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Eigenvector corresponding to smallest eigenvalue = normal
            best_normal = eigenvectors[
                :, 0
            ]  # eigh returns sorted (smallest to largest)
            best_centroid = centroid

    if best_normal is None:
        return None

    inlier_ratio = best_inlier_count / n_points
    return best_normal, best_centroid, inlier_ratio


class StandOrientationEstimator:
    """
    ROS node that estimates valve stand orientation
    (surface normal) using RealSense depth + color.
    """

    def __init__(self):
        rospy.init_node("stand_orientation_estimator", anonymous=True)
        rospy.loginfo("Stand orientation estimator node starting...")

        # Stand color: Yellow panels (HSV range) - for orientation
        self.hsv_lower = np.array(rospy.get_param("~hsv_lower", [20, 80, 80]))
        self.hsv_upper = np.array(rospy.get_param("~hsv_upper", [40, 255, 255]))

        # Valve color: Orange (HSV range) - for position
        self.valve_hsv_lower = np.array(
            rospy.get_param("~valve_hsv_lower", [5, 150, 150])
        )
        self.valve_hsv_upper = np.array(
            rospy.get_param("~valve_hsv_upper", [20, 255, 255])
        )
        self.min_valve_area = rospy.get_param("~min_valve_area", 200)

        # Minimum detection area (pixels squared)
        self.min_contour_area = rospy.get_param("~min_contour_area", 500)

        # Morphological operation kernel size
        self.morph_kernel_size = rospy.get_param("~morph_kernel_size", 7)

        # RANSAC parameters
        self.ransac_iterations = rospy.get_param("~ransac_iterations", 300)
        self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.03)
        self.min_inlier_ratio = rospy.get_param("~min_inlier_ratio", 0.5)
        self.min_points_for_plane = rospy.get_param("~min_points_for_plane", 50)

        # Debug
        self.publish_debug_image = rospy.get_param("~publish_debug_image", True)

        # ---- State ----
        self.bridge = CvBridge()
        self.latest_cloud = None
        self.cloud_header = None

        # ---- TF ----
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ---- Publishers ----
        self.transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )
        self.debug_image_pub = rospy.Publisher(
            "stand_orientation/debug_image", Image, queue_size=1
        )
        self.mask_pub = rospy.Publisher("stand_orientation/mask", Image, queue_size=1)
        self.valve_mask_pub = rospy.Publisher(
            "stand_orientation/valve_mask", Image, queue_size=1
        )

        # ---- Subscribers ----
        color_topic = rospy.get_param("~color_topic", "/taluy/camera/color/image_raw")
        cloud_topic = rospy.get_param(
            "~cloud_topic", "/taluy/camera/depth/color/points"
        )

        # Continuously update point cloud
        rospy.Subscriber(
            cloud_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
            buff_size=2**24,
        )

        # Run pipeline when color image arrives
        rospy.Subscriber(
            color_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(
            f"Stand orientation estimator ready.\n"
            f"  Color: {color_topic}\n"
            f"  Cloud: {cloud_topic}\n"
            f"  HSV lower: {self.hsv_lower}\n"
            f"  HSV upper: {self.hsv_upper}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def cloud_callback(self, msg):
        """Store the latest point cloud message."""
        self.latest_cloud = msg
        self.cloud_header = msg.header

    def image_callback(self, msg):
        """
        Called for each color frame. Main pipeline entry point.

        Strategy:
        - Yellow panels -> RANSAC -> orientation (surface normal)
        - Orange valve -> 3D centroid -> position
        - If valve not visible -> use panel centroid as fallback
        """
        if self.latest_cloud is None:
            rospy.loginfo_throttle(5.0, "Waiting for point cloud data...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 1) Detect yellow panels of the stand (for orientation)
        mask, contour, bbox = self.detect_stand_panels(frame)

        if contour is None:
            if self.publish_debug_image:
                self.publish_debug(frame, None, None, None, None)
            return

        # 2) Extract yellow panel 3D points from point cloud
        points_3d = self.extract_3d_points(mask)

        if points_3d is None or len(points_3d) < self.min_points_for_plane:
            rospy.logwarn_throttle(
                3.0,
                f"Not enough valid 3D points for plane fit: "
                f"{0 if points_3d is None else len(points_3d)} "
                f"(need {self.min_points_for_plane})",
            )
            if self.publish_debug_image:
                self.publish_debug(frame, contour, None, None, None)
            return

        # 3) Fit plane with RANSAC -> orientation
        result = ransac_plane_fit(
            points_3d,
            max_iterations=self.ransac_iterations,
            distance_threshold=self.ransac_threshold,
        )

        if result is None:
            rospy.logwarn_throttle(3.0, "RANSAC plane fit failed!")
            if self.publish_debug_image:
                self.publish_debug(frame, contour, None, None, None)
            return

        normal, panel_centroid, inlier_ratio = result

        if inlier_ratio < self.min_inlier_ratio:
            rospy.logwarn_throttle(
                3.0, f"Low inlier ratio: {inlier_ratio:.2f} < {self.min_inlier_ratio}"
            )
            return

        # 4) Ensure normal points towards the camera
        if normal[2] > 0:
            normal = -normal

        # 5) Detect orange valve -> position
        valve_contour = self.detect_valve(frame)
        valve_position = None

        if valve_contour is not None:
            valve_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(valve_mask, [valve_contour], -1, 255, -1)
            valve_points = self.extract_3d_points(valve_mask)
            if valve_points is not None and len(valve_points) >= 10:
                valve_position = np.mean(valve_points, axis=0)

        # 6) Position: use valve center if visible, else panel centroid
        position = valve_position if valve_position is not None else panel_centroid
        source = "VALVE" if valve_position is not None else "PANEL"

        # 7) Calculate quaternion from normal vector
        orientation = self.normal_to_quaternion(normal)

        # 8) Publish TF
        self.publish_stand_tf(self.cloud_header, position, orientation)

        rospy.loginfo_throttle(
            2.0,
            f"Stand detected [{source}] — pos: [{position[0]:.2f}, {position[1]:.2f}, "
            f"{position[2]:.2f}] normal: [{normal[0]:.2f}, {normal[1]:.2f}, "
            f"{normal[2]:.2f}] inlier: {inlier_ratio:.1%}",
        )

        if self.publish_debug_image:
            self.publish_debug(frame, contour, position, normal, valve_contour)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_stand_panels(self, frame):
        """
        Detect yellow panels of the stand from color image using HSV masking.

        Returns: (mask, contour, bbox) or (mask, None, None)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological operations - clean noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Publish mask
        if self.mask_pub.get_num_connections() > 0:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            self.mask_pub.publish(mask_msg)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask, None, None

        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_contour_area:
            return mask, None, None

        bbox = cv2.boundingRect(largest)
        return mask, largest, bbox

    def detect_valve(self, frame):
        """
        Detect orange valve using HSV.
        Returns: contour or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.valve_hsv_lower, self.valve_hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Publish valve mask
        if self.valve_mask_pub.get_num_connections() > 0:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            self.valve_mask_pub.publish(mask_msg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_valve_area:
            return None

        return largest

    # ------------------------------------------------------------------
    # 3D Point Extraction
    # ------------------------------------------------------------------

    def extract_3d_points(self, mask):
        """
        Extract 3D points corresponding to white pixels in mask
        from the point cloud.

        If PointCloud2 is organized (H x W), direct pixel mapping is used.
        """
        cloud = self.latest_cloud
        if cloud is None:
            return None

        # Cloud dimensions
        cloud_h = cloud.height
        cloud_w = cloud.width
        mask_h, mask_w = mask.shape[:2]

        # Find white pixels in mask
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        # Scale if there is a resolution difference between mask and cloud
        if mask_w != cloud_w or mask_h != cloud_h:
            scale_x = cloud_w / mask_w
            scale_y = cloud_h / mask_h
            xs = (xs * scale_x).astype(int)
            ys = (ys * scale_y).astype(int)
            xs = np.clip(xs, 0, cloud_w - 1)
            ys = np.clip(ys, 0, cloud_h - 1)

        # Subsample for efficiency (if too many points)
        max_points = 2000
        if len(xs) > max_points:
            indices = np.random.choice(len(xs), max_points, replace=False)
            xs = xs[indices]
            ys = ys[indices]

        # Find field offsets
        field_map = {}
        for field in cloud.fields:
            field_map[field.name] = field.offset

        x_off = field_map.get("x", 0)
        y_off = field_map.get("y", 4)
        z_off = field_map.get("z", 8)
        point_step = cloud.point_step
        row_step = cloud.row_step
        data = cloud.data

        points = []
        for u, v in zip(xs, ys):
            idx = v * row_step + u * point_step
            if idx + z_off + 4 > len(data):
                continue
            x = struct.unpack_from("f", data, idx + x_off)[0]
            y = struct.unpack_from("f", data, idx + y_off)[0]
            z = struct.unpack_from("f", data, idx + z_off)[0]

            # NaN/Inf check
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue
            if math.isinf(x) or math.isinf(y) or math.isinf(z):
                continue
            # Filter very far points (10m+)
            if abs(x) > 10 or abs(y) > 10 or abs(z) > 10:
                continue

            points.append([x, y, z])

        if not points:
            return None

        return np.array(points, dtype=np.float32)

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------

    def normal_to_quaternion(self, normal):
        """
        Calculate quaternion from surface normal vector.

        Convention (compatible with valve_trajectory_publisher):
        - x-axis of valve_stand_link = surface normal direction
        - This TF is then read by valve_trajectory_publisher
          to generate approach/contact frames.

        Since normal points towards camera (-z direction):
        transformation from camera frame to odom frame is handled by TF.
        We publish in camera frame, header.frame_id is camera frame.
        """
        normal = normal / np.linalg.norm(normal)

        # Align x-axis with normal direction
        # Target: x_axis = normal
        x_axis = normal

        # y-axis (perpendicular to up vector): z_world x x_axis
        # In camera frame: y = up, z = forward
        # Use y-axis (0, -1, 0) as 'up' (camera optical frame)
        up = np.array([0, -1, 0], dtype=np.float64)

        y_axis = np.cross(x_axis, up)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            # Normal is parallel to up - use alternative up
            up = np.array([1, 0, 0], dtype=np.float64)
            y_axis = np.cross(x_axis, up)
            y_norm = np.linalg.norm(y_axis)

        y_axis = y_axis / y_norm
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Rotation matrix (x, y, z columns)
        rot = np.eye(4)
        rot[:3, 0] = x_axis
        rot[:3, 1] = y_axis
        rot[:3, 2] = z_axis

        # Matrix -> quaternion
        quat = tft.quaternion_from_matrix(rot)
        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    # ------------------------------------------------------------------
    # TF Publishing
    # ------------------------------------------------------------------

    def publish_stand_tf(self, header, centroid, orientation):
        """
        Publish the valve_stand_link TF frame.

        header.frame_id = camera depth optical frame
        centroid = 3D position in camera frame
        orientation = quaternion calculated from surface normal
        """
        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = header.frame_id
        t.child_frame_id = "valve_stand_link"
        t.transform.translation = Vector3(
            x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])
        )
        t.transform.rotation = orientation

        # TF broadcast
        self.tf_broadcaster.sendTransform(t)

        # Publish to topic as well (for object_map_tf_server)
        self.transform_pub.publish(t)

    # ------------------------------------------------------------------
    # Debug Visualization
    # ------------------------------------------------------------------

    def publish_debug(self, frame, contour, centroid, normal, valve_contour):
        """Publish debug image."""
        debug = frame.copy()

        # Yellow panel contour (green)
        if contour is not None:
            cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)

        # Orange valve contour (orange/red)
        if valve_contour is not None:
            cv2.drawContours(debug, [valve_contour], -1, (0, 128, 255), 3)
            M = cv2.moments(valve_contour)
            if M["m00"] > 0:
                vcx = int(M["m10"] / M["m00"])
                vcy = int(M["m01"] / M["m00"])
                cv2.circle(debug, (vcx, vcy), 8, (0, 0, 255), -1)
                cv2.putText(
                    debug,
                    "VALVE",
                    (vcx + 12, vcy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 128, 255),
                    2,
                )

        if centroid is not None:
            text = f"Pos: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
            cv2.putText(
                debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        if normal is not None:
            text = f"Normal: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            cv2.putText(
                debug, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            # Draw normal direction as arrow (2D projection)
            if contour is not None:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    arrow_len = 80
                    dx = int(normal[0] * arrow_len)
                    dy = int(normal[1] * arrow_len)
                    cv2.arrowedLine(
                        debug,
                        (cx, cy),
                        (cx + dx, cy + dy),
                        (0, 0, 255),
                        3,
                        tipLength=0.3,
                    )
                    cv2.circle(debug, (cx, cy), 5, (0, 255, 0), -1)

        # Source info
        source = "VALVE" if valve_contour is not None else "PANEL"
        status = f"DETECTED [{source}]" if centroid is not None else "SEARCHING..."
        color = (0, 255, 0) if centroid is not None else (0, 0, 255)
        cv2.putText(
            debug,
            status,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        self.debug_image_pub.publish(debug_msg)


def main():
    node = StandOrientationEstimator()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
