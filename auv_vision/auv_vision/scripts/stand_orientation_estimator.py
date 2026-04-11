#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stand Orientation Estimator Node
---------------------------------
Estimates the 3D position of the valve from a YoloResult bounding box
and the stand surface orientation from RealSense point cloud + RANSAC.

Position pipeline (from front camera):
  1. YoloResult bbox (sim_bbox_node or real YOLO) gives valve pixel location
  2. Known valve diameter + camera intrinsics -> 3D position via pinhole model
  3. Published as valve_stand_link TF

Orientation pipeline (from RealSense, best-effort):
  1. Transform valve 3D position into RealSense point cloud frame
  2. Extract nearby 3D points (stand surface region)
  3. RANSAC plane fit -> surface normal
  4. Cache orientation for use in TF publication

Subscribe:
  - /yolo_result_front                       (ultralytics_ros/YoloResult)
  - /<ns>/cameras/cam_front/camera_info      (sensor_msgs/CameraInfo)
  - /<ns>/camera/depth/color/points          (sensor_msgs/PointCloud2)

Publish:
  - valve_stand_link TF frame
  - object_transform_updates                 (geometry_msgs/TransformStamped)
"""

import rospy
import numpy as np
import struct
import math

from sensor_msgs.msg import CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
import tf2_ros
import tf.transformations as tft

from ultralytics_ros.msg import YoloResult


def ransac_plane_fit(points_3d, max_iterations=300, distance_threshold=0.03):
    """
    Find the best fitting plane using RANSAC.

    Returns:
        (normal, centroid, inlier_ratio) or None
    """
    if len(points_3d) < 3:
        return None

    best_normal = None
    best_centroid = None
    best_inlier_count = 0
    n_points = len(points_3d)

    for _ in range(max_iterations):
        indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points_3d[indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            continue

        normal = normal / norm

        diffs = points_3d - p1
        distances = np.abs(np.dot(diffs, normal))

        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            inlier_points = points_3d[inliers]
            centroid = np.mean(inlier_points, axis=0)
            # PCA for more accurate normal from inliers
            centered = inlier_points - centroid
            cov_matrix = np.dot(centered.T, centered) / len(inlier_points)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            best_normal = eigenvectors[:, 0]  # smallest eigenvalue = normal
            best_centroid = centroid

    if best_normal is None:
        return None

    inlier_ratio = best_inlier_count / n_points
    return best_normal, best_centroid, inlier_ratio


class StandOrientationEstimator:
    """
    ROS node that estimates valve stand position from YoloResult bbox
    and stand orientation from RealSense point cloud RANSAC.
    """

    def __init__(self):
        rospy.init_node("stand_orientation_estimator", anonymous=True)
        rospy.loginfo("Stand orientation estimator node starting...")

        # Valve position source: YoloResult bbox (sim_bbox_node or real YOLO)
        self.bbox_topic = rospy.get_param("~bbox_topic", "/yolo_result_front")
        self.valve_class_id = rospy.get_param("~valve_class_id", 8)

        # Front camera intrinsics - for geometric 3D estimation from bbox
        ns = rospy.get_param("~namespace", "taluy")
        self.front_optical_frame = rospy.get_param(
            "~front_optical_frame",
            f"{ns}/base_link/front_camera_optical_link_stabilized",
        )
        self.valve_diameter = rospy.get_param("~valve_diameter", 0.24)  # metres

        # RealSense point cloud - for RANSAC orientation
        self.realsense_optical_frame = rospy.get_param(
            "~realsense_optical_frame",
            f"{ns}/camera_depth_optical_frame",
        )

        # RANSAC parameters
        self.ransac_iterations = rospy.get_param("~ransac_iterations", 300)
        self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.03)
        self.min_inlier_ratio = rospy.get_param("~min_inlier_ratio", 0.3)
        self.min_points_for_plane = rospy.get_param("~min_points_for_plane", 50)
        # Search radius around valve position to extract stand surface points
        self.search_radius = rospy.get_param("~search_radius", 0.5)
        # Exclusion radius around valve center — skip the valve itself
        # (valve is circular ~0.24m diameter, exclude a bit more to be safe)
        self.valve_exclusion_radius = rospy.get_param("~valve_exclusion_radius", 0.18)

        # ---- State ----
        self.front_K = None
        self.latest_valve_bbox_center = None
        self.latest_valve_bbox_size = None
        self.latest_cloud = None
        self.cached_orientation = None  # last good RANSAC orientation
        self.latest_valve_position = (
            None  # last known 3D valve pos (front optical frame)
        )

        # Default orientation: surface normal toward camera (-Z in optical frame)
        self.default_orientation = self._normal_to_quaternion(
            np.array([0.0, 0.0, -1.0])
        )

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ---- Publishers ----
        self.transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )

        # ---- Subscribers ----
        front_info_topic = rospy.get_param(
            "~front_camera_info_topic",
            f"/{ns}/cameras/cam_front/camera_info",
        )
        rospy.Subscriber(
            front_info_topic, CameraInfo, self._front_info_cb, queue_size=1
        )

        # YoloResult: valve bbox
        rospy.Subscriber(self.bbox_topic, YoloResult, self._bbox_callback, queue_size=1)

        # RealSense point cloud: for RANSAC orientation
        cloud_topic = rospy.get_param(
            "~cloud_topic", f"/{ns}/camera/depth/color/points"
        )
        rospy.Subscriber(
            cloud_topic,
            PointCloud2,
            self._cloud_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(
            f"Stand orientation estimator ready.\n"
            f"  Front info:    {front_info_topic}\n"
            f"  Front frame:   {self.front_optical_frame}\n"
            f"  BBox topic:    {self.bbox_topic} (class_id={self.valve_class_id})\n"
            f"  Cloud topic:   {cloud_topic}\n"
            f"  RS frame:      {self.realsense_optical_frame}\n"
            f"  Search radius: {self.search_radius}m\n"
            f"  Valve diameter: {self.valve_diameter}m"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _front_info_cb(self, msg):
        """Cache front camera K matrix (once)."""
        if self.front_K is None:
            self.front_K = list(msg.K)
            rospy.loginfo("Front camera intrinsics received.")

    def _cloud_callback(self, msg):
        """Store latest point cloud and attempt RANSAC orientation update."""
        self.latest_cloud = msg
        self._update_orientation_from_cloud(msg)

    def _bbox_callback(self, msg):
        """
        Extract valve bbox from YoloResult, estimate 3D position,
        and publish valve_stand_link TF.
        """
        for det in msg.detections.detections:
            if det.results and det.results[0].id == self.valve_class_id:
                self.latest_valve_bbox_center = (
                    det.bbox.center.x,
                    det.bbox.center.y,
                )
                self.latest_valve_bbox_size = (det.bbox.size_x, det.bbox.size_y)
                break
        else:
            return

        position = self._estimate_3d_from_bbox()
        if position is None:
            return

        self.latest_valve_position = position

        # Use cached RANSAC orientation if available, otherwise default
        orientation = self.cached_orientation or self.default_orientation

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.front_optical_frame
        t.child_frame_id = "valve_stand_link"
        t.transform.translation = Vector3(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        )
        t.transform.rotation = orientation
        self.tf_broadcaster.sendTransform(t)
        self.transform_pub.publish(t)

        source = "RANSAC" if self.cached_orientation else "DEFAULT"
        rospy.loginfo_throttle(
            2.0,
            f"valve_stand_link [BBOX+{source}] "
            f"pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]",
        )

    # ------------------------------------------------------------------
    # Position Estimation (from YOLO bbox)
    # ------------------------------------------------------------------

    def _estimate_3d_from_bbox(self):
        """
        Estimate 3D valve position from bbox + known valve diameter.
        Returns: np.array([x, y, z]) in front camera optical frame, or None.
        """
        if (
            self.latest_valve_bbox_center is None
            or self.latest_valve_bbox_size is None
            or self.front_K is None
        ):
            return None

        cx, cy = self.latest_valve_bbox_center
        bw, bh = self.latest_valve_bbox_size
        bbox_size = max(bw, bh)
        if bbox_size < 1.0:
            return None

        fx = self.front_K[0]
        fy = self.front_K[4]
        K_cx = self.front_K[2]
        K_cy = self.front_K[5]

        distance = fx * self.valve_diameter / bbox_size
        x = (cx - K_cx) * distance / fx
        y = (cy - K_cy) * distance / fy
        z = distance

        return np.array([x, y, z], dtype=np.float32)

    # ------------------------------------------------------------------
    # Orientation Estimation (from RealSense point cloud RANSAC)
    # ------------------------------------------------------------------

    def _update_orientation_from_cloud(self, cloud_msg):
        """
        Best-effort orientation update from RealSense point cloud.

        1. Transform valve position to point cloud frame
        2. Extract nearby 3D points (stand surface)
        3. RANSAC plane fit -> surface normal
        4. Cache orientation
        """
        if self.latest_valve_position is None:
            return

        # Transform valve position from front optical frame to RS optical frame
        valve_in_rs = self._transform_point_to_rs_frame(
            self.latest_valve_position, cloud_msg.header.stamp
        )
        if valve_in_rs is None:
            return

        # Extract points near the valve from the organized point cloud
        points_3d = self._extract_nearby_points(cloud_msg, valve_in_rs)
        if points_3d is None or len(points_3d) < self.min_points_for_plane:
            rospy.logdebug_throttle(
                5.0,
                f"Not enough points for RANSAC: {0 if points_3d is None else len(points_3d)}",
            )
            return

        result = ransac_plane_fit(
            points_3d,
            max_iterations=self.ransac_iterations,
            distance_threshold=self.ransac_threshold,
        )
        if result is None:
            return

        normal, _, inlier_ratio = result
        if inlier_ratio < self.min_inlier_ratio:
            rospy.logdebug_throttle(
                5.0, f"RANSAC inlier ratio too low: {inlier_ratio:.1%}"
            )
            return

        # Ensure normal points toward camera (-Z in RS optical frame)
        if normal[2] > 0:
            normal = -normal

        # Transform normal from RS optical frame to front optical frame
        normal_in_front = self._transform_normal_to_front_frame(
            normal, cloud_msg.header.stamp
        )
        if normal_in_front is None:
            return

        self.cached_orientation = self._normal_to_quaternion(normal_in_front)
        rospy.loginfo_throttle(
            5.0,
            f"RANSAC orientation updated — "
            f"normal(front): [{normal_in_front[0]:.2f}, {normal_in_front[1]:.2f}, {normal_in_front[2]:.2f}] "
            f"inlier: {inlier_ratio:.1%} points: {len(points_3d)}",
        )

    def _transform_point_to_rs_frame(self, point_in_front, stamp):
        """
        Transform a 3D point from front camera optical frame to
        RealSense depth optical frame.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.realsense_optical_frame,
                self.front_optical_frame,
                stamp,
                rospy.Duration(0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0,
                f"TF lookup failed ({self.front_optical_frame} -> "
                f"{self.realsense_optical_frame}): {e}",
            )
            return None

        q = tf_stamped.transform.rotation
        t = tf_stamped.transform.translation
        rot_matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        translation = np.array([t.x, t.y, t.z])

        return rot_matrix @ point_in_front + translation

    def _transform_normal_to_front_frame(self, normal, stamp):
        """
        Rotate a normal vector from RealSense depth optical frame
        to front camera optical frame.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.front_optical_frame,
                self.realsense_optical_frame,
                stamp,
                rospy.Duration(0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0,
                f"TF lookup failed ({self.realsense_optical_frame} -> "
                f"{self.front_optical_frame}): {e}",
            )
            return None

        q = tf_stamped.transform.rotation
        rot_matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        normal_transformed = rot_matrix @ normal
        norm = np.linalg.norm(normal_transformed)
        if norm < 1e-8:
            return None
        return normal_transformed / norm

    def _extract_nearby_points(self, cloud_msg, center_point):
        """
        Extract 3D points from organized point cloud that are within
        search_radius of center_point.

        Uses the organized structure (H x W) for efficient access:
        projects center_point to pixel, then searches in a pixel window.
        """
        cloud_h = cloud_msg.height
        cloud_w = cloud_msg.width

        if cloud_h <= 1:
            # Unorganized cloud — brute force search
            return self._extract_nearby_points_unorganized(cloud_msg, center_point)

        # Find field offsets
        field_map = {}
        for field in cloud_msg.fields:
            field_map[field.name] = field.offset
        x_off = field_map.get("x", 0)
        y_off = field_map.get("y", 4)
        z_off = field_map.get("z", 8)
        point_step = cloud_msg.point_step
        row_step = cloud_msg.row_step
        data = cloud_msg.data

        # Project center point to approximate pixel coordinates
        # In depth optical frame: x=right, y=down, z=forward
        if abs(center_point[2]) < 0.01:
            return None

        # Use point cloud's implicit intrinsics (approximate)
        # For organized clouds, u ~ x/z * fx + cx, v ~ y/z * fy + cy
        # We estimate the search window in pixels from the search radius
        depth = center_point[2]
        # Approximate: 1m at depth d covers roughly (cloud_w * 1.0/d) / (2*tan(fov/2)) pixels
        # For RealSense D435 hfov ~85deg: tan(42.5deg) ~ 0.916
        # pixels_per_meter ~ fx / depth ~ (cloud_w / (2 * 0.916)) / depth
        pixels_per_meter_approx = (cloud_w / 1.832) / depth
        search_pixels = int(self.search_radius * pixels_per_meter_approx) + 10

        # Approximate pixel center
        u_center = int(center_point[0] / depth * (cloud_w / 1.832) + cloud_w / 2)
        v_center = int(center_point[1] / depth * (cloud_h / 1.38) + cloud_h / 2)

        # Clamp to image bounds
        u_min = max(0, u_center - search_pixels)
        u_max = min(cloud_w, u_center + search_pixels)
        v_min = max(0, v_center - search_pixels)
        v_max = min(cloud_h, v_center + search_pixels)

        if u_min >= u_max or v_min >= v_max:
            return None

        radius_sq = self.search_radius**2
        exclusion_sq = self.valve_exclusion_radius**2
        points = []

        # Subsample: step by 2 for efficiency
        step = 2 if (u_max - u_min) * (v_max - v_min) > 5000 else 1

        for v in range(v_min, v_max, step):
            for u in range(u_min, u_max, step):
                idx = v * row_step + u * point_step
                if idx + z_off + 4 > len(data):
                    continue

                px = struct.unpack_from("f", data, idx + x_off)[0]
                py = struct.unpack_from("f", data, idx + y_off)[0]
                pz = struct.unpack_from("f", data, idx + z_off)[0]

                if math.isnan(px) or math.isnan(py) or math.isnan(pz):
                    continue
                if math.isinf(px) or math.isinf(py) or math.isinf(pz):
                    continue

                dx = px - center_point[0]
                dy = py - center_point[1]
                dz = pz - center_point[2]
                dist_sq = dx * dx + dy * dy + dz * dz

                # Include if within search radius but OUTSIDE valve exclusion zone
                # This keeps stand surface points and skips the valve body
                if dist_sq <= radius_sq and dist_sq >= exclusion_sq:
                    points.append([px, py, pz])

        if not points:
            return None

        # Limit to max 2000 points for RANSAC efficiency
        pts = np.array(points, dtype=np.float32)
        if len(pts) > 2000:
            indices = np.random.choice(len(pts), 2000, replace=False)
            pts = pts[indices]

        return pts

    def _extract_nearby_points_unorganized(self, cloud_msg, center_point):
        """Fallback for unorganized point clouds."""
        field_map = {}
        for field in cloud_msg.fields:
            field_map[field.name] = field.offset
        x_off = field_map.get("x", 0)
        y_off = field_map.get("y", 4)
        z_off = field_map.get("z", 8)
        point_step = cloud_msg.point_step
        data = cloud_msg.data
        n_points = cloud_msg.width * cloud_msg.height

        radius_sq = self.search_radius**2
        exclusion_sq = self.valve_exclusion_radius**2
        points = []

        # Subsample for large clouds
        step = max(1, n_points // 20000)

        for i in range(0, n_points, step):
            idx = i * point_step
            if idx + z_off + 4 > len(data):
                break

            px = struct.unpack_from("f", data, idx + x_off)[0]
            py = struct.unpack_from("f", data, idx + y_off)[0]
            pz = struct.unpack_from("f", data, idx + z_off)[0]

            if math.isnan(px) or math.isnan(py) or math.isnan(pz):
                continue

            dx = px - center_point[0]
            dy = py - center_point[1]
            dz = pz - center_point[2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq <= radius_sq and dist_sq >= exclusion_sq:
                points.append([px, py, pz])

        if not points:
            return None

        pts = np.array(points, dtype=np.float32)
        if len(pts) > 2000:
            indices = np.random.choice(len(pts), 2000, replace=False)
            pts = pts[indices]

        return pts

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_to_quaternion(normal):
        """
        Quaternion from surface normal.
        Convention: x-axis of valve_stand_link = surface normal direction
        (compatible with valve_trajectory_publisher).
        """
        normal = normal / np.linalg.norm(normal)
        x_axis = normal

        up = np.array([0, -1, 0], dtype=np.float64)
        y_axis = np.cross(x_axis, up)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            up = np.array([1, 0, 0], dtype=np.float64)
            y_axis = np.cross(x_axis, up)
            y_norm = np.linalg.norm(y_axis)

        y_axis = y_axis / y_norm
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        rot = np.eye(4)
        rot[:3, 0] = x_axis
        rot[:3, 1] = y_axis
        rot[:3, 2] = z_axis

        quat = tft.quaternion_from_matrix(rot)
        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


def main():
    node = StandOrientationEstimator()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
