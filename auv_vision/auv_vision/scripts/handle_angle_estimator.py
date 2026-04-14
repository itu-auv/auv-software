#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Handle Angle Estimator
-----------------------
Estimates valve handle rotation angle using 3D point-cloud protrusion + PCA.

Pipeline:
  1. Subscribe to RealSense point cloud + valve_stand_link TF.
  2. Transform each cloud point into the valve frame.
     (valve_stand_link x-axis = surface normal; y,z span the valve plane)
  3. Filter handle points:
       - yz radius < bucket_radius (close to valve center)
       - protrusion_min < x < protrusion_max (sticks out of valve face)
  4. Project to (y, z) plane, run PCA to get principal direction.
  5. Compute angle in valve frame (atan2 of principal axis).
  6. Median-filter over a sliding window of recent frames.
  7. Publish angle (Float32) + debug overlay image on front camera.

Topics:
  subscribe:
    ~cloud_topic                (sensor_msgs/PointCloud2)
    ~front_camera_info_topic    (sensor_msgs/CameraInfo)
    ~front_image_topic          (sensor_msgs/Image)
  publish:
    valve_handle_angle          (std_msgs/Float32)       — radians, [0, pi)
    valve_handle_debug/image    (sensor_msgs/Image)      — BGR overlay
"""

import math

import cv2
import numpy as np
import rospy
import tf.transformations as tft
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Float32


class HandleAngleEstimator:
    def __init__(self):
        rospy.init_node("handle_angle_estimator", anonymous=True)

        ns = rospy.get_param("~namespace", "taluy")
        self.valve_frame = rospy.get_param("~valve_frame", "valve_stand_link")

        # Geometry thresholds (m)
        self.bucket_radius = rospy.get_param("~bucket_radius", 0.07)
        self.protrusion_min = rospy.get_param("~protrusion_min", 0.02)
        self.protrusion_max = rospy.get_param("~protrusion_max", 0.08)
        self.min_handle_points = rospy.get_param("~min_handle_points", 30)
        self.median_window = rospy.get_param("~median_window", 15)
        self.debug_max_points = rospy.get_param("~debug_max_points", 300)
        self.debug_rate_hz = rospy.get_param("~debug_rate_hz", 10.0)
        self._last_debug_stamp = rospy.Time(0)

        self.front_optical_frame = rospy.get_param(
            "~front_optical_frame",
            f"{ns}/base_link/front_camera_optical_link_stabilized",
        )

        # State
        self.front_K = None
        self.latest_front_image = None
        self.angle_history = []  # values in [0, pi)
        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.angle_pub = rospy.Publisher(
            "valve_handle_angle", Float32, queue_size=1
        )
        self.debug_img_pub = rospy.Publisher(
            "valve_handle_debug/image", Image, queue_size=1
        )

        # Subscribers
        cloud_topic = rospy.get_param(
            "~cloud_topic", f"/{ns}/camera/depth/color/points"
        )
        front_info_topic = rospy.get_param(
            "~front_camera_info_topic",
            f"/{ns}/cameras/cam_front/camera_info",
        )
        front_image_topic = rospy.get_param(
            "~front_image_topic",
            f"/{ns}/cameras/cam_front/image_raw",
        )

        rospy.Subscriber(
            cloud_topic,
            PointCloud2,
            self._cloud_cb,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            front_info_topic, CameraInfo, self._front_info_cb, queue_size=1
        )
        rospy.Subscriber(
            front_image_topic, Image, self._front_image_cb, queue_size=1
        )

        rospy.loginfo(
            "Handle angle estimator ready.\n"
            f"  Cloud:           {cloud_topic}\n"
            f"  Valve frame:     {self.valve_frame}\n"
            f"  Front image:     {front_image_topic}\n"
            f"  Bucket radius:   {self.bucket_radius} m\n"
            f"  Protrusion band: [{self.protrusion_min}, {self.protrusion_max}] m\n"
            f"  Median window:   {self.median_window}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _front_info_cb(self, msg):
        if self.front_K is None:
            self.front_K = list(msg.K)

    def _front_image_cb(self, msg):
        self.latest_front_image = msg

    def _cloud_cb(self, cloud_msg):
        # Transform cloud_frame -> valve_frame
        try:
            tf_c2v = self.tf_buffer.lookup_transform(
                self.valve_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                rospy.Duration(0.3),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logdebug_throttle(2.0, f"TF lookup failed: {e}")
            return

        R, trans = self._tf_to_matrix(tf_c2v)

        # Extract Nx3 point array from cloud (NaN-filtered)
        pts_cam = self._cloud_to_numpy(cloud_msg)
        if pts_cam is None or len(pts_cam) < self.min_handle_points:
            return

        # Transform to valve frame
        pts_valve = (R @ pts_cam.T).T + trans

        # Filter: near bucket + protrusion band
        yz_r2 = pts_valve[:, 1] ** 2 + pts_valve[:, 2] ** 2
        x = pts_valve[:, 0]
        mask = (
            (yz_r2 < self.bucket_radius**2)
            & (x > self.protrusion_min)
            & (x < self.protrusion_max)
        )
        handle_pts = pts_valve[mask]

        if len(handle_pts) < self.min_handle_points:
            rospy.logdebug_throttle(
                2.0, f"Not enough handle points: {len(handle_pts)}"
            )
            return

        # PCA on (y, z) projection
        yz = handle_pts[:, 1:3]
        centroid = np.mean(yz, axis=0)
        centered = yz - centroid
        cov = (centered.T @ centered) / len(yz)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, 1]  # largest eigenvalue -> principal axis

        # Angle in valve plane, normalize to [0, pi) (handle is 180° symmetric)
        raw_angle = math.atan2(principal[1], principal[0]) % math.pi

        self.angle_history.append(raw_angle)
        if len(self.angle_history) > self.median_window:
            self.angle_history.pop(0)

        median_angle = self._circular_median_pi(self.angle_history)
        self.angle_pub.publish(Float32(data=median_angle))

        rospy.loginfo_throttle(
            1.0,
            f"Handle angle: {math.degrees(median_angle):6.1f}° "
            f"(raw {math.degrees(raw_angle):6.1f}°, "
            f"n={len(handle_pts)}, window={len(self.angle_history)})",
        )

        self._publish_debug_image(
            cloud_stamp=cloud_msg.header.stamp,
            handle_pts_valve=handle_pts,
            centroid_yz=centroid,
            principal_yz=principal,
            median_angle=median_angle,
            raw_angle=raw_angle,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tf_to_matrix(tf_stamped):
        q = tf_stamped.transform.rotation
        t = tf_stamped.transform.translation
        T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        R = T[:3, :3]
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return R.astype(np.float32), trans

    @staticmethod
    def _cloud_to_numpy(cloud_msg):
        """Efficient Nx3 xyz extraction from a PointCloud2 message."""
        field_map = {f.name: f.offset for f in cloud_msg.fields}
        x_off = field_map.get("x", 0)
        y_off = field_map.get("y", 4)
        z_off = field_map.get("z", 8)

        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [x_off, y_off, z_off],
                "itemsize": cloud_msg.point_step,
            }
        )

        try:
            raw = np.frombuffer(cloud_msg.data, dtype=dtype)
        except ValueError:
            return None

        pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float32)
        valid = np.isfinite(pts).all(axis=1)
        return pts[valid]

    @staticmethod
    def _circular_median_pi(angles):
        """
        Median over angles in [0, pi). Because the space is modulo pi,
        we double angles to [0, 2pi), take circular mean/median via
        unit vectors, and halve back.
        """
        a = np.asarray(angles) * 2.0
        x = np.median(np.cos(a))
        y = np.median(np.sin(a))
        m = math.atan2(y, x) / 2.0
        if m < 0:
            m += math.pi
        return m

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------

    def _publish_debug_image(
        self,
        cloud_stamp,
        handle_pts_valve,
        centroid_yz,
        principal_yz,
        median_angle,
        raw_angle,
    ):
        if self.latest_front_image is None or self.front_K is None:
            return

        # Skip entirely if nobody is listening — this callback is the hot path.
        if self.debug_img_pub.get_num_connections() == 0:
            return

        # Throttle debug publishing independent of cloud rate.
        if self.debug_rate_hz > 0.0:
            min_dt = rospy.Duration(1.0 / self.debug_rate_hz)
            now = rospy.Time.now()
            if (now - self._last_debug_stamp) < min_dt:
                return
            self._last_debug_stamp = now

        try:
            tf_v2f = self.tf_buffer.lookup_transform(
                self.front_optical_frame,
                self.valve_frame,
                cloud_stamp,
                rospy.Duration(0.2),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return

        R, trans = self._tf_to_matrix(tf_v2f)

        try:
            img = self.bridge.imgmsg_to_cv2(self.latest_front_image, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"cv_bridge convert failed: {e}")
            return

        fx = self.front_K[0]
        fy = self.front_K[4]
        cx = self.front_K[2]
        cy = self.front_K[5]

        def project(pt_front):
            if pt_front[2] <= 0.01:
                return None
            u = int(fx * pt_front[0] / pt_front[2] + cx)
            v = int(fy * pt_front[1] / pt_front[2] + cy)
            return u, v

        # Handle points (valve->front) in green — vectorized projection + draw.
        if len(handle_pts_valve) > 0:
            pts_draw = handle_pts_valve
            if (
                self.debug_max_points > 0
                and len(pts_draw) > self.debug_max_points
            ):
                idx = np.random.choice(
                    len(pts_draw), self.debug_max_points, replace=False
                )
                pts_draw = pts_draw[idx]
            pts_front = (R @ pts_draw.T).T + trans
            zc = pts_front[:, 2]
            valid = zc > 0.01
            if np.any(valid):
                pf = pts_front[valid]
                u = (fx * pf[:, 0] / pf[:, 2] + cx).astype(np.int32)
                v = (fy * pf[:, 1] / pf[:, 2] + cy).astype(np.int32)
                h, w = img.shape[:2]
                in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                u = u[in_bounds]
                v = v[in_bounds]
                img[v, u] = (0, 255, 0)
                if u.size:
                    img[np.clip(v + 1, 0, h - 1), u] = (0, 255, 0)
                    img[v, np.clip(u + 1, 0, w - 1)] = (0, 255, 0)

        # Draw line at the actual handle protrusion height (median x of points)
        # so the projected line overlaps the handle visually instead of the valve face.
        handle_x = float(np.median(handle_pts_valve[:, 0]))
        half_len = 0.075  # 15 cm line, slightly longer than handle for visibility
        centroid_valve = np.array(
            [handle_x, centroid_yz[0], centroid_yz[1]], dtype=np.float32
        )
        principal_valve = np.array(
            [0.0, principal_yz[0], principal_yz[1]], dtype=np.float32
        )
        p1_valve = centroid_valve - half_len * principal_valve
        p2_valve = centroid_valve + half_len * principal_valve
        # Valve face center (on the surface, x=0)
        origin_valve = np.zeros(3, dtype=np.float32)

        p1_front = R @ p1_valve + trans
        p2_front = R @ p2_valve + trans
        o_front = R @ origin_valve + trans

        uv1 = project(p1_front)
        uv2 = project(p2_front)
        uvo = project(o_front)

        # Handle axis line — bright red, thick
        if uv1 and uv2:
            cv2.line(img, uv1, uv2, (0, 0, 255), 4, cv2.LINE_AA)
            # Endpoint dots so the line looks like a dumbbell — easier to read
            cv2.circle(img, uv1, 6, (0, 0, 255), -1)
            cv2.circle(img, uv2, 6, (0, 0, 255), -1)

        # Valve face center marker
        if uvo:
            cv2.circle(img, uvo, 8, (255, 0, 0), 2)
            cv2.drawMarker(
                img, uvo, (255, 0, 0), cv2.MARKER_CROSS, 12, 2
            )

        # Compass: show valve y and z axes at top-right corner
        # so the operator can see which direction "0°" points to.
        self._draw_compass(img, R, trans, project)

        # Text overlay with dark background for readability
        lines = [
            f"Handle: {math.degrees(median_angle):5.1f} deg "
            f"(raw {math.degrees(raw_angle):5.1f})",
            f"Pts: {len(handle_pts_valve)}  window: {len(self.angle_history)}",
        ]
        for i, text in enumerate(lines):
            y = 30 + i * 28
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(img, (8, y - th - 4), (12 + tw, y + 6), (0, 0, 0), -1)
            cv2.putText(
                img,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        debug_msg.header = self.latest_front_image.header
        self.debug_img_pub.publish(debug_msg)

    def _draw_compass(self, img, R, trans, project):
        """
        Draw small valve-frame y/z axes near top-right of the image.
        Anchored at the projected valve center but offset by a fixed valve-frame
        translation so it sits in the corner without overlapping the handle.
        """
        # Anchor compass 12cm above and 12cm right of valve center, on valve face.
        anchor_valve = np.array([0.0, 0.12, -0.12], dtype=np.float32)
        axis_len = 0.05  # 5 cm visual axes

        anchor_front = R @ anchor_valve + trans
        y_tip_front = R @ (anchor_valve + np.array([0.0, axis_len, 0.0])) + trans
        z_tip_front = R @ (anchor_valve + np.array([0.0, 0.0, axis_len])) + trans

        uv_a = project(anchor_front)
        uv_y = project(y_tip_front)
        uv_z = project(z_tip_front)
        if not (uv_a and uv_y and uv_z):
            return

        # +y axis (yellow) — angle 0° reference
        cv2.arrowedLine(img, uv_a, uv_y, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(
            img, "y(0)", (uv_y[0] + 4, uv_y[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
        # +z axis (cyan) — angle 90° reference
        cv2.arrowedLine(img, uv_a, uv_z, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(
            img, "z(90)", (uv_z[0] + 4, uv_z[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
        )


def main():
    HandleAngleEstimator()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
