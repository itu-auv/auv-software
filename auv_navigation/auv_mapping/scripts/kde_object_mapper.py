#!/usr/bin/env python3
"""KDE Object Mapper Node

Performs Kernel Density Estimation on accumulated detection points to find
the strongest spatial clusters for each object class. Publishes:
- Filtered object poses via set_object_transform service
- TF frames for each detected peak
- Compressed image visualization of all class distributions
"""

import threading
import rospy
import numpy as np
import cv2
from collections import defaultdict
from scipy.stats import gaussian_kde

from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


# ──────────────────────────────────────────────────────────────────────────────
# Color palette (BGR) for per-class visualization
# ──────────────────────────────────────────────────────────────────────────────
CLASS_COLORS_BGR = [
    (80, 80, 255),    # soft red
    (80, 220, 80),    # green
    (255, 180, 50),   # blue-ish
    (180, 50, 255),   # magenta
    (220, 220, 50),   # cyan
    (50, 140, 255),   # orange
    (200, 100, 200),  # purple
    (100, 255, 255),  # yellow
    (255, 150, 150),  # light blue
    (150, 255, 150),  # light green
]


class KdeObjectMapper:
    def __init__(self):
        rospy.init_node("kde_object_mapper")

        # ── Parameters ────────────────────────────────────────────────────
        self.bandwidth = rospy.get_param("~bandwidth", 0.2)
        self.min_peak_density = rospy.get_param("~min_peak_density", 0.01)
        self.max_peaks_per_class = rospy.get_param("~max_peaks_per_class", 5)
        self.suppression_radius = rospy.get_param("~suppression_radius", 0.5)
        self.update_rate = rospy.get_param("~update_rate", 2.0)
        self.static_frame = rospy.get_param("~static_frame", "odom")
        self.grid_resolution = rospy.get_param("~grid_resolution", 0.05)
        self.min_points_for_kde = rospy.get_param("~min_points_for_kde", 5)
        self.max_points_per_class = rospy.get_param("~max_points_per_class", 2000)
        self.frame_suffix = rospy.get_param("~frame_suffix", "_kde")
        self.image_width = rospy.get_param("~image_width", 800)
        self.image_height = rospy.get_param("~image_height", 600)
        object_classes = rospy.get_param("~object_classes", [])

        if not object_classes:
            rospy.logwarn("No object_classes configured — KDE mapper has nothing to do.")

        # ── State ─────────────────────────────────────────────────────────
        self.point_buffers = defaultdict(list)  # class_name -> [[x,y,z], ...]
        self.current_results = {}  # class_name -> [(x, y, z, confidence), ...]
        self.lock = threading.Lock()
        self._kde_running = False  # re-entrancy guard

        # Assign a stable colour to each class
        self.class_colors = {}
        for i, cls in enumerate(object_classes):
            self.class_colors[cls] = CLASS_COLORS_BGR[i % len(CLASS_COLORS_BGR)]

        # ── Subscribers (one per class) ───────────────────────────────────
        self.subscribers = {}
        for cls_name in object_classes:
            topic = f"kde_map/points/{cls_name}"
            self.subscribers[cls_name] = rospy.Subscriber(
                topic,
                PointStamped,
                lambda msg, c=cls_name: self._point_callback(msg, c),
                queue_size=50,
            )
            rospy.loginfo(f"Subscribed to {topic}")

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub = rospy.Publisher(
            "kde_map/visualization/compressed", CompressedImage, queue_size=1
        )

        # ── Service proxy ─────────────────────────────────────────────────
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        # ── Services ─────────────────────────────────────────────────────
        rospy.Service("kde_map/clear", Trigger, self._clear_callback)
        rospy.Service("kde_map/trigger_update", Trigger, self._trigger_callback)

        # ── Timer ─────────────────────────────────────────────────────────
        self.update_timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate), self._update_callback
        )

        rospy.loginfo(
            f"KDE Object Mapper ready — {len(object_classes)} classes, "
            f"bw={self.bandwidth}m, rate={self.update_rate}Hz"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════

    def _point_callback(self, msg, class_name):
        with self.lock:
            buf = self.point_buffers[class_name]
            buf.append([msg.point.x, msg.point.y, msg.point.z])
            # Keep buffer bounded
            if len(buf) > self.max_points_per_class:
                del buf[: len(buf) - self.max_points_per_class]

    def _clear_callback(self, _req):
        with self.lock:
            self.point_buffers.clear()
            self.current_results.clear()
        rospy.loginfo("KDE: cleared all point buffers.")
        return TriggerResponse(success=True, message="Cleared all KDE buffers")

    def _trigger_callback(self, _req):
        self._run_kde()
        return TriggerResponse(success=True, message="KDE update triggered")

    def _update_callback(self, _event):
        # Re-entrancy guard: skip if previous KDE computation is still running
        with self.lock:
            if self._kde_running:
                return
            self._kde_running = True
        try:
            self._run_kde()
        finally:
            with self.lock:
                self._kde_running = False

    # ══════════════════════════════════════════════════════════════════════
    # Core KDE processing
    # ══════════════════════════════════════════════════════════════════════

    def _run_kde(self):
        # Snapshot the buffers under lock
        with self.lock:
            buffers_snapshot = {
                cls: np.array(pts).copy()
                for cls, pts in self.point_buffers.items()
                if len(pts) >= self.min_points_for_kde
            }

        if not buffers_snapshot:
            return

        results = {}
        kde_data = {}

        for cls_name, pts in buffers_snapshot.items():
            xy = pts[:, :2].T  # shape (2, N)

            # Compute bandwidth factor relative to data std
            xy_std = max(np.std(xy, axis=1).mean(), 1e-6)
            bw_factor = self.bandwidth / xy_std

            try:
                kde = gaussian_kde(xy, bw_method=bw_factor)
            except np.linalg.LinAlgError:
                rospy.logwarn_throttle(
                    5.0, f"KDE singular matrix for {cls_name}. Skipping."
                )
                continue

            # Evaluation grid
            x_min, x_max = xy[0].min() - 1.0, xy[0].max() + 1.0
            y_min, y_max = xy[1].min() - 1.0, xy[1].max() + 1.0
            nx = max(int((x_max - x_min) / self.grid_resolution), 10)
            ny = max(int((y_max - y_min) / self.grid_resolution), 10)

            xi = np.linspace(x_min, x_max, nx)
            yi = np.linspace(y_min, y_max, ny)
            xx, yy = np.meshgrid(xi, yi)
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])

            density = kde(grid_coords).reshape(xx.shape)

            # Store for visualisation
            kde_data[cls_name] = {
                "xx": xx,
                "yy": yy,
                "density": density,
                "points": pts,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }

            # ── Non-max suppression to extract peaks ──────────────────────
            peaks = []
            density_work = density.copy()

            for _ in range(self.max_peaks_per_class):
                max_idx = np.unravel_index(
                    np.argmax(density_work), density_work.shape
                )
                max_val = density_work[max_idx]
                if max_val < self.min_peak_density:
                    break

                peak_x = float(xx[max_idx])
                peak_y = float(yy[max_idx])

                # Mean Z from nearby raw points
                dists = np.sqrt(
                    (pts[:, 0] - peak_x) ** 2 + (pts[:, 1] - peak_y) ** 2
                )
                nearby = dists < self.suppression_radius
                peak_z = float(
                    np.mean(pts[nearby, 2]) if np.any(nearby) else np.mean(pts[:, 2])
                )

                peaks.append((peak_x, peak_y, peak_z, float(max_val)))

                # Suppress around this peak
                dist_grid = np.sqrt(
                    (xx - peak_x) ** 2 + (yy - peak_y) ** 2
                )
                density_work[dist_grid < self.suppression_radius] = 0.0

            results[cls_name] = peaks

        # Take an immutable snapshot for publishing (prevents concurrent modification)
        results_snapshot = dict(results)

        with self.lock:
            self.current_results = results_snapshot

        self._publish_results(results_snapshot)
        self._publish_visualization(kde_data, results_snapshot)

    # ══════════════════════════════════════════════════════════════════════
    # Publishing
    # ══════════════════════════════════════════════════════════════════════

    def _publish_results(self, results):
        """Publish peaks via set_object_transform service.

        We do NOT broadcast TF directly — the ObjectMapTFServer already
        broadcasts TF for frames registered via set_object_transform.
        Publishing from both sources causes duplicate/conflicting frames.
        """
        for cls_name, peaks in list(results.items()):
            peaks_sorted = sorted(peaks, key=lambda p: -p[3])

            for i, (px, py, pz, _conf) in enumerate(peaks_sorted):
                frame_id = (
                    f"{cls_name}{self.frame_suffix}"
                    if i == 0
                    else f"{cls_name}{self.frame_suffix}_{i - 1}"
                )

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.static_frame
                t.child_frame_id = frame_id
                t.transform.translation = Vector3(px, py, pz)
                t.transform.rotation = Quaternion(0, 0, 0, 1)

                try:
                    req = SetObjectTransformRequest()
                    req.transform = t
                    self.set_object_transform_service.call(req)
                except rospy.ServiceException as e:
                    rospy.logwarn_throttle(
                        10.0, f"set_object_transform failed for {frame_id}: {e}"
                    )

    # ══════════════════════════════════════════════════════════════════════
    # Visualisation (CompressedImage)
    # ══════════════════════════════════════════════════════════════════════

    def _publish_visualization(self, kde_data, results):
        if not kde_data:
            return
        if self.image_pub.get_num_connections() == 0:
            return

        img_w = self.image_width
        img_h = self.image_height
        padding = 60

        # ── Global world bounds ───────────────────────────────────────────
        g_xmin = min(d["x_min"] for d in kde_data.values())
        g_xmax = max(d["x_max"] for d in kde_data.values())
        g_ymin = min(d["y_min"] for d in kde_data.values())
        g_ymax = max(d["y_max"] for d in kde_data.values())

        world_w = max(g_xmax - g_xmin, 0.1)
        world_h = max(g_ymax - g_ymin, 0.1)
        plot_w = img_w - 2 * padding
        plot_h = img_h - 2 * padding
        scale = min(plot_w / world_w, plot_h / world_h)

        # Centering offsets
        off_x = padding + (plot_w - world_w * scale) / 2.0
        off_y = padding + (plot_h - world_h * scale) / 2.0

        def w2p(wx, wy):
            """World coordinate → pixel coordinate."""
            u = int((wx - g_xmin) * scale + off_x)
            v = int(img_h - ((wy - g_ymin) * scale + off_y))
            return u, v

        # ── Canvas ────────────────────────────────────────────────────────
        canvas = np.full((img_h, img_w, 3), 30, dtype=np.uint8)

        # Draw grid lines
        grid_step = self._nice_grid_step(max(world_w, world_h))
        gx = np.arange(
            np.floor(g_xmin / grid_step) * grid_step,
            g_xmax + grid_step,
            grid_step,
        )
        gy = np.arange(
            np.floor(g_ymin / grid_step) * grid_step,
            g_ymax + grid_step,
            grid_step,
        )
        for x in gx:
            u, _ = w2p(x, 0)
            cv2.line(canvas, (u, padding), (u, img_h - padding), (50, 50, 50), 1)
            cv2.putText(
                canvas, f"{x:.1f}", (u - 10, img_h - padding + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1,
            )
        for y in gy:
            _, v = w2p(0, y)
            cv2.line(canvas, (padding, v), (img_w - padding, v), (50, 50, 50), 1)
            cv2.putText(
                canvas, f"{y:.1f}", (5, v + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1,
            )

        # ── Per-class density + points ────────────────────────────────────
        legend_y = 25
        for cls_name, data in kde_data.items():
            color = self.class_colors.get(cls_name, (255, 255, 255))
            pts = data["points"]
            density = data["density"]

            # Draw density heatmap on a temporary layer
            if density.max() > 0:
                dn = (density / density.max() * 200).astype(np.uint8)
                # Map KDE grid corners to pixel space
                u_min, v_max_p = w2p(data["x_min"], data["y_min"])
                u_max, v_min_p = w2p(data["x_max"], data["y_max"])
                tw = max(abs(u_max - u_min), 1)
                th = max(abs(v_max_p - v_min_p), 1)

                dn_resized = cv2.resize(dn, (tw, th), interpolation=cv2.INTER_LINEAR)
                layer = np.zeros((th, tw, 3), dtype=np.uint8)
                for ch in range(3):
                    layer[:, :, ch] = (dn_resized * (color[ch] / 255.0)).astype(
                        np.uint8
                    )

                # Composite onto canvas with alpha blending
                y1 = max(v_min_p, 0)
                y2 = min(v_min_p + th, img_h)
                x1 = max(u_min, 0)
                x2 = min(u_min + tw, img_w)
                cy1, cy2 = y1 - v_min_p, y1 - v_min_p + (y2 - y1)
                cx1, cx2 = x1 - u_min, x1 - u_min + (x2 - x1)

                if y2 > y1 and x2 > x1 and cy2 > cy1 and cx2 > cx1:
                    roi = layer[cy1:cy2, cx1:cx2]
                    mask = np.any(roi > 15, axis=2)
                    croi = canvas[y1:y2, x1:x2]
                    blended = cv2.addWeighted(croi, 0.6, roi, 0.4, 0)
                    croi[mask] = blended[mask]

            # Draw raw detection points
            for pt in pts:
                u, v = w2p(pt[0], pt[1])
                if 0 <= u < img_w and 0 <= v < img_h:
                    cv2.circle(canvas, (u, v), 2, color, -1)

            # Draw peaks with labels
            if cls_name in results:
                short = cls_name.replace("_link", "")
                for i, (px, py, _pz, conf) in enumerate(results[cls_name]):
                    u, v = w2p(px, py)
                    cv2.circle(canvas, (u, v), 10, color, 2)
                    cv2.circle(canvas, (u, v), 3, (255, 255, 255), -1)
                    label = f"{short}#{i} d={conf:.3f}"
                    cv2.putText(
                        canvas, label, (u + 14, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                    )

            # Legend entry
            short_name = cls_name.replace("_link", "")
            n_pts = len(pts)
            n_peaks = len(results.get(cls_name, []))
            cv2.rectangle(
                canvas, (img_w - 280, legend_y - 10), (img_w - 270, legend_y),
                color, -1,
            )
            cv2.putText(
                canvas, f"{short_name} ({n_pts}pts, {n_peaks}pk)",
                (img_w - 265, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )
            legend_y += 18

        # ── Title & axes ──────────────────────────────────────────────────
        cv2.putText(
            canvas, "KDE Object Map", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
        cv2.putText(
            canvas,
            f"X:[{g_xmin:.1f},{g_xmax:.1f}]  Y:[{g_ymin:.1f},{g_ymax:.1f}]  "
            f"bw={self.bandwidth}m  res={self.grid_resolution}m",
            (10, img_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1,
        )

        # ── Publish ──────────────────────────────────────────────────────
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        success, buf = cv2.imencode(
            ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if success:
            msg.data = buf.tobytes()
            self.image_pub.publish(msg)

    @staticmethod
    def _nice_grid_step(span):
        """Choose a visually clean grid spacing for the given world span."""
        raw = span / 8.0
        mag = 10 ** int(np.floor(np.log10(max(raw, 1e-9))))
        ratio = raw / mag
        if ratio < 1.5:
            return mag
        elif ratio < 3.5:
            return 2 * mag
        elif ratio < 7.5:
            return 5 * mag
        else:
            return 10 * mag

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = KdeObjectMapper()
    node.spin()
