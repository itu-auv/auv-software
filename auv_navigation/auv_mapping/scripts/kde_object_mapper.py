#!/usr/bin/env python3
"""KDE Object Mapper Node

Performs Kernel Density Estimation on accumulated detection points to find
the strongest spatial clusters for each object class. Publishes:
- Filtered object poses via set_object_transform service
- TF frames for each detected peak
- Compressed image visualization of all class distributions
"""

import threading
import os
import sys
import rospy
import numpy as np
from collections import defaultdict
from scipy.stats import gaussian_kde

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from kde_visualizer import KdeVisualizer, CLASS_COLORS_BGR




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
        image_width = rospy.get_param("~image_width", 800)
        image_height = rospy.get_param("~image_height", 600)
        object_classes = rospy.get_param("~object_classes", [])

        if not object_classes:
            rospy.logwarn(
                "No object_classes configured — KDE mapper has nothing to do."
            )

        self.point_buffers = defaultdict(list)  # class_name -> [[x, y], ...]
        self.current_results = {}  # class_name -> [(x, y, confidence), ...]
        self.lock = threading.Lock()
        self._kde_running = False  # re-entrancy guard

        # Assign a stable colour to each class
        self.class_colors = {}
        for i, cls in enumerate(object_classes):
            self.class_colors[cls] = CLASS_COLORS_BGR[i % len(CLASS_COLORS_BGR)]

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

        # ── Visualizer (runs on its own thread) ──────────────────────────
        self.visualizer = KdeVisualizer(
            image_width, image_height, self.class_colors,
            self.bandwidth, self.grid_resolution,
        )
        self.visualizer.start()

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

    def _point_callback(self, msg, class_name):
        with self.lock:
            buf = self.point_buffers[class_name]
            buf.append([msg.point.x, msg.point.y])
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

            peaks = []
            density_work = density.copy()

            for _ in range(self.max_peaks_per_class):
                max_idx = np.unravel_index(np.argmax(density_work), density_work.shape)
                max_val = density_work[max_idx]
                if max_val < self.min_peak_density:
                    break

                peak_x = float(xx[max_idx])
                peak_y = float(yy[max_idx])

                peaks.append((peak_x, peak_y, float(max_val)))

                # Suppress around this peak
                dist_grid = np.sqrt((xx - peak_x) ** 2 + (yy - peak_y) ** 2)
                density_work[dist_grid < self.suppression_radius] = 0.0

            results[cls_name] = peaks

        # Take an immutable snapshot for publishing (prevents concurrent modification)
        results_snapshot = dict(results)

        with self.lock:
            self.current_results = results_snapshot

        self._publish_results(results_snapshot)
        self.visualizer.update(kde_data, results_snapshot)

    def _publish_results(self, results):
        """Publish peaks via set_object_transform service.

        We do NOT broadcast TF directly — the ObjectMapTFServer already
        broadcasts TF for frames registered via set_object_transform.
        Publishing from both sources causes duplicate/conflicting frames.
        """
        for cls_name, peaks in list(results.items()):
            peaks_sorted = sorted(peaks, key=lambda p: -p[2])

            for i, (px, py, _conf) in enumerate(peaks_sorted):
                frame_id = (
                    f"{cls_name}{self.frame_suffix}"
                    if i == 0
                    else f"{cls_name}{self.frame_suffix}_{i - 1}"
                )

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.static_frame
                t.child_frame_id = frame_id
                t.transform.translation = Vector3(px, py, 0.0)
                t.transform.rotation = Quaternion(0, 0, 0, 1)

                try:
                    req = SetObjectTransformRequest()
                    req.transform = t
                    self.set_object_transform_service.call(req)
                except rospy.ServiceException as e:
                    rospy.logwarn_throttle(
                        10.0, f"set_object_transform failed for {frame_id}: {e}"
                    )



    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = KdeObjectMapper()
    node.spin()
