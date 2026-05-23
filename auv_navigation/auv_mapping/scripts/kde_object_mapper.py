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
import yaml
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

        # Temporal decay parameter (default 0.5 seconds half-life)
        self.temporal_decay = rospy.get_param("~temporal_decay", 25)

        # Premap parameters
        self.premap_path = rospy.get_param("~premap_path", "")
        self.premap_max_distance = rospy.get_param("~premap_max_distance", 5.0)

        if not object_classes:
            rospy.logwarn(
                "No object_classes configured — KDE mapper has nothing to do."
            )

        self.point_buffers = defaultdict(list)  # class_name -> [[x, y, timestamp], ...]
        self.current_results = {}  # class_name -> [(x, y, confidence), ...]
        self.lock = threading.Lock()
        self._kde_running = False  # re-entrancy guard

        # Store file modification time to detect dynamic updates
        self._premap_mtime = 0.0
        if self.premap_path and os.path.exists(self.premap_path):
            try:
                self._premap_mtime = os.path.getmtime(self.premap_path)
            except Exception:
                pass

        # Load premap
        self.premap = self._load_premap(self.premap_path)

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
            image_width,
            image_height,
            self.class_colors,
            self.bandwidth,
            self.grid_resolution,
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

    def _load_premap(self, premap_path: str) -> dict:
        """Loads a premap YAML file mapping object class names to their known [x, y] coordinates.

        Supports the structured format where objects are nested under the 'objects' key,
        and each object specifies a 'position' key with [x, y, z] coordinates.

        Args:
            premap_path: Path to the YAML premap file.

        Returns:
            A dictionary mapping class names to np.ndarray [x, y] positions.
        """
        premap = {}
        if not premap_path:
            rospy.logwarn("KDE: ~premap_path is empty. Premap is disabled.")
            return premap

        if not os.path.exists(premap_path):
            rospy.logwarn(
                f"KDE: Premap file at {premap_path} is missing. Premap is disabled."
            )
            return premap

        try:
            with open(premap_path, "r") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                rospy.logwarn(
                    f"KDE: Failed to parse premap YAML from {premap_path} — top-level is not a dict."
                )
                return premap

            objects_data = data.get("objects", {})
            if not isinstance(objects_data, dict):
                rospy.logwarn(
                    f"KDE: 'objects' key in premap {premap_path} is not a dictionary."
                )
                return premap

            for k, v in objects_data.items():
                if isinstance(v, dict) and "position" in v:
                    pos = v["position"]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        premap[k] = np.array([float(pos[0]), float(pos[1])])

            rospy.loginfo(
                f"KDE: Loaded premap with {len(premap)} classes from {premap_path}: {list(premap.keys())}"
            )
        except Exception as e:
            rospy.logwarn(f"KDE: Failed to load premap from {premap_path}: {e}")
            premap = {}
        return premap

    def _check_and_reload_premap(self):
        """Checks if the premap YAML file has been updated and reloads it dynamically."""
        if not self.premap_path or not os.path.exists(self.premap_path):
            return

        try:
            mtime = os.path.getmtime(self.premap_path)
            if mtime != self._premap_mtime:
                new_premap = self._load_premap(self.premap_path)
                with self.lock:
                    self.premap = new_premap
                self._premap_mtime = mtime
                rospy.loginfo(
                    f"KDE: Dynamically reloaded premap due to file modification (mtime={mtime})"
                )
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"KDE: Failed to check/reload premap: {e}")

    def _point_callback(self, msg, class_name):
        with self.lock:
            if class_name in self.premap:
                premap_pos = self.premap[class_name]
                dist = float(
                    np.sqrt(
                        (msg.point.x - premap_pos[0]) ** 2
                        + (msg.point.y - premap_pos[1]) ** 2
                    )
                )
                if dist > self.premap_max_distance:
                    # rospy.logwarn_throttle(
                    #     5.0,
                    #     f"KDE: Rejected incoming point for {class_name} at ({msg.point.x:.2f}, {msg.point.y:.2f}) "
                    #     f"due to distance {dist:.2f}m > limit {self.premap_max_distance:.2f}m",
                    # )
                    return

            buf = self.point_buffers[class_name]
            buf.append([msg.point.x, msg.point.y, rospy.Time.now().to_sec()])
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
        # Dynamically reload premap if the file has been modified on disk
        self._check_and_reload_premap()

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
        now = rospy.Time.now().to_sec()

        for cls_name, pts in buffers_snapshot.items():
            xy = pts[:, :2].T  # shape (2, N)

            # Compute bandwidth factor relative to data std
            xy_std = max(np.std(xy, axis=1).mean(), 1e-6)
            bw_factor = self.bandwidth / xy_std

            # Temporal Weighting: compute weights for gaussian_kde
            if pts.shape[1] < 3:
                timestamps = np.full(pts.shape[0], now)
            else:
                timestamps = pts[:, 2]

            age = now - timestamps
            age = np.maximum(age, 0.0)

            half_life = self.temporal_decay
            if half_life > 0.0:
                weights = np.exp(-np.log(2.0) * age / half_life)
            else:
                weights = np.ones_like(age)

            weights_sum = np.sum(weights)
            if weights_sum > 1e-9:
                weights = weights / weights_sum
            else:
                weights = np.ones_like(weights) / len(weights)

            try:
                kde = gaussian_kde(xy, bw_method=bw_factor, weights=weights)
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
                "timestamps": timestamps,
                "rejected_peaks": [],
            }
            if cls_name in self.premap:
                kde_data[cls_name]["premap_position"] = (
                    float(self.premap[cls_name][0]),
                    float(self.premap[cls_name][1]),
                )
                kde_data[cls_name]["premap_max_distance"] = float(
                    self.premap_max_distance
                )

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
