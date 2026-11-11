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
import rclpy
from auv_common_lib.ros import get_param, service_client
from auv_common_lib import ros
from rclpy.node import Node
import yaml
import numpy as np
from collections import defaultdict

# NOTE: scipy.stats.gaussian_kde removed — replaced with lightweight
# weighted-centroid clustering (see _run_kde). This cuts per-cycle cost
# from O(N² + N·G) to O(N·max_peaks).

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from std_srvs.srv import Trigger
from auv_msgs.srv import SetObjectTransform
from kde_visualizer import KdeVisualizer, CLASS_COLORS_BGR


class KdeObjectMapper(Node):
    def __init__(self):
        super().__init__("kde_object_mapper")

        # ── Parameters ────────────────────────────────────────────────────
        self.bandwidth = get_param(self, "~bandwidth", 0.2)
        self.min_peak_density = get_param(self, "~min_peak_density", 0.01)
        self.max_peaks_per_class = get_param(self, "~max_peaks_per_class", 5)
        self.suppression_radius = get_param(self, "~suppression_radius", 0.5)
        self.update_rate = get_param(self, "~update_rate", 2.0)
        self.static_frame = get_param(self, "~static_frame", "odom")
        self.grid_resolution = get_param(self, "~grid_resolution", 0.05)
        self.min_points_for_kde = get_param(self, "~min_points_for_kde", 5)
        self.max_points_per_class = get_param(self, "~max_points_per_class", 2000)
        self.frame_suffix = get_param(self, "~frame_suffix", "_kde")
        image_width = get_param(self, "~image_width", 800)
        image_height = get_param(self, "~image_height", 600)
        object_classes = get_param(self, "~object_classes", [])

        # Temporal decay parameter (default 0.5 seconds half-life)
        self.temporal_decay = get_param(self, "~temporal_decay", 25)

        # Premap parameters
        self.premap_path = get_param(self, "~premap_path", "")
        self.premap_max_distance = get_param(self, "~premap_max_distance", 5.0)

        if not object_classes:
            self.get_logger().warning(
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
            self.subscribers[cls_name] = self.create_subscription(
                PointStamped,
                topic,
                lambda msg, c=cls_name: self._point_callback(msg, c),
                50,
            )
            self.get_logger().info(f"Subscribed to {topic}")

        # ── Visualizer (runs on its own thread) ──────────────────────────
        self.visualizer = KdeVisualizer(
            self,
            image_width,
            image_height,
            self.class_colors,
            self.bandwidth,
            self.grid_resolution,
        )
        self.visualizer.start()

        # ── Service proxy ─────────────────────────────────────────────────
        self.set_object_transform_service = service_client(
            self, SetObjectTransform, "set_object_transform"
        )

        # ── Services ─────────────────────────────────────────────────────
        self.create_service(Trigger, "kde_map/clear", self._clear_callback)
        self.create_service(Trigger, "kde_map/trigger_update", self._trigger_callback)

        # ── Timer ─────────────────────────────────────────────────────────
        self.update_timer = self.create_timer(
            1.0 / self.update_rate, self._update_callback
        )

        self.get_logger().info(
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
            self.get_logger().warning("KDE: ~premap_path is empty. Premap is disabled.")
            return premap

        if not os.path.exists(premap_path):
            self.get_logger().warning(
                f"KDE: Premap file at {premap_path} is missing. Premap is disabled."
            )
            return premap

        try:
            with open(premap_path, "r") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                self.get_logger().warning(
                    f"KDE: Failed to parse premap YAML from {premap_path} — top-level is not a dict."
                )
                return premap

            objects_data = data.get("objects", {})
            if not isinstance(objects_data, dict):
                self.get_logger().warning(
                    f"KDE: 'objects' key in premap {premap_path} is not a dictionary."
                )
                return premap

            for k, v in objects_data.items():
                if isinstance(v, dict) and "position" in v:
                    pos = v["position"]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        premap[k] = np.array([float(pos[0]), float(pos[1])])

            self.get_logger().info(
                f"KDE: Loaded premap with {len(premap)} classes from {premap_path}: {list(premap.keys())}"
            )
        except Exception as e:
            self.get_logger().warning(
                f"KDE: Failed to load premap from {premap_path}: {e}"
            )
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
                self.get_logger().info(
                    f"KDE: Dynamically reloaded premap due to file modification (mtime={mtime})"
                )
        except Exception as e:
            self.get_logger().warning(
                f"KDE: Failed to check/reload premap: {e}", throttle_duration_sec=10.0
            )

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
            buf.append(
                [msg.point.x, msg.point.y, self.get_clock().now().nanoseconds / 1e9]
            )
            # Keep buffer bounded
            if len(buf) > self.max_points_per_class:
                del buf[: len(buf) - self.max_points_per_class]

    def _clear_callback(self, _req, response):
        with self.lock:
            self.point_buffers.clear()
            self.current_results.clear()
        self.get_logger().info("KDE: cleared all point buffers.")

        return Trigger.Response(success=True, message="Cleared all KDE buffers")

    def _trigger_callback(self, _req, response):
        self._run_kde()
        return Trigger.Response(success=True, message="KDE update triggered")

    def _update_callback(self):
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

        results = {}
        kde_data = {}
        now = self.get_clock().now().nanoseconds / 1e9

        for cls_name in list(self.class_colors.keys()):
            if cls_name in buffers_snapshot:
                pts = buffers_snapshot[cls_name]
                xy = pts[:, :2]  # shape (N, 2)

                # ── Temporal weighting (unchanged formula) ─────────────
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

                # Normalize weights
                weights_sum = np.sum(weights)
                if weights_sum > 1e-9:
                    weights = weights / weights_sum
                else:
                    weights = np.ones_like(weights) / len(weights)

                # ── Weighted-centroid peak finding ─────────────────────
                # Computes weighted centroid of ALL remaining points for
                # each peak. suppression_radius is only used AFTER the
                # centroid to remove nearby points for the next peak.
                # This ensures old + new detections all contribute to
                # the average, preventing jumps to latest detections.
                peaks = []
                remaining = np.ones(len(xy), dtype=bool)

                for _ in range(self.max_peaks_per_class):
                    if not np.any(remaining):
                        break

                    rem_idx = np.where(remaining)[0]
                    rem_w = weights[rem_idx]
                    w_sum = float(np.sum(rem_w))
                    if w_sum < self.min_peak_density:
                        break

                    # Weighted centroid of ALL remaining points
                    centroid = np.sum(xy[rem_idx] * rem_w[:, None], axis=0) / w_sum
                    peaks.append((float(centroid[0]), float(centroid[1]), w_sum))

                    # Suppress: remove points near this centroid so
                    # the next iteration finds a separate cluster
                    dists_to_centroid = np.linalg.norm(xy - centroid, axis=1)
                    remaining[dists_to_centroid < self.suppression_radius] = False

                results[cls_name] = peaks

                # Visualisation metadata (no density grid → heatmap disabled)
                x_min = float(xy[:, 0].min()) - 1.0
                x_max = float(xy[:, 0].max()) + 1.0
                y_min = float(xy[:, 1].min()) - 1.0
                y_max = float(xy[:, 1].max()) + 1.0
                kde_data[cls_name] = {
                    "points": pts,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "timestamps": timestamps,
                    "rejected_peaks": [],
                    "kde_active": False,  # no density heatmap
                }
            elif cls_name in self.premap:
                pm_x, pm_y = self.premap[cls_name]
                results[cls_name] = [(float(pm_x), float(pm_y), 1.0)]

                # Store fallback metadata for visualisation
                kde_data[cls_name] = {
                    "points": np.empty((0, 3)),
                    "x_min": float(pm_x) - 1.0,
                    "x_max": float(pm_x) + 1.0,
                    "y_min": float(pm_y) - 1.0,
                    "y_max": float(pm_y) + 1.0,
                    "timestamps": np.array([]),
                    "rejected_peaks": [],
                    "kde_active": False,
                }

            if cls_name in kde_data and cls_name in self.premap:
                kde_data[cls_name]["premap_position"] = (
                    float(self.premap[cls_name][0]),
                    float(self.premap[cls_name][1]),
                )
                kde_data[cls_name]["premap_max_distance"] = float(
                    self.premap_max_distance
                )

        if not results:
            return

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
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = self.static_frame
                t.child_frame_id = frame_id
                t.transform.translation = Vector3(x=px, y=py, z=0.0)
                t.transform.rotation = Quaternion(x=0, y=0, z=0, w=1)

                try:
                    req = SetObjectTransform.Request()
                    req.transform = t
                    self.set_object_transform_service.call(req)
                except Exception as e:
                    self.get_logger().warning(
                        f"set_object_transform failed for {frame_id}: {e}",
                        throttle_duration_sec=10.0,
                    )


def main():
    ros.init()
    node = KdeObjectMapper()
    ros.spin(node)


if __name__ == "__main__":
    main()
