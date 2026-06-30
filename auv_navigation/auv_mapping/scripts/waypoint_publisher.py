#!/usr/bin/env python3

import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime

import rospy
import tf2_ros
import yaml
from auv_msgs.srv import SetWaypoint, SetWaypointResponse
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from tf.transformations import euler_from_quaternion, quaternion_from_euler


DEFAULT_STATE_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "waypoints",
        "waypoint_gui_state.yaml",
    )
)

B_MODE_FIXED = 0
B_MODE_RELATIVE = 1


@dataclass
class PathState:
    index: int
    name: str
    ref_a: str
    ref_b: str
    b_mode: int
    b_reference_distance: float
    waypoint_prefix: str
    waypoints: list

    def composite_frame_name(self):
        return f"{self.name}_ref"

    def wp_frame_name(self, i):
        return f"{self.waypoint_prefix}{self.index}_wp{i + 1}"


class WaypointPublisher:
    def __init__(self):
        rospy.init_node("waypoint_publisher", anonymous=False)
        rospy.loginfo("[WaypointPublisher] Starting...")

        self.broadcast_rate_hz = float(rospy.get_param("~broadcast_rate_hz", 20.0))
        self.state_file = os.path.expanduser(
            rospy.get_param("~state_file", DEFAULT_STATE_FILE)
        )
        self.default_waypoint_prefix = rospy.get_param("~waypoint_prefix", "path")
        self.default_b_reference_distance = float(
            rospy.get_param("~b_reference_distance", 12.0)
        )
        self.paths_rosparam = rospy.get_param("~paths_rosparam", "/waypoint_gui/paths")

        self.lock = threading.Lock()
        self.paths = []
        self._last_broadcast_stamp = None
        self._rosparam_sync_counter = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self._load_state()

        self.set_waypoint_srv = rospy.Service(
            "set_waypoint", SetWaypoint, self.set_waypoint_handler
        )

        rospy.loginfo(
            f"[WaypointPublisher] Ready. paths={len(self.paths)}, "
            f"state_file='{self.state_file}'"
        )

    def set_waypoint_handler(self, req):
        new_paths = []
        for i, path_msg in enumerate(req.paths, start=1):
            path = self._path_from_msg(i, path_msg)
            if path is not None:
                new_paths.append(path)

        if not new_paths:
            return SetWaypointResponse(
                success=False,
                message="No valid waypoint paths in request; state unchanged",
            )

        with self.lock:
            self.paths = new_paths

        self._write_state_to_disk(self._build_state_dict(new_paths))
        self._sync_paths_rosparam(new_paths)

        waypoint_count = sum(len(path.waypoints) for path in new_paths)
        msg = (
            f"Waypoint set with {len(new_paths)} path(s), "
            f"{waypoint_count} waypoint(s)."
        )
        rospy.loginfo(f"[WaypointPublisher] {msg}")
        return SetWaypointResponse(success=True, message=msg)

    def _path_from_msg(self, index, path_msg):
        ref_a = path_msg.ref_a.strip()
        ref_b = path_msg.ref_b.strip()
        b_mode = self._normalize_b_mode(getattr(path_msg, "b_mode", B_MODE_FIXED))
        b_reference_distance = self._positive_float_or_default(
            getattr(
                path_msg, "b_reference_distance", self.default_b_reference_distance
            ),
            self.default_b_reference_distance,
        )
        waypoint_prefix = (
            path_msg.waypoint_prefix.strip() or self.default_waypoint_prefix
        )

        if not ref_a:
            rospy.logwarn(
                f"[WaypointPublisher] Skipping path {index}: ref_a is required"
            )
            return None

        waypoints = [self._waypoint_from_pose(pose) for pose in path_msg.waypoints]

        if not waypoints:
            rospy.logwarn(f"[WaypointPublisher] Skipping path {index}: no waypoints")
            return None

        return PathState(
            index=index,
            name=path_msg.name.strip() or f"path{index}",
            ref_a=ref_a,
            ref_b=ref_b,
            b_mode=b_mode,
            b_reference_distance=b_reference_distance,
            waypoint_prefix=waypoint_prefix,
            waypoints=waypoints,
        )

    def _normalize_b_mode(self, value):
        try:
            mode = int(value)
        except (TypeError, ValueError):
            return B_MODE_FIXED
        return mode if mode in (B_MODE_FIXED, B_MODE_RELATIVE) else B_MODE_FIXED

    def _positive_float_or_default(self, value, default):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if parsed <= 1e-6:
            return float(default)
        return parsed

    def _waypoint_from_pose(self, pose):
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "yaw": self._yaw_from_quaternion(pose.orientation),
        }

    def _yaw_from_quaternion(self, q):
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-8:
            return 0.0
        _, _, yaw = euler_from_quaternion(
            [q.x / norm, q.y / norm, q.z / norm, q.w / norm]
        )
        return math.degrees(yaw)

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return

        with open(self.state_file, "r") as f:
            data = yaml.safe_load(f) or {}

        saved_paths = data.get("paths", []) if isinstance(data, dict) else []
        if not isinstance(saved_paths, list):
            return

        loaded_paths = []
        for i, entry in enumerate(saved_paths, start=1):
            path = self._path_from_dict(i, entry)
            if path is not None:
                loaded_paths.append(path)

        with self.lock:
            self.paths = loaded_paths

        if loaded_paths:
            self._sync_paths_rosparam(loaded_paths)
            rospy.loginfo(
                f"[WaypointPublisher] Loaded {len(loaded_paths)} path(s) "
                f"from {self.state_file}"
            )

    def _path_from_dict(self, index, entry):
        if not isinstance(entry, dict):
            return None

        ref_a = str(entry.get("ref_a", "")).strip()
        ref_b = str(entry.get("ref_b", "")).strip()
        b_mode = self._normalize_b_mode(entry.get("b_mode", B_MODE_FIXED))
        b_reference_distance = self._positive_float_or_default(
            entry.get("b_reference_distance", self.default_b_reference_distance),
            self.default_b_reference_distance,
        )
        if not ref_a:
            return None

        waypoints = []
        for wp in entry.get("waypoints", []) or []:
            parsed = self._waypoint_from_dict(wp)
            if parsed is not None:
                waypoints.append(parsed)

        if not waypoints:
            return None

        return PathState(
            index=index,
            name=str(entry.get("name") or f"path{index}"),
            ref_a=ref_a,
            ref_b=ref_b,
            b_mode=b_mode,
            b_reference_distance=b_reference_distance,
            waypoint_prefix=str(
                entry.get("waypoint_prefix") or self.default_waypoint_prefix
            ),
            waypoints=waypoints,
        )

    def _waypoint_from_dict(self, wp):
        if not isinstance(wp, dict):
            return None
        if "x" not in wp or "y" not in wp:
            return None
        return {
            "x": float(wp["x"]),
            "y": float(wp["y"]),
            "z": float(wp.get("z", 0.0)),
            "yaw": float(wp.get("yaw", 0.0)),
        }

    def _build_state_dict(self, paths_snapshot):
        return {
            "created_at": datetime.now().isoformat(),
            "paths": [
                {
                    "name": path.name,
                    "ref_a": path.ref_a,
                    "ref_b": path.ref_b,
                    "b_mode": int(path.b_mode),
                    "b_reference_distance": float(path.b_reference_distance),
                    "waypoint_prefix": path.waypoint_prefix,
                    "waypoints": [self._waypoint_to_dict(wp) for wp in path.waypoints],
                }
                for path in paths_snapshot
            ],
        }

    def _waypoint_to_dict(self, wp):
        return {
            "x": float(wp["x"]),
            "y": float(wp["y"]),
            "z": float(wp["z"]),
            "yaw": float(wp["yaw"]),
        }

    def _write_state_to_disk(self, data):
        directory = os.path.dirname(self.state_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if os.path.exists(self.state_file):
            timestamp = self._existing_state_timestamp()
            base, ext = os.path.splitext(self.state_file)
            backup_path = f"{base}_{timestamp}{ext}"
            os.rename(self.state_file, backup_path)
            rospy.loginfo(f"[WaypointPublisher] Backed up old state to {backup_path}")

        with open(self.state_file, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

        rospy.loginfo(f"[WaypointPublisher] Saved state to {self.state_file}")

    def _existing_state_timestamp(self):
        mtime = os.path.getmtime(self.state_file)
        return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M")

    def broadcast_transforms(self):
        now = rospy.Time.now()
        if self._last_broadcast_stamp is not None and now <= self._last_broadcast_stamp:
            return
        self._last_broadcast_stamp = now

        with self.lock:
            paths_snapshot = list(self.paths)

        transforms = []
        for path in paths_snapshot:
            composite = self._build_composite_transform(path, now)
            if composite is None:
                continue
            composite_tf, ab_distance = composite
            transforms.append(composite_tf)

            ref_frame_name = path.composite_frame_name()
            waypoint_scale = self._waypoint_scale(path, ab_distance)
            transforms.extend(
                self._build_wp_transform(
                    path,
                    i,
                    wp,
                    ref_frame_name,
                    now,
                    waypoint_scale,
                )
                for i, wp in enumerate(path.waypoints)
            )

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

        self._rosparam_sync_counter += 1
        if self._rosparam_sync_counter >= 10:
            self._rosparam_sync_counter = 0
            self._sync_paths_rosparam(paths_snapshot)

    def _build_composite_transform(self, path, now):
        if not path.ref_a:
            return None

        if not path.ref_b or path.ref_a == path.ref_b:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = path.ref_a
            t.child_frame_id = path.composite_frame_name()
            t.transform.rotation.w = 1.0
            return t, None

        try:
            tf_a_b = self.tf_buffer.lookup_transform(
                path.ref_a,
                path.ref_b,
                rospy.Time(0),
                rospy.Duration(0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                5.0,
                f"[WaypointPublisher] Waiting for TF {path.ref_a}->{path.ref_b}: {exc}",
            )
            return None

        bx = tf_a_b.transform.translation.x
        by = tf_a_b.transform.translation.y
        ab_distance = math.hypot(bx, by)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = path.ref_a
        t.child_frame_id = path.composite_frame_name()
        q = quaternion_from_euler(0.0, 0.0, math.atan2(by, bx))
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return t, ab_distance

    def _waypoint_scale(self, path, ab_distance):
        if path.b_mode != B_MODE_RELATIVE:
            return 1.0
        if ab_distance is None:
            return 1.0
        return float(ab_distance) / float(path.b_reference_distance)

    def _build_wp_transform(self, path, i, wp, ref_frame_name, now, scale=1.0):
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = ref_frame_name
        t.child_frame_id = path.wp_frame_name(i)
        t.transform.translation = Vector3(
            x=float(wp["x"]) * scale,
            y=float(wp["y"]) * scale,
            z=float(wp["z"]),
        )
        q = quaternion_from_euler(0.0, 0.0, math.radians(float(wp["yaw"])))
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return t

    def _sync_paths_rosparam(self, paths_snapshot):
        data = {}
        for path in paths_snapshot:
            data[path.name] = {
                "ref_a": path.ref_a,
                "ref_b": path.ref_b,
                "b_mode": int(path.b_mode),
                "b_reference_distance": float(path.b_reference_distance),
                "reference_frame": path.composite_frame_name(),
                "waypoints": len(path.waypoints),
                "waypoint_frames": [
                    path.wp_frame_name(i) for i in range(len(path.waypoints))
                ],
            }
        rospy.set_param(self.paths_rosparam, data)

    def run(self):
        rate = rospy.Rate(self.broadcast_rate_hz)
        while not rospy.is_shutdown():
            self.broadcast_transforms()
            rate.sleep()


if __name__ == "__main__":
    try:
        WaypointPublisher().run()
    except rospy.ROSInterruptException:
        pass
