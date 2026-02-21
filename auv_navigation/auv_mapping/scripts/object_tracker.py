#!/usr/bin/env python3

import os
import time
import rospy
import numpy as np
import yaml
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from norfair import Detection, Tracker
from norfair.tracker import TrackedObject
from norfair.filter import FilterPyKalmanFilterFactory

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import SetPremap, SetPremapResponse
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from tf.transformations import quaternion_multiply, quaternion_slerp


class ObjectTracker:
    def __init__(self):
        rospy.init_node("object_tracker", anonymous=False)
        rospy.loginfo("Object Tracker starting...")

        self.world_frame = rospy.get_param("~static_frame", "odom")
        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link")
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.distance_threshold = rospy.get_param("~distance_threshold", 4.0)
        self.hit_counter_max = rospy.get_param("~hit_counter_max", 5)
        self.initialization_delay = rospy.get_param("~initialization_delay", 4)

        self.kalman_Q = rospy.get_param("~kalman_process_noise", 0.1)
        self.kalman_R = rospy.get_param("~kalman_measurement_noise", 0.5)

        self.slalom_distance_threshold = rospy.get_param(
            "~slalom_distance_threshold", 1.0
        )
        self.premap_initial_covariance = rospy.get_param(
            "~premap_initial_covariance", 100.0
        )
        self.slalom_labels = ["red_pipe_link", "white_pipe_link"]
        self.trackers: Dict[str, Tracker] = {}
        self.orientations: Dict[str, np.ndarray] = {}
        self.confirmed_tracks: Dict[str, Dict] = {}
        self.orientation_alpha = rospy.get_param("~orientation_alpha", 0.2)
        self.lock = threading.Lock()

        self.stats_lock = threading.Lock()
        self.stats = {
            "callback_count": 0,
            "callback_time_sum": 0.0,
            "tf_time_sum": 0.0,
            "track_time_sum": 0.0,
            "broadcast_count": 0,
            "broadcast_time_sum": 0.0,
            "total_tracks": 0,
        }
        self.last_stats_time = time.time()

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.sub = rospy.Subscriber(
            "object_transform_updates",
            TransformStamped,
            self.transform_callback,
            queue_size=10,
        )

        self.clear_srv = rospy.Service(
            "clear_object_transforms", Trigger, self.clear_tracks_handler
        )

        self.premap: Dict[str, Dict] = {}
        premap_file = rospy.get_param("~premap_file", "")
        self.premap_yaml_path = premap_file
        if premap_file:
            self._load_premap(premap_file)

        self.set_premap_srv = rospy.Service(
            "set_premap", SetPremap, self.set_premap_handler
        )

        rospy.loginfo(
            f"Tracker initialized. "
            f"distance_threshold={self.distance_threshold}m, "
            f"hit_counter_max={self.hit_counter_max}, "
            f"premap_objects={len(self.premap)}"
        )

    def get_or_create_tracker(self, label: str) -> Tracker:
        """Get existing tracker for label or create a new one."""
        if label not in self.trackers:
            threshold = (
                self.slalom_distance_threshold
                if label in self.slalom_labels
                else self.distance_threshold
            )

            filter_factory = FilterPyKalmanFilterFactory(
                R=self.kalman_R,
                Q=self.kalman_Q,
            )

            self.trackers[label] = Tracker(
                distance_function=self.euclidean_distance,
                distance_threshold=threshold,
                hit_counter_max=self.hit_counter_max,
                initialization_delay=self.initialization_delay,
                past_detections_length=10,
                filter_factory=filter_factory,
                reid_hit_counter_max=100000,
            )

        return self.trackers[label]

    @staticmethod
    def euclidean_distance(
        detection: Detection, tracked_object: TrackedObject
    ) -> float:
        """Euclidean distance between detection and tracked object."""
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def transform_callback(self, msg: TransformStamped):
        """Transform detection to world frame and update tracker."""
        t_start = time.perf_counter()

        label = msg.child_frame_id
        if not label:
            rospy.logwarn_throttle(5, "Received transform with empty child_frame_id")
            return

        parent_frame = msg.header.frame_id
        msg_stamp = msg.header.stamp

        if parent_frame == self.world_frame:
            pos = msg.transform.translation
            rot = msg.transform.rotation
            point_in_world = np.array([[pos.x, pos.y, pos.z]])
            quat_in_world = np.array([rot.x, rot.y, rot.z, rot.w])
            t_tf = time.perf_counter()
        else:
            result = self._transform_pose_to_world(
                msg.transform, parent_frame, msg_stamp
            )
            t_tf = time.perf_counter()
            if result is None:
                return
            point_in_world, quat_in_world = result

        detection = Detection(points=point_in_world, label=label)

        with self.lock:
            tracker = self.get_or_create_tracker(label)
            tracker.update(detections=[detection])
            self._update_orientation(label, tracker, quat_in_world)
            self._mark_confirmed_tracks(label, tracker)

        t_end = time.perf_counter()

        with self.stats_lock:
            self.stats["callback_count"] += 1
            self.stats["callback_time_sum"] += (t_end - t_start) * 1000
            self.stats["tf_time_sum"] += (t_tf - t_start) * 1000
            self.stats["track_time_sum"] += (t_end - t_tf) * 1000

    def _update_orientation(self, label: str, tracker: Tracker, new_quat: np.ndarray):
        """SLERP for orientation updates"""
        for obj in tracker.tracked_objects:
            track_id = obj.id
            if track_id is None:
                continue
            track_key = self._track_key(label, track_id)
            if track_key not in self.orientations:
                self.orientations[track_key] = new_quat
            else:
                current = self.orientations[track_key]
                self.orientations[track_key] = quaternion_slerp(
                    current, new_quat, self.orientation_alpha
                )

    def _mark_confirmed_tracks(self, label: str, tracker: Tracker):
        """Latch confirmed tracks and keep their latest pose for publishing."""
        for obj in tracker.tracked_objects:
            if obj.id is None or obj.estimate is None or len(obj.estimate) == 0:
                continue

            track_key = self._track_key(label, obj.id)

            if track_key not in self.confirmed_tracks and obj.is_initializing:
                continue

            self.confirmed_tracks[track_key] = {
                "label": label,
                "position": np.array(obj.estimate[0]),
                "orientation": np.array(
                    self.orientations.get(track_key, np.array([0.0, 0.0, 0.0, 1.0]))
                ),
            }

    def _transform_pose_to_world(
        self, transform, parent_frame: str, stamp: rospy.Time
    ) -> tuple:
        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                parent_frame,
                stamp,
                rospy.Duration(0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            rospy.logwarn_throttle(
                5.0,
                f"Failed to lookup transform {parent_frame} -> {self.world_frame} "
                f"at time {stamp.to_sec():.3f}: {e}",
            )
            return None

        point_stamped = PointStamped()
        point_stamped.header.frame_id = parent_frame
        point_stamped.header.stamp = stamp
        point_stamped.point.x = transform.translation.x
        point_stamped.point.y = transform.translation.y
        point_stamped.point.z = transform.translation.z
        point_in_world = tf2_geometry_msgs.do_transform_point(
            point_stamped, tf_transform
        )

        tf_rot = tf_transform.transform.rotation
        local_rot = transform.rotation
        q_tf = np.array([tf_rot.x, tf_rot.y, tf_rot.z, tf_rot.w])
        q_local = np.array([local_rot.x, local_rot.y, local_rot.z, local_rot.w])
        q_world = quaternion_multiply(q_tf, q_local)

        position = np.array(
            [
                [
                    point_in_world.point.x,
                    point_in_world.point.y,
                    point_in_world.point.z,
                ]
            ]
        )
        return position, q_world

    def broadcast_transforms(self):
        """Broadcast TF for all confirmed tracks."""
        t_start = time.perf_counter()

        with self.lock:
            tracks_by_label = self._get_confirmed_tracks()

        n_tracks = 0
        for label, tracks in tracks_by_label.items():
            tracks_sorted = self.sort_tracks_by_premap(tracks, label)

            for idx, track in enumerate(tracks_sorted):
                frame_name = self._get_frame_name(label, idx)
                tf_msg = self._build_transform_msg(track, frame_name)
                self.tf_broadcaster.sendTransform(tf_msg)
                n_tracks += 1

        t_end = time.perf_counter()

        with self.stats_lock:
            self.stats["broadcast_count"] += 1
            self.stats["broadcast_time_sum"] += (t_end - t_start) * 1000
            self.stats["total_tracks"] = n_tracks

    def _get_confirmed_tracks(self) -> Dict[str, List[Dict]]:
        """Collect all latched confirmed tracks grouped by label."""
        tracks_by_label: Dict[str, List[Dict]] = defaultdict(list)

        for track in self.confirmed_tracks.values():
            tracks_by_label[track["label"]].append(track)

        return tracks_by_label

    def _get_frame_name(self, label: str, idx: int) -> str:
        """Assign frame name. Prefixed with 'p_' to distinguish from legacy."""
        if idx == 0:
            return f"p_{label}"
        return f"p_{label}_{idx - 1}"

    @staticmethod
    def _track_key(label: str, track_id: int) -> str:
        return f"{label}:{track_id}"

    def _build_transform_msg(self, track: Dict, frame_name: str) -> TransformStamped:
        """Build TransformStamped message from cached confirmed track."""
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.child_frame_id = frame_name

        est = track["position"]
        tf_msg.transform.translation = Vector3(x=est[0], y=est[1], z=est[2])

        q = track.get("orientation", np.array([0.0, 0.0, 0.0, 1.0]))
        tf_msg.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return tf_msg

    def _load_premap(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)

            if data is None:
                rospy.logwarn(f"Pre-map file is empty: {filepath}")
                return

            if "objects" not in data:
                rospy.logwarn(f"Pre-map file has no 'objects' key: {filepath}")
                return

            source_frame = data.get("reference_frame", self.world_frame)
            needs_transform = source_frame != self.world_frame
            transform_stamped = None

            if needs_transform:
                rospy.loginfo(
                    f"Premap reference_frame='{source_frame}' differs from "
                    f"world_frame='{self.world_frame}', waiting for TF..."
                )
                try:
                    transform_stamped = self.tf_buffer.lookup_transform(
                        self.world_frame,
                        source_frame,
                        rospy.Time(0),
                        rospy.Duration(10.0),
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException,
                ) as e:
                    rospy.logerr(
                        f"Cannot transform premap from '{source_frame}' to "
                        f"'{self.world_frame}': {e}. Skipping premap load."
                    )
                    return

            for label, obj_data in data["objects"].items():
                validated = self._validate_premap_object(label, obj_data)
                if validated is None:
                    continue
                pos, orient = validated

                if transform_stamped:
                    p_stamped = PoseStamped()
                    p_stamped.header.frame_id = source_frame
                    p_stamped.pose.position.x = pos[0]
                    p_stamped.pose.position.y = pos[1]
                    p_stamped.pose.position.z = pos[2]
                    if len(orient) == 4:
                        p_stamped.pose.orientation.x = orient[0]
                        p_stamped.pose.orientation.y = orient[1]
                        p_stamped.pose.orientation.z = orient[2]
                        p_stamped.pose.orientation.w = orient[3]
                    else:
                        p_stamped.pose.orientation.w = 1.0

                    tf_pose = tf2_geometry_msgs.do_transform_pose(
                        p_stamped, transform_stamped
                    ).pose
                    pos = [tf_pose.position.x, tf_pose.position.y, tf_pose.position.z]
                    orient = [
                        tf_pose.orientation.x,
                        tf_pose.orientation.y,
                        tf_pose.orientation.z,
                        tf_pose.orientation.w,
                    ]

                self.premap[label] = {
                    "position": np.array(pos),
                    "orientation": np.array(orient),
                }

            frame_msg = f" (transformed from {source_frame})" if needs_transform else ""
            rospy.loginfo(
                f"Loaded pre-map with {len(self.premap)} objects from {filepath}{frame_msg}"
            )
            self._initialize_tracks_from_premap()
        except Exception as e:
            rospy.logerr(f"Failed to load pre-map: {e}")

    def _validate_premap_object(self, label: str, obj_data: Dict):
        """Validate and normalize one premap object entry."""
        if not isinstance(obj_data, dict):
            rospy.logwarn(f"Skipping premap object '{label}': entry must be a dict")
            return None

        pos = obj_data.get("position", [0, 0, 0])
        orient = obj_data.get("orientation", [0, 0, 0, 1])

        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            rospy.logwarn(
                f"Skipping premap object '{label}': position must be length-3 list"
            )
            return None
        if not isinstance(orient, (list, tuple)) or len(orient) != 4:
            rospy.logwarn(
                f"Skipping premap object '{label}': orientation must be length-4 list"
            )
            return None

        try:
            pos = [float(pos[0]), float(pos[1]), float(pos[2])]
            orient = [
                float(orient[0]),
                float(orient[1]),
                float(orient[2]),
                float(orient[3]),
            ]
        except (TypeError, ValueError):
            rospy.logwarn(
                f"Skipping premap object '{label}': non-numeric position/orientation"
            )
            return None

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(orient)):
            rospy.logwarn(
                f"Skipping premap object '{label}': non-finite position/orientation"
            )
            return None

        q_norm = np.linalg.norm(orient)
        if q_norm < 1e-8:
            rospy.logwarn(
                f"Skipping premap object '{label}': zero-norm orientation quaternion"
            )
            return None
        orient = [o / q_norm for o in orient]

        return pos, orient

    def _initialize_tracks_from_premap(self):
        """Pre-create Kalman tracks at premap positions with high initial uncertainty."""
        for label, data in self.premap.items():
            pos = data["position"]
            point = np.array([[pos[0], pos[1], pos[2]]])
            detection = Detection(points=point, label=label)

            tracker = self.get_or_create_tracker(label)
            for _ in range(self.initialization_delay + 1):
                tracker.update(detections=[detection])

            if tracker.tracked_objects:
                obj = tracker.tracked_objects[-1]
                if hasattr(obj.filter, "P"):
                    dim_z = obj.filter.P.shape[0] // 2
                    obj.filter.P[:dim_z, :dim_z] *= self.premap_initial_covariance
                elif hasattr(obj.filter, "pos_variance"):
                    obj.filter.pos_variance *= self.premap_initial_covariance

                obj.hit_counter = self.hit_counter_max
                if obj.id is not None:
                    track_key = self._track_key(label, obj.id)
                    if "orientation" in data and len(data["orientation"]) == 4:
                        orientation = np.array(data["orientation"])
                    else:
                        orientation = np.array([0.0, 0.0, 0.0, 1.0])

                    self.orientations[track_key] = orientation
                    self.confirmed_tracks[track_key] = {
                        "label": label,
                        "position": np.array(pos),
                        "orientation": np.array(orientation),
                    }
                else:
                    rospy.logwarn(
                        f"Premap track for '{label}' is still initializing; skipping orientation init"
                    )

            else:
                rospy.logwarn(
                    f"Failed to create track for {label} - tracked_objects is empty!"
                )

        rospy.loginfo(
            f"Pre-initialized {len(self.premap)} tracks from premap "
            f"(initial_covariance={self.premap_initial_covariance})"
        )

    def sort_tracks_by_premap(self, tracks: List[Dict], label: str) -> List[Dict]:
        """Sort tracks by distance to premap position (closest first)."""

        expected_pos = None
        if label in self.premap:
            expected_pos = self.premap[label]["position"]

        def get_distance(track: Dict) -> float:
            est = np.array(track["position"])

            if expected_pos is not None:
                return np.linalg.norm(est - expected_pos)
            else:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.base_link_frame,
                        self.world_frame,
                        rospy.Time(0),
                        rospy.Duration(0.1),
                    )
                    point = PointStamped()
                    point.header.frame_id = self.world_frame
                    point.point.x, point.point.y, point.point.z = est[0], est[1], est[2]
                    point_in_base = tf2_geometry_msgs.do_transform_point(
                        point, transform
                    )
                    return np.sqrt(
                        point_in_base.point.x**2
                        + point_in_base.point.y**2
                        + point_in_base.point.z**2
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException,
                ):
                    return np.linalg.norm(est)

        return sorted(tracks, key=get_distance)

    def clear_tracks_handler(self, req) -> TriggerResponse:
        """Service handler to clear all tracks."""
        with self.lock:
            self.trackers.clear()
            self.orientations.clear()
            self.confirmed_tracks.clear()
        rospy.loginfo("Cleared all object tracks.")
        return TriggerResponse(success=True, message="Cleared all object tracks.")

    def set_premap_handler(self, req: SetPremap) -> SetPremapResponse:
        """Service handler to set pre-map and initialize KF tracks."""
        try:
            target_frame = self.world_frame
            source_frame = req.reference_frame or target_frame

            new_premap_data, yaml_data_objects = (
                self._transform_and_parse_service_objects(
                    req.objects, source_frame, target_frame
                )
            )
            if new_premap_data is None:
                return SetPremapResponse(
                    success=False,
                    message=f"Failed to transform objects from {source_frame} to {target_frame}",
                )
            if not new_premap_data:
                return SetPremapResponse(
                    success=False,
                    message="No valid objects in request; pre-map unchanged",
                )

            self._atomic_update_and_save(
                new_premap_data, yaml_data_objects, target_frame
            )

            msg = f"Pre-map set with {len(self.premap)} objects: {list(self.premap.keys())}"
            rospy.loginfo(msg)
            return SetPremapResponse(success=True, message=msg)

        except Exception as e:
            msg = f"Failed to set pre-map: {e}"
            rospy.logerr(msg)
            return SetPremapResponse(success=False, message=msg)

    def _transform_and_parse_service_objects(self, objects, source_frame, target_frame):
        """Transform request objects to target frame."""
        new_premap_data = {}
        yaml_data_objects = {}
        transform_stamped = None

        if source_frame != target_frame:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
            ) as e:
                rospy.logerr(f"Transform error: {e}")
                return None, None

        for obj_pose in objects:
            label = obj_pose.label
            p_stamped = PoseStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.pose = obj_pose.pose

            try:
                if transform_stamped:
                    tf_pose = tf2_geometry_msgs.do_transform_pose(
                        p_stamped, transform_stamped
                    ).pose
                else:
                    tf_pose = obj_pose.pose

                pos = tf_pose.position
                orient = tf_pose.orientation

                if label in new_premap_data:
                    rospy.logwarn(
                        f"[SetPremap] Duplicate label '{label}' in request. "
                        f"Overwriting previous value."
                    )

                new_premap_data[label] = {
                    "position": np.array([pos.x, pos.y, pos.z]),
                    "orientation": np.array([orient.x, orient.y, orient.z, orient.w]),
                }

                yaml_data_objects[label] = {
                    "position": [float(pos.x), float(pos.y), float(pos.z)],
                    "orientation": [
                        float(orient.x),
                        float(orient.y),
                        float(orient.z),
                        float(orient.w),
                    ],
                }

            except Exception as e:
                rospy.logwarn(f"Failed to process object {label}: {e}")
                continue

        return new_premap_data, yaml_data_objects

    def _atomic_update_and_save(self, new_premap_data, yaml_data_objects, target_frame):
        """Update internal state and save to YAML."""
        with self.lock:
            labels_to_keep = set(new_premap_data.keys())

            for label in list(self.trackers.keys()):
                if label not in labels_to_keep:
                    for obj in self.trackers[label].tracked_objects:
                        if obj.id is not None:
                            self.orientations.pop(self._track_key(label, obj.id), None)
                    del self.trackers[label]

            for label in labels_to_keep:
                if label in self.trackers:
                    for obj in self.trackers[label].tracked_objects:
                        if obj.id is not None:
                            self.orientations.pop(self._track_key(label, obj.id), None)
                    del self.trackers[label]

            self.confirmed_tracks.clear()
            self.orientations.clear()

            self.premap = new_premap_data
            self._initialize_tracks_from_premap()

        rospy.loginfo(
            f"Pre-map reset update complete. Loaded {len(self.premap)} objects (all in {target_frame})."
        )
        if self.premap_yaml_path:
            try:
                yaml_dir = os.path.dirname(self.premap_yaml_path)
                if yaml_dir and not os.path.exists(yaml_dir):
                    os.makedirs(yaml_dir)

                if os.path.exists(self.premap_yaml_path):
                    try:
                        with open(self.premap_yaml_path, "r") as old_f:
                            old_data = yaml.safe_load(old_f)
                        old_timestamp = old_data.get("created_at", "")
                        if old_timestamp:
                            old_dt = datetime.fromisoformat(old_timestamp)
                            timestamp = old_dt.strftime("%Y%m%d_%H%M")
                        else:
                            mtime = os.path.getmtime(self.premap_yaml_path)
                            timestamp = datetime.fromtimestamp(mtime).strftime(
                                "%Y%m%d_%H%M"
                            )
                    except Exception:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

                    base, ext = os.path.splitext(self.premap_yaml_path)
                    backup_path = f"{base}_{timestamp}{ext}"
                    os.rename(self.premap_yaml_path, backup_path)
                    rospy.loginfo(f"Backed up old premap to {backup_path}")

                yaml_final_data = {
                    "reference_frame": target_frame,
                    "created_at": datetime.now().isoformat(),
                    "objects": yaml_data_objects,
                }

                with open(self.premap_yaml_path, "w") as f:
                    yaml.dump(
                        yaml_final_data, f, default_flow_style=False, sort_keys=False
                    )

                rospy.loginfo(f"Saved premap to {self.premap_yaml_path}")
            except Exception as yaml_err:
                rospy.logerr(f"Failed to save YAML: {yaml_err}")
        else:
            rospy.logwarn("No premap_file parameter set, not saving to disk")

    def _log_stats(self):
        """Log consolidated stats every 10 seconds."""
        now = time.time()
        elapsed = now - self.last_stats_time
        if elapsed < 10.0:
            return

        with self.stats_lock:
            cb_count = self.stats["callback_count"]
            cb_time = self.stats["callback_time_sum"]
            tf_time = self.stats["tf_time_sum"]
            track_time = self.stats["track_time_sum"]
            bc_count = self.stats["broadcast_count"]
            bc_time = self.stats["broadcast_time_sum"]
            n_tracks = self.stats["total_tracks"]

            self.stats = {
                "callback_count": 0,
                "callback_time_sum": 0.0,
                "tf_time_sum": 0.0,
                "track_time_sum": 0.0,
                "broadcast_count": 0,
                "broadcast_time_sum": 0.0,
                "total_tracks": 0,
            }
            self.last_stats_time = now

        if cb_count > 0:
            avg_cb = cb_time / cb_count
            avg_tf = tf_time / cb_count
            avg_track = track_time / cb_count
        else:
            avg_cb = avg_tf = avg_track = 0.0

        avg_bc = bc_time / bc_count if bc_count > 0 else 0.0

        rospy.loginfo(
            f"[Stats] {elapsed:.1f}s | "
            f"cb: {cb_count} @ {avg_cb:.2f}ms (tf:{avg_tf:.2f} track:{avg_track:.2f}) | "
            f"bc: {bc_count} @ {avg_bc:.2f}ms | "
            f"tracks: {n_tracks}"
        )

    def run(self):
        """Main loop: broadcast transforms at configured rate."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.broadcast_transforms()
            self._log_stats()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ObjectTracker()
        node.run()
    except rospy.ROSInterruptException:
        pass
