#!/usr/bin/env python3

import os
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
from auv_msgs.msg import ObjectPose
from auv_msgs.srv import SetPremap, SetPremapResponse
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


class SemanticMapper:

    def __init__(self):
        rospy.init_node("semantic_mapper", anonymous=False)
        rospy.loginfo("Semantic Mapper starting...")

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
        self.lock = threading.Lock()

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
            # Use tighter threshold for slalom gates
            threshold = (
                self.slalom_distance_threshold
                if label in self.slalom_labels
                else self.distance_threshold
            )

            # CP-like Kalman filter: high Q makes filter follow measurements (odom drift)
            # When measurements stop, position stays put (no velocity extrapolation)
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
            rospy.loginfo(
                f"Created tracker for '{label}': threshold={threshold}m, "
                f"Q={self.kalman_Q}, R={self.kalman_R}"
            )

        return self.trackers[label]

    @staticmethod
    def euclidean_distance(
        detection: Detection, tracked_object: TrackedObject
    ) -> float:
        """Euclidean distance between detection and tracked object."""
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def transform_callback(self, msg: TransformStamped):
        """Handle incoming object transform updates."""
        label = msg.child_frame_id
        if not label:
            rospy.logwarn_throttle(5, "Received transform with empty child_frame_id")
            return

        pos = msg.transform.translation
        point = np.array([[pos.x, pos.y, pos.z]])
        detection = Detection(points=point, label=label)

        with self.lock:
            tracker = self.get_or_create_tracker(label)
            tracker.update(detections=[detection])

    def broadcast_transforms(self):
        """Broadcast TF for all confirmed tracks."""
        with self.lock:
            tracks_by_label = self._get_confirmed_tracks()

        for label, tracks in tracks_by_label.items():
            # Sort by distance to pre-map expected position
            tracks_sorted = self.sort_tracks_by_premap(tracks, label)

            for idx, obj in enumerate(tracks_sorted):
                if obj.estimate is None or len(obj.estimate) == 0:
                    continue

                frame_name = self._get_frame_name(label, idx)
                tf_msg = self._build_transform_msg(obj, frame_name)
                self.tf_broadcaster.sendTransform(tf_msg)

    def _get_confirmed_tracks(self) -> Dict[str, List[TrackedObject]]:
        """Collect all confirmed tracks grouped by label."""
        tracks_by_label: Dict[str, List[TrackedObject]] = defaultdict(list)

        for label, tracker in self.trackers.items():
            for obj in tracker.tracked_objects:
                if obj.hit_counter >= self.hit_counter_max:
                    tracks_by_label[label].append(obj)

        return tracks_by_label

    # TODO remove p_ prefix
    def _get_frame_name(self, label: str, idx: int) -> str:
        """Assign frame name based on distance index. Prefixed with 'p_' to distinguish from legacy tracker."""
        if idx == 0:
            return f"p_{label}"
        return f"p_{label}_{idx - 1}"

    def _build_transform_msg(
        self, obj: TrackedObject, frame_name: str
    ) -> TransformStamped:
        """Build TransformStamped message from tracked object."""
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.child_frame_id = frame_name

        est = obj.estimate[0]
        tf_msg.transform.translation = Vector3(x=est[0], y=est[1], z=est[2])
        tf_msg.transform.rotation = Quaternion(x=0, y=0, z=0, w=1)

        return tf_msg

    def _load_premap(self, filepath: str):
        """Load pre-map YAML file for indexing reference."""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)

            if "objects" not in data:
                rospy.logwarn(f"Pre-map file has no 'objects' key: {filepath}")
                return

            for label, obj_data in data["objects"].items():
                pos = obj_data.get("position", [0, 0, 0])
                orient = obj_data.get("orientation", [0, 0, 0])
                self.premap[label] = {
                    "position": np.array(pos),
                    "orientation": np.array(orient),
                }

            rospy.loginfo(
                f"Loaded pre-map with {len(self.premap)} objects from {filepath}"
            )
            self._initialize_tracks_from_premap()
        except Exception as e:
            rospy.logerr(f"Failed to load pre-map: {e}")

    def _initialize_tracks_from_premap(self):
        """Pre-create Kalman tracks at premap positions with high initial uncertainty."""
        for label, data in self.premap.items():
            pos = data["position"]
            point = np.array([[pos[0], pos[1], pos[2]]])
            detection = Detection(points=point, label=label)

            tracker = self.get_or_create_tracker(label)
            tracker.update(detections=[detection])

            # Inflate covariance for the newly created track
            if tracker.tracked_objects:
                obj = tracker.tracked_objects[-1]
                if hasattr(obj.filter, 'P'):
                    # FilterPy KalmanFilter: P is the covariance matrix
                    dim_z = obj.filter.P.shape[0] // 2
                    obj.filter.P[:dim_z, :dim_z] *= self.premap_initial_covariance
                elif hasattr(obj.filter, 'pos_variance'):
                    # OptimizedKalmanFilter: pos_variance is separate
                    obj.filter.pos_variance *= self.premap_initial_covariance
                
                # Mark as confirmed for immediate broadcast
                obj.hit_counter = self.hit_counter_max
                rospy.loginfo(f"Initialized track for {label} at {pos} with hit_counter={obj.hit_counter}")
            else:
                 rospy.logwarn(f"Failed to create track for {label} - tracked_objects is empty!")

        rospy.loginfo(
            f"Pre-initialized {len(self.premap)} tracks from premap "
            f"(initial_covariance={self.premap_initial_covariance})"
        )

    def sort_tracks_by_premap(
        self, tracks: List[TrackedObject], label: str
    ) -> List[TrackedObject]:
        """Sort tracks by distance to pre-map expected position (closest first).
        Falls back to base_link distance if label not in pre-map."""

        expected_pos = None
        if label in self.premap:
            expected_pos = self.premap[label]["position"]

        def get_distance(obj: TrackedObject) -> float:
            if obj.estimate is None or len(obj.estimate) == 0:
                return float("inf")

            est = np.array(obj.estimate[0])

            if expected_pos is not None:
                return np.linalg.norm(est - expected_pos)
            else:
                # Fallback: distance to base_link
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
        self.trackers.clear()
        rospy.loginfo("Cleared all object tracks.")
        return TriggerResponse(success=True, message="Cleared all object tracks.")

    def set_premap_handler(self, req: SetPremap) -> SetPremapResponse:
        """Service handler to set pre-map and initialize KF tracks."""
        try:
            target_frame = self.world_frame
            source_frame = req.reference_frame or target_frame
            
            new_premap_data, yaml_data_objects = self._transform_and_parse_service_objects(
                req.objects, source_frame, target_frame
            )
            if new_premap_data is None: 
                 return SetPremapResponse(success=False, message=f"Failed to transform objects from {source_frame} to {target_frame}")

            self._atomic_update_and_save(new_premap_data, yaml_data_objects, target_frame)

            msg = f"Pre-map set with {len(self.premap)} objects: {list(self.premap.keys())}"
            rospy.loginfo(msg)
            return SetPremapResponse(success=True, message=msg)

        except Exception as e:
            msg = f"Failed to set pre-map: {e}"
            rospy.logerr(msg)
            return SetPremapResponse(success=False, message=msg)

    def _transform_and_parse_service_objects(self, objects, source_frame, target_frame):
        """Processes request objects, transforms them to target frame, and prepares data structures."""
        new_premap_data = {}
        yaml_data_objects = {}
        transform_stamped = None

        if source_frame != target_frame:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(1.0)
                )
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                rospy.logerr(f"Transform error: {e}")
                return None, None

        for obj_pose in objects:
            label = obj_pose.label
            p_stamped = PoseStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.pose = obj_pose.pose

            try:
                if transform_stamped:
                    tf_pose = tf2_geometry_msgs.do_transform_pose(p_stamped, transform_stamped).pose
                else:
                    tf_pose = obj_pose.pose
                
                # Normalize to world frame
                pos = tf_pose.position
                orient = tf_pose.orientation
                
                new_premap_data[label] = {
                    "position": np.array([pos.x, pos.y, pos.z]),
                    "orientation": np.array([orient.x, orient.y, orient.z, orient.w]),
                }

                yaml_data_objects[label] = {
                    "position": [float(pos.x), float(pos.y), float(pos.z)],
                    "orientation": [float(orient.x), float(orient.y), float(orient.z), float(orient.w)],
                }

            except Exception as e:
                rospy.logwarn(f"Failed to process object {label}: {e}")
                continue
        
        return new_premap_data, yaml_data_objects

    def _atomic_update_and_save(self, new_premap_data, yaml_data_objects, target_frame):
        """Atomically updates internal state and saves to YAML."""
        with self.lock:
            self.trackers.clear()
            self.premap = new_premap_data
            self._initialize_tracks_from_premap()
            
        rospy.loginfo(f"Pre-map reset update complete. Loaded {len(self.premap)} objects (all in {target_frame}).")
        if self.premap_yaml_path:
            try:
                yaml_dir = os.path.dirname(self.premap_yaml_path)
                if not os.path.exists(yaml_dir):
                    os.makedirs(yaml_dir)

                # Backup existing file before overwriting, using its created_at timestamp
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
                            timestamp = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M")
                    except Exception:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    
                    base, ext = os.path.splitext(self.premap_yaml_path)
                    backup_path = f"{base}_{timestamp}{ext}"
                    os.rename(self.premap_yaml_path, backup_path)
                    rospy.loginfo(f"Backed up old premap to {backup_path}")

                # Enforce world_frame reference for saved data
                yaml_final_data = {
                    "reference_frame": target_frame, 
                    "created_at": datetime.now().isoformat(),
                    "objects": yaml_data_objects
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


    def run(self):
        """Main loop: broadcast transforms at configured rate."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.broadcast_transforms()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = SemanticMapper()
        node.run()
    except rospy.ROSInterruptException:
        pass
