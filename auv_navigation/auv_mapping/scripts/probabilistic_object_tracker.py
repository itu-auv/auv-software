#!/usr/bin/env python3

import rospy
import numpy as np
import yaml
from collections import defaultdict
from typing import Dict, List

from norfair import Detection, Tracker
from norfair.tracker import TrackedObject
from norfair.filter import FilterPyKalmanFilterFactory

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.msg import ObjectPose
from auv_msgs.srv import SetPremap, SetPremapResponse
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


class ProbabilisticObjectTracker:

    def __init__(self):
        rospy.init_node("probabilistic_object_tracker", anonymous=False)
        rospy.loginfo("Probabilistic Object Tracker starting...")

        # Parameters
        self.world_frame = rospy.get_param("~static_frame", "odom")
        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link")
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.distance_threshold = rospy.get_param("~distance_threshold", 4.0) 
        self.hit_counter_max = rospy.get_param("~hit_counter_max", 5)
        self.initialization_delay = rospy.get_param("~initialization_delay", 4)

        self.kalman_Q = rospy.get_param("~kalman_process_noise", 0.1)
        self.kalman_R = rospy.get_param("~kalman_measurement_noise", 0.5)

        self.slalom_distance_threshold = rospy.get_param("~slalom_distance_threshold", 1.0)
        self.slalom_labels = ["red_pipe_link", "white_pipe_link"]
        self.trackers: Dict[str, Tracker] = {}

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.sub = rospy.Subscriber(
            "object_transform_updates",
            TransformStamped,
            self.transform_callback,
            queue_size=10
        )

        # Service to clear all tracks
        self.clear_srv = rospy.Service(
            "clear_object_transforms",
            Trigger,
            self.clear_tracks_handler
        )

        # Optional pre-map 
        self.premap: Dict[str, Dict] = {}
        premap_file = rospy.get_param("~premap_file", "")
        if premap_file:
            self._load_premap(premap_file)

        # Service to set pre-map dynamically
        self.set_premap_srv = rospy.Service(
            "set_premap",
            SetPremap,
            self.set_premap_handler
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
            threshold = self.slalom_distance_threshold if label in self.slalom_labels else self.distance_threshold

            # CP-like Kalman filter: high Q makes filter follow measurements (odom drift)
            # When measurements stop, position stays put (no velocity extrapolation)
            filter_factory = FilterPyKalmanFilterFactory(
                R=self.kalman_R,  # Measurement noise
                Q=self.kalman_Q,  # Process noise (high = trust measurements)
            )

            self.trackers[label] = Tracker(
                distance_function=self.euclidean_distance,
                distance_threshold=threshold,
                hit_counter_max=self.hit_counter_max,
                initialization_delay=self.initialization_delay,
                hit_inactivity_cutoff=None,  # Tracks never expire due to inactivity
                past_detections_length=10,
                filter_factory=filter_factory,
            )
            rospy.loginfo(
                f"Created tracker for '{label}': threshold={threshold}m, "
                f"Q={self.kalman_Q}, R={self.kalman_R}"
            )

        return self.trackers[label]

    @staticmethod
    def euclidean_distance(detection: Detection, tracked_object: TrackedObject) -> float:
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

        # Get or create tracker and update
        tracker = self.get_or_create_tracker(label)
        tracker.update(detections=[detection])

    def broadcast_transforms(self):
        """Broadcast TF for all confirmed tracks."""
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
    #TODO remove p_ prefix
    def _get_frame_name(self, label: str, idx: int) -> str:
        """Assign frame name based on distance index. Prefixed with 'p_' to distinguish from legacy tracker."""
        if idx == 0:
            return f"p_{label}"
        return f"p_{label}_{idx - 1}"

    def _build_transform_msg(self, obj: TrackedObject, frame_name: str) -> TransformStamped:
        """Build TransformStamped message from tracked object."""
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.child_frame_id = frame_name

        est = obj.estimate[0]
        tf_msg.transform.translation = Vector3(x=est[0], y=est[1], z=est[2])
        tf_msg.transform.rotation = Quaternion(x=0, y=0, z=0, w=1)  # Identity

        return tf_msg

    def _load_premap(self, filepath: str):
        """Load pre-map YAML file for indexing reference."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'objects' not in data:
                rospy.logwarn(f"Pre-map file has no 'objects' key: {filepath}")
                return
            
            for label, obj_data in data['objects'].items():
                pos = obj_data.get('position', [0, 0, 0])
                orient = obj_data.get('orientation', [0, 0, 0])
                self.premap[label] = {
                    'position': np.array(pos),
                    'orientation': np.array(orient),
                }
            
            rospy.loginfo(f"Loaded pre-map with {len(self.premap)} objects from {filepath}")
        except Exception as e:
            rospy.logerr(f"Failed to load pre-map: {e}")

    def sort_tracks_by_premap(self, tracks: List[TrackedObject], label: str) -> List[TrackedObject]:
        """Sort tracks by distance to pre-map expected position (closest first).
        Falls back to base_link distance if label not in pre-map."""
        
        # Get expected position from pre-map
        expected_pos = None
        if label in self.premap:
            expected_pos = self.premap[label]['position']
        
        def get_distance(obj: TrackedObject) -> float:
            if obj.estimate is None or len(obj.estimate) == 0:
                return float('inf')
            
            est = np.array(obj.estimate[0])
            
            if expected_pos is not None:
                # Distance to pre-map expected position
                return np.linalg.norm(est - expected_pos)
            else:
                # Fallback: distance to base_link
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.base_link_frame, self.world_frame, rospy.Time(0), rospy.Duration(0.1)
                    )
                    point = PointStamped()
                    point.header.frame_id = self.world_frame
                    point.point.x, point.point.y, point.point.z = est[0], est[1], est[2]
                    point_in_base = tf2_geometry_msgs.do_transform_point(point, transform)
                    return np.sqrt(
                        point_in_base.point.x**2 +
                        point_in_base.point.y**2 +
                        point_in_base.point.z**2
                    )
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
                    return np.linalg.norm(est)

        return sorted(tracks, key=get_distance)

    def clear_tracks_handler(self, req) -> TriggerResponse:
        """Service handler to clear all tracks."""
        self.trackers.clear()
        rospy.loginfo("Cleared all object tracks.")
        return TriggerResponse(success=True, message="Cleared all object tracks.")

    def set_premap_handler(self, req) -> SetPremapResponse:
        """Service handler to set pre-map (for indexing only, not KF initialization)."""
        try:
            # Clear existing state
            self.trackers.clear()
            self.track_orientations.clear()
            self.premap.clear()
            
            # Update world frame if provided
            if req.reference_frame:
                self.world_frame = req.reference_frame
            
            # Load pre-map from service request (used only for indexing)
            for obj_pose in req.objects:
                label = obj_pose.label
                pos = obj_pose.pose.position
                orient = obj_pose.pose.orientation
                
                self.premap[label] = {
                    'position': np.array([pos.x, pos.y, pos.z]),
                    'orientation': np.array([orient.x, orient.y, orient.z, orient.w]),
                }
            
            msg = f"Pre-map set with {len(self.premap)} objects (indexing only)."
            rospy.loginfo(msg)
            return SetPremapResponse(success=True, message=msg)
            
        except Exception as e:
            msg = f"Failed to set pre-map: {e}"
            rospy.logerr(msg)
            return SetPremapResponse(success=False, message=msg)

    def run(self):
        """Main loop: broadcast transforms at configured rate."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.broadcast_transforms()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ProbabilisticObjectTracker()
        node.run()
    except rospy.ROSInterruptException:
        pass
