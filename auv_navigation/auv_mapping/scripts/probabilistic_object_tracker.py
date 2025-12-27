#!/usr/bin/env python3

import rospy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from norfair import Detection, Tracker
from norfair.tracker import TrackedObject

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
import tf2_geometry_msgs


class ProbabilisticObjectTracker:

    def __init__(self):
        rospy.init_node("probabilistic_object_tracker", anonymous=False)
        rospy.loginfo("Probabilistic Object Tracker starting...")

        # Parameters
        self.world_frame = rospy.get_param("~world_frame", "odom")
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.distance_threshold = rospy.get_param("~distance_threshold", 4.0) 
        self.hit_counter_max = rospy.get_param("~hit_counter_max", 5)
        self.initialization_delay = rospy.get_param("~initialization_delay", 4)
        self.pointwise_hit_counter_max = rospy.get_param("~pointwise_hit_counter_max", 4)

        # Slalom-specific parameters
        self.slalom_distance_threshold = rospy.get_param("~slalom_distance_threshold", 1.0)
        self.slalom_labels = ["red_pipe_link", "white_pipe_link"]

        # Per-class trackers (each object class has its own tracker)
        self.trackers: Dict[str, Tracker] = {}

        # TF
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscriber
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

        # store orientation (norfair tracks position only)
        # TODO: Implement SLERP filtering for orientation (like old C++ system)
        #       Currently just stores last seen value, should interpolate with alpha=0.2
        self.track_orientations: Dict[int, Quaternion] = {}

        rospy.loginfo(
            f"Tracker initialized. "
            f"distance_threshold={self.distance_threshold}m, "
            f"hit_counter_max={self.hit_counter_max}"
        )

    def get_or_create_tracker(self, label: str) -> Tracker:
        """Get existing tracker for label or create a new one."""
        if label not in self.trackers:
            # Use tighter threshold for slalom gates
            threshold = self.slalom_distance_threshold if label in self.slalom_labels else self.distance_threshold

            self.trackers[label] = Tracker(
                distance_function=self.euclidean_distance,
                distance_threshold=threshold,
                hit_counter_max=self.hit_counter_max,
                initialization_delay=self.initialization_delay,
                pointwise_hit_counter_max=self.pointwise_hit_counter_max,
                past_detections_length=10,
            )
            rospy.loginfo(f"Created new tracker for '{label}' with threshold={threshold}m")

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

        # get the tracker for the class
        tracker = self.get_or_create_tracker(label)
        tracked_objects = tracker.update(detections=[detection])
        
        for obj in tracked_objects:
            if obj.last_detection is not None:
                self.track_orientations[obj.id] = msg.transform.rotation

    def broadcast_transforms(self):
        """Broadcast TF for all confirmed tracks."""
        tracks_by_label = self._get_confirmed_tracks()

        for label, tracks in tracks_by_label.items():
            tracks_sorted = self.sort_tracks_by_distance(tracks)

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

        if obj.id in self.track_orientations:
            tf_msg.transform.rotation = self.track_orientations[obj.id]
        else:
            tf_msg.transform.rotation = Quaternion(x=0, y=0, z=0, w=1)

        return tf_msg

    def sort_tracks_by_distance(self, tracks: List[TrackedObject]) -> List[TrackedObject]:
        """Sort tracks by distance to base_link (closest first)."""
        base_link_frame = "taluy/base_link"

        def get_distance(obj: TrackedObject) -> float:
            if obj.estimate is None or len(obj.estimate) == 0:
                return float('inf')

            est = obj.estimate[0]
            try:
                # Look up transform from world_frame to base_link
                transform = self.tf_buffer.lookup_transform(
                    base_link_frame, self.world_frame, rospy.Time(0), rospy.Duration(0.1)
                )
                # Transform point to base_link
                from geometry_msgs.msg import PointStamped
                point = PointStamped()
                point.header.frame_id = self.world_frame
                point.point.x, point.point.y, point.point.z = est[0], est[1], est[2]

                point_in_base = tf2_geometry_msgs.do_transform_point(point, transform)
                return np.sqrt(
                    point_in_base.point.x**2 +
                    point_in_base.point.y**2 +
                    point_in_base.point.z**2
                )
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                # Fallback: distance from origin
                return np.sqrt(est[0]**2 + est[1]**2 + est[2]**2)

        return sorted(tracks, key=get_distance)

    def clear_tracks_handler(self, req) -> TriggerResponse:
        """Service handler to clear all tracks."""
        self.trackers.clear()
        self.track_orientations.clear()
        rospy.loginfo("Cleared all object tracks.")
        return TriggerResponse(success=True, message="Cleared all object tracks.")

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
