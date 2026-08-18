#!/usr/bin/env python3

"""Place tetra frames from bottom-camera bboxes and DVL altitude."""

import math
from collections import deque

import rospy

from utils.detection_utils import transform_to_odom_and_publish


class TetraBottomCameraHandler:
    def __init__(
        self,
        camera_config,
        id_tf_map,
        _props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    ):
        self.camera_frame = camera_config["frame"]
        self.image_width = camera_config.get("image_width", 1920)
        self.image_height = camera_config.get("image_height", 1080)
        self.id_tf_map = id_tf_map
        self.calibration = calibration
        self.tf_buffer = tf_buffer
        self.object_transform_pub = publishers["object_transform"]
        self.shared_state = shared_state
        self.stability_min_detections = int(
            camera_config.get("stability_min_detections", 5)
        )
        self.stability_radius_px = float(
            camera_config.get("stability_radius_px", 200.0)
        )
        if self.stability_min_detections < 1:
            raise ValueError("stability_min_detections must be at least 1")
        if self.stability_radius_px <= 0.0:
            raise ValueError("stability_radius_px must be positive")

        self._candidate_history = {}
        self._confirmed_ids = set()

    def handle(self, detection_msg):
        stamp = detection_msg.header.stamp
        processed_ids = set()
        for detection in detection_msg.detections.detections:
            if not detection.results:
                continue
            detection_id = detection.results[0].id
            if detection_id not in self.id_tf_map or detection_id in processed_ids:
                continue
            processed_ids.add(detection_id)

            stable_center = self._update_stability_filter(
                detection_id,
                (detection.bbox.center.x, detection.bbox.center.y),
            )
            if stable_center is None:
                continue

            distance = self.shared_state.get("altitude")
            if distance is None:
                rospy.logwarn_throttle(
                    5.0, "No altitude data for bottom camera tetra detection"
                )
                continue

            angles = self.calibration.calculate_angles(stable_center)
            offset_x = math.tan(angles[0]) * distance
            offset_y = math.tan(angles[1]) * distance
            if (offset_x**2 + offset_y**2 + distance**2) > 30**2:
                rospy.logdebug("Bottom-camera tetra detection is too far; skipping")
                continue

            transform_to_odom_and_publish(
                self.camera_frame,
                self.id_tf_map[detection_id],
                offset_x,
                offset_y,
                distance,
                stamp,
                self.tf_buffer,
                self.object_transform_pub,
            )

        missing_ids = set(self._candidate_history) - processed_ids
        for detection_id in missing_ids:
            self._candidate_history[detection_id].clear()
            self._confirmed_ids.discard(detection_id)

    def _update_stability_filter(self, detection_id, bbox_center):
        history = self._candidate_history.setdefault(detection_id, deque())

        if history:
            centroid = tuple(
                sum(sample[index] for sample in history) / len(history)
                for index in range(2)
            )
            distance_to_centroid = math.hypot(
                bbox_center[0] - centroid[0], bbox_center[1] - centroid[1]
            )
            if distance_to_centroid > self.stability_radius_px:
                history.clear()
                self._confirmed_ids.discard(detection_id)

        history.append(bbox_center)
        while len(history) > self.stability_min_detections:
            history.popleft()

        if len(history) < self.stability_min_detections:
            rospy.logdebug_throttle(
                1.0,
                "Bottom tetra stability: %d/%d nearby bboxes",
                len(history),
                self.stability_min_detections,
            )
            return None

        if detection_id not in self._confirmed_ids:
            self._confirmed_ids.add(detection_id)
            rospy.loginfo(
                "Bottom tetra detection confirmed after %d nearby bboxes",
                self.stability_min_detections,
            )

        return tuple(
            sum(sample[index] for sample in history) / len(history)
            for index in range(2)
        )


def create_handler(
    camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
):
    return TetraBottomCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
