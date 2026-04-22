#!/usr/bin/env python3

import rospy
from ultralytics_ros.msg import YoloResult
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
)


class TorpedoCameraHandler:
    HOLE_FRAME_IDS = (
        "torpedo_hole_left_mid_link",
        "torpedo_hole_bottom_right_link",
        "torpedo_hole_bottom_mid_link",
        "torpedo_hole_top_mid_link",
    )

    def __init__(
        self,
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    ):
        self.camera_ns = camera_config["ns"]
        self.camera_frame = camera_config["frame"]
        self.image_width = camera_config.get("image_width", 640)
        self.image_height = camera_config.get("image_height", 480)
        self.id_tf_map = id_tf_map
        self.props = props
        self.calibration = calibration
        self.tf_buffer = tf_buffer
        self.object_transform_pub = publishers["object_transform"]
        self.props_yaw_pub = publishers["props_yaw"]
        self.shared_state = shared_state

        self.torpedo_hole_props = {
            frame_id: self.props.get(frame_id) for frame_id in self.HOLE_FRAME_IDS
        }
        self.tracked_holes = {}
        self.bootstrap_holes_required = camera_config.get("bootstrap_holes_required", 4)
        self.hole_tracking_min_distance_px = camera_config.get(
            "hole_tracking_min_distance_px", 60.0
        )
        self.hole_tracking_distance_scale = camera_config.get(
            "hole_tracking_distance_scale", 0.9
        )
        self.hole_tracking_min_iou = camera_config.get("hole_tracking_min_iou", 0.05)
        self.hole_tracking_iou_weight = camera_config.get(
            "hole_tracking_iou_weight", 0.35
        )

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        self._process_torpedo_holes(detection_msg, stamp)

    def _process_torpedo_holes(self, detection_msg: YoloResult, stamp):
        detected_holes = self._get_detected_holes(detection_msg)

        if len(detected_holes) == self.bootstrap_holes_required:
            assignments = self._assign_labels_from_layout(detected_holes)
        elif self.tracked_holes:
            assignments = self._assign_labels_from_tracking(detected_holes)
        else:
            rospy.logwarn_throttle(
                5.0,
                "Torpedo holes need an initial 4-hole view before tracking can continue.",
            )
            return

        if not assignments:
            rospy.logwarn_throttle(
                5.0,
                f"Could not match {len(detected_holes)} torpedo hole detection(s) to tracked holes.",
            )
            return

        for child_frame_id, detection in assignments.items():
            self._update_tracked_hole(child_frame_id, detection)
            self._publish_hole_transform(detection, child_frame_id, stamp)

    def _get_detected_holes(self, detection_msg: YoloResult):
        detected_holes = []
        target_detection_id = self.id_tf_map.id_of("torpedo_hole_link")

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id != target_detection_id:
                continue

            if not check_inside_image(detection, self.image_width, self.image_height):
                continue

            detected_holes.append(detection)

        return detected_holes

    def _assign_labels_from_layout(self, detected_holes):
        selected_holes = sorted(
            detected_holes,
            key=lambda detection: detection.bbox.size_x * detection.bbox.size_y,
            reverse=True,
        )[: self.bootstrap_holes_required]

        remaining_holes = list(selected_holes)

        left_mid_hole = min(remaining_holes, key=self._bbox_center_x)
        remaining_holes.remove(left_mid_hole)

        bottom_right_hole = max(remaining_holes, key=self._bbox_center_x)
        remaining_holes.remove(bottom_right_hole)

        top_mid_hole = min(remaining_holes, key=self._bbox_center_y)
        bottom_mid_hole = max(remaining_holes, key=self._bbox_center_y)

        return {
            "torpedo_hole_left_mid_link": left_mid_hole,
            "torpedo_hole_bottom_right_link": bottom_right_hole,
            "torpedo_hole_bottom_mid_link": bottom_mid_hole,
            "torpedo_hole_top_mid_link": top_mid_hole,
        }

    def _assign_labels_from_tracking(self, detected_holes):
        if not detected_holes:
            return {}

        candidates = []
        for child_frame_id, previous_bbox in self.tracked_holes.items():
            for detection_idx, detection in enumerate(detected_holes):
                center_distance = self._bbox_center_distance(detection, previous_bbox)
                current_diag = self._bbox_diag(detection)
                previous_diag = self._bbox_diag(previous_bbox)
                adaptive_threshold = max(
                    self.hole_tracking_min_distance_px,
                    self.hole_tracking_distance_scale
                    * max(current_diag, previous_diag),
                )
                iou = self._bbox_iou(detection, previous_bbox)

                if (
                    center_distance > adaptive_threshold
                    and iou < self.hole_tracking_min_iou
                ):
                    continue

                normalized_distance = center_distance / max(
                    max(current_diag, previous_diag), 1.0
                )
                cost = normalized_distance - (self.hole_tracking_iou_weight * iou)
                candidates.append(
                    (cost, center_distance, child_frame_id, detection_idx)
                )

        candidates.sort(key=lambda item: (item[0], item[1]))

        matched_labels = set()
        matched_detection_indices = set()
        assignments = {}

        for _, _, child_frame_id, detection_idx in candidates:
            if (
                child_frame_id in matched_labels
                or detection_idx in matched_detection_indices
            ):
                continue

            matched_labels.add(child_frame_id)
            matched_detection_indices.add(detection_idx)
            assignments[child_frame_id] = detected_holes[detection_idx]

        return assignments

    def _update_tracked_hole(self, child_frame_id, detection):
        self.tracked_holes[child_frame_id] = {
            "center_x": detection.bbox.center.x,
            "center_y": detection.bbox.center.y,
            "size_x": detection.bbox.size_x,
            "size_y": detection.bbox.size_y,
        }

    def _publish_hole_transform(self, detection, child_frame_id, stamp):
        prop = self.torpedo_hole_props.get(child_frame_id)
        if not prop:
            rospy.logerr(f"Prop for '{child_frame_id}' not found.")
            return

        distance = prop.estimate_distance(
            detection.bbox.size_y,
            detection.bbox.size_x,
            self.calibration,
        )

        if distance is None:
            return

        angles, offset_x, offset_y = calculate_angles_and_offsets(
            self.calibration, detection.bbox.center, distance
        )

        transform_to_odom_and_publish(
            self.camera_frame,
            child_frame_id,
            offset_x,
            offset_y,
            distance,
            stamp,
            self.tf_buffer,
            self.object_transform_pub,
        )

    @staticmethod
    def _bbox_center_x(detection):
        return detection.bbox.center.x

    @staticmethod
    def _bbox_center_y(detection):
        return detection.bbox.center.y

    @staticmethod
    def _bbox_diag(detection_or_bbox):
        if hasattr(detection_or_bbox, "bbox"):
            bbox = detection_or_bbox.bbox
            size_x = bbox.size_x
            size_y = bbox.size_y
        else:
            size_x = detection_or_bbox["size_x"]
            size_y = detection_or_bbox["size_y"]

        return (size_x**2 + size_y**2) ** 0.5

    @staticmethod
    def _bbox_center_distance(detection, previous_bbox):
        dx = detection.bbox.center.x - previous_bbox["center_x"]
        dy = detection.bbox.center.y - previous_bbox["center_y"]
        return (dx**2 + dy**2) ** 0.5

    @staticmethod
    def _bbox_iou(detection, previous_bbox):
        current_left = detection.bbox.center.x - (detection.bbox.size_x * 0.5)
        current_right = detection.bbox.center.x + (detection.bbox.size_x * 0.5)
        current_top = detection.bbox.center.y - (detection.bbox.size_y * 0.5)
        current_bottom = detection.bbox.center.y + (detection.bbox.size_y * 0.5)

        previous_left = previous_bbox["center_x"] - (previous_bbox["size_x"] * 0.5)
        previous_right = previous_bbox["center_x"] + (previous_bbox["size_x"] * 0.5)
        previous_top = previous_bbox["center_y"] - (previous_bbox["size_y"] * 0.5)
        previous_bottom = previous_bbox["center_y"] + (previous_bbox["size_y"] * 0.5)

        inter_left = max(current_left, previous_left)
        inter_right = min(current_right, previous_right)
        inter_top = max(current_top, previous_top)
        inter_bottom = min(current_bottom, previous_bottom)

        if inter_left >= inter_right or inter_top >= inter_bottom:
            return 0.0

        intersection_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        current_area = detection.bbox.size_x * detection.bbox.size_y
        previous_area = previous_bbox["size_x"] * previous_bbox["size_y"]
        union_area = current_area + previous_area - intersection_area

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area


def create_handler(
    camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
):
    return TorpedoCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
