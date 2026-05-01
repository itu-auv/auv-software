#!/usr/bin/env python3

import rospy
from ultralytics_ros.msg import YoloResult
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
)
from geometry_msgs.msg import PoseStamped
import tf2_ros


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
        self.tracked_holes_odom = {}
        self.bootstrap_holes_required = camera_config.get("bootstrap_holes_required", 4)
        self.hole_tracking_min_distance_px = camera_config.get(
            "hole_tracking_min_distance_px", 60.0
        )
        self.hole_tracking_distance_scale = camera_config.get(
            "hole_tracking_distance_scale", 0.9
        )

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        self._process_torpedo_holes(detection_msg, stamp)

    def _process_torpedo_holes(self, detection_msg: YoloResult, stamp):
        detected_holes = self._get_detected_holes(detection_msg)

        if len(detected_holes) == self.bootstrap_holes_required:
            assignments = self._assign_labels_from_layout(detected_holes)
        elif self.tracked_holes_odom:
            assignments = self._assign_labels_from_tracking_3d(detected_holes, stamp)
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
            self._publish_hole_transform(detection, child_frame_id, stamp)

    def _get_detected_holes(self, detection_msg: YoloResult):
        detected_holes = []
        target_detection_id = self.id_tf_map.id_of("torpedo_hole_link")

        for detection in detection_msg.detections.detections:
            if not detection.results:
                continue
            if detection.results[0].id != target_detection_id:
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

    def _assign_labels_from_tracking_3d(self, detected_holes, stamp):
        if not detected_holes:
            return {}

        projected_points_2d = {}
        fx = self.calibration.calibration.K[0]
        fy = self.calibration.calibration.K[4]
        cx = self.calibration.calibration.K[2]
        cy = self.calibration.calibration.K[5]

        # Project 3D odom points to camera pixels
        for child_frame_id, odom_point in self.tracked_holes_odom.items():
            try:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = stamp
                pose_stamped.header.frame_id = "odom"
                pose_stamped.pose.position = odom_point
                pose_stamped.pose.orientation.w = 1.0

                transformed_pose = self.tf_buffer.transform(
                    pose_stamped, self.camera_frame, rospy.Duration(0.1)
                )
                p = transformed_pose.pose.position
                if p.z > 0:
                    u = (p.x * fx / p.z) + cx
                    v = (p.y * fy / p.z) + cy
                    projected_points_2d[child_frame_id] = (u, v)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    2.0, f"Failed to project {child_frame_id} from odom: {e}"
                )

        candidates = []
        for child_frame_id, proj_pt in projected_points_2d.items():
            for detection_idx, detection in enumerate(detected_holes):
                center = detection.bbox.center
                dist = (
                    (center.x - proj_pt[0]) ** 2 + (center.y - proj_pt[1]) ** 2
                ) ** 0.5

                adaptive_threshold = max(
                    self.hole_tracking_min_distance_px * 2.0,
                    self.hole_tracking_distance_scale
                    * self._bbox_diag(detection)
                    * 2.0,
                )

                if dist <= adaptive_threshold:
                    candidates.append((dist, child_frame_id, detection_idx))

        candidates.sort(key=lambda item: item[0])
        assignments = {}
        matched_detection_indices = set()

        for dist, child_frame_id, detection_idx in candidates:
            if (
                child_frame_id in assignments
                or detection_idx in matched_detection_indices
            ):
                continue
            matched_detection_indices.add(detection_idx)
            assignments[child_frame_id] = detected_holes[detection_idx]

        return assignments

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

        final_transform = transform_to_odom_and_publish(
            self.camera_frame,
            child_frame_id,
            offset_x,
            offset_y,
            distance,
            stamp,
            self.tf_buffer,
            self.object_transform_pub,
        )

        if final_transform:
            self.tracked_holes_odom[child_frame_id] = (
                final_transform.transform.translation
            )

    @staticmethod
    def _bbox_center_x(detection):
        return detection.bbox.center.x

    @staticmethod
    def _bbox_center_y(detection):
        return detection.bbox.center.y

    @staticmethod
    def _bbox_diag(detection):
        return (detection.bbox.size_x**2 + detection.bbox.size_y**2) ** 0.5


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
