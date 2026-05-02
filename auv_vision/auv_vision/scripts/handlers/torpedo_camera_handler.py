#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from ultralytics_ros.msg import YoloResult
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
)


class TorpedoCameraHandler:
    HOLE_FRAME_IDS = (
        "torpedo_hole_left_link",
        "torpedo_hole_right_link",
        "torpedo_hole_bottom_link",
        "torpedo_hole_top_link",
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

        self.hole_focus = "none"

    def set_hole_focus(self, focus: str):
        focus = focus.strip().lower()
        if focus not in ("top", "bottom", "none"):
            message = "Invalid torpedo hole focus. Expected one of: top, bottom, none."
            rospy.logwarn(message)
            return False, message

        self.hole_focus = focus
        message = f"Torpedo hole focus set to: {self.hole_focus}"
        rospy.loginfo(message)
        return True, message

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        self._process_torpedo_holes(detection_msg, stamp)

    def _process_torpedo_holes(self, detection_msg: YoloResult, stamp):
        detected_holes = self._get_detected_holes(detection_msg)

        if len(detected_holes) == 4:
            assignments = self._assign_all_holes(detected_holes)
        elif len(detected_holes) in (2, 3):
            assignments = (
                self._assign_nearest_existing_holes(detected_holes, stamp)
                if self.hole_focus == "none"
                else self._assign_focused_holes(detected_holes)
            )
        elif len(detected_holes) == 1:
            assignments = self._assign_nearest_existing_holes(detected_holes, stamp)
        else:
            return

        self._publish_hole_assignments(assignments, stamp)

    def _get_detected_holes(self, detection_msg: YoloResult):
        detected_holes = []
        target_detection_id = self.id_tf_map.id_of("torpedo_hole_link")

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue

            if detection.results[0].id != target_detection_id:
                continue

            if check_inside_image(detection, self.image_width, self.image_height):
                detected_holes.append(detection)

        return detected_holes

    def _assign_all_holes(self, detected_holes):
        remaining_holes = list(detected_holes)

        left_hole = min(remaining_holes, key=self._bbox_center_x)
        remaining_holes.remove(left_hole)

        right_hole = max(remaining_holes, key=self._bbox_center_x)
        remaining_holes.remove(right_hole)

        top_hole = min(remaining_holes, key=self._bbox_center_y)
        bottom_hole = max(remaining_holes, key=self._bbox_center_y)

        return {
            "torpedo_hole_left_link": left_hole,
            "torpedo_hole_right_link": right_hole,
            "torpedo_hole_bottom_link": bottom_hole,
            "torpedo_hole_top_link": top_hole,
        }

    def _assign_focused_holes(self, detected_holes):
        if len(detected_holes) == 2:
            hole_names = (
                ("left", "top") if self.hole_focus == "top" else ("bottom", "right")
            )
            sorted_holes = sorted(detected_holes, key=self._bbox_center_x)
            return {
                self._frame_for_hole(hole_name): detection
                for hole_name, detection in zip(hole_names, sorted_holes)
            }

        focus_top = self.hole_focus == "top"
        holes_by_focus_y = sorted(
            enumerate(detected_holes),
            key=lambda indexed_hole: self._bbox_center_y(indexed_hole[1]),
            reverse=not focus_top,
        )
        focus_pair = [indexed_hole[1] for indexed_hole in holes_by_focus_y[:2]]
        remaining_hole = holes_by_focus_y[2][1]
        left_side_hole, right_side_hole = sorted(focus_pair, key=self._bbox_center_x)

        if focus_top:
            return {
                "torpedo_hole_left_link": left_side_hole,
                "torpedo_hole_top_link": right_side_hole,
                "torpedo_hole_bottom_link": remaining_hole,
            }

        return {
            "torpedo_hole_bottom_link": left_side_hole,
            "torpedo_hole_right_link": right_side_hole,
            "torpedo_hole_top_link": remaining_hole,
        }

    def _assign_nearest_existing_hole(self, detection, stamp):
        hole_positions = self._lookup_hole_positions(stamp)
        if not hole_positions:
            rospy.logwarn_throttle(
                5,
                "Could not read any torpedo hole frames. Skipping nearest-hole assignment.",
            )
            return {}

        nearest_frame = self._nearest_hole_frame(detection, hole_positions, stamp)
        return {} if nearest_frame is None else {nearest_frame: detection}

    def _assign_nearest_existing_holes(self, detected_holes, stamp):
        assignments = []
        for detection in detected_holes:
            assignments.extend(
                self._assign_nearest_existing_hole(detection, stamp).items()
            )
        return assignments

    def _publish_hole_assignments(self, assignments, stamp):
        assignment_items = (
            assignments.items() if hasattr(assignments, "items") else assignments
        )
        for child_frame_id, detection in assignment_items:
            self._publish_hole_transform(detection, child_frame_id, stamp)

    def _publish_hole_transform(self, detection, child_frame_id, stamp):
        """Estimate distance and publish transform for a single torpedo hole."""
        prop = self._prop_for_hole_frame(child_frame_id)
        if not prop:
            rospy.logerr(f"Prop for '{child_frame_id}' not found.")
            return None

        distance = prop.estimate_distance(
            detection.bbox.size_y,
            detection.bbox.size_x,
            self.calibration,
        )

        if distance is None:
            return None

        _, offset_x, offset_y = calculate_angles_and_offsets(
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

    def _nearest_hole_frame(self, detection, hole_positions, stamp):
        nearest_hole_frame = None
        nearest_distance = None

        for hole_frame, hole_position in hole_positions.items():
            projected_position = self._project_hole_to_odom(
                detection, stamp, hole_frame
            )
            if projected_position is None:
                continue

            distance_squared = self._distance_squared(projected_position, hole_position)
            if nearest_distance is None or distance_squared < nearest_distance:
                nearest_hole_frame = hole_frame
                nearest_distance = distance_squared

        return nearest_hole_frame

    def _project_hole_to_odom(self, detection, stamp, reference_frame):
        prop = self._prop_for_hole_frame(reference_frame)
        if not prop:
            rospy.logerr(f"Prop for '{reference_frame}' not found.")
            return None

        distance = prop.estimate_distance(
            detection.bbox.size_y,
            detection.bbox.size_x,
            self.calibration,
        )
        if distance is None:
            return None

        _, offset_x, offset_y = calculate_angles_and_offsets(
            self.calibration, detection.bbox.center, distance
        )
        return self._transform_camera_point_to_odom(offset_x, offset_y, distance, stamp)

    def _transform_camera_point_to_odom(self, offset_x, offset_y, distance, stamp):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.camera_frame
        pose_stamped.pose.position = Point(offset_x, offset_y, distance)
        pose_stamped.pose.orientation = Quaternion(0, 0, 0, 1)

        try:
            transformed_pose_stamped = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(4.0)
            )
            return transformed_pose_stamped.pose.position
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Transform error for torpedo hole: {e}")
            return None

    def _lookup_hole_positions(self, stamp):
        hole_positions = {}

        for frame_id in self.HOLE_FRAME_IDS:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "odom",
                    frame_id,
                    stamp,
                    rospy.Duration(1.0),
                )
                hole_positions[frame_id] = transform.transform.translation
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(5.0, f"Transform error for {frame_id}: {e}")
                continue

        return hole_positions

    def _prop_for_hole_frame(self, frame_id):
        return self.props.get(frame_id) or self.props.get("torpedo_hole_link")

    @staticmethod
    def _frame_for_hole(hole_name):
        return f"torpedo_hole_{hole_name}_link"

    @staticmethod
    def _bbox_center_x(detection):
        return detection.bbox.center.x

    @staticmethod
    def _bbox_center_y(detection):
        return detection.bbox.center.y

    @staticmethod
    def _distance_squared(point_a, point_b):
        return (
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
            + (point_a.z - point_b.z) ** 2
        )


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
