#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TransformStamped, Vector3
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
)


class TorpedoCameraHandler:
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

        self.reference_hole_frames = [
            "torpedo_reference_hole_left_mid_link",
            "torpedo_reference_hole_bottom_right_link",
            "torpedo_reference_hole_bottom_mid_link",
            "torpedo_reference_hole_top_mid_link",
        ]
        self.reference_holes_ready = False

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        self._process_torpedo_holes(detection_msg, stamp)

    def _process_torpedo_holes(self, detection_msg: YoloResult, stamp):
        detected_holes = []

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id == self.id_tf_map.id_of("torpedo_hole_link"):
                if check_inside_image(detection, self.image_width, self.image_height):
                    detected_holes.append(detection)

        self.reference_holes_ready = self._can_transform_all_reference_holes(stamp)
        if len(detected_holes) == len(self.reference_hole_frames):
            self._publish_reference_holes(detected_holes, stamp)
        if not self.reference_holes_ready:
            return

        reference_positions = self._lookup_reference_hole_positions(stamp)
        if len(reference_positions) != len(self.reference_hole_frames):
            rospy.logwarn_throttle(
                5,
                "Could not read all torpedo reference hole frames. Skipping close holes.",
            )
            return

        for detection in detected_holes:
            nearest_reference_frame = self._nearest_reference_frame(
                detection, reference_positions, stamp
            )
            if nearest_reference_frame is None:
                continue

            close_frame = self._close_frame_for_reference(nearest_reference_frame)
            self._publish_hole_transform(detection, close_frame, stamp)

    def _publish_reference_holes(self, detected_holes, stamp):
        if len(detected_holes) != len(self.reference_hole_frames):
            rospy.logwarn_throttle(
                5,
                f"Expected {len(self.reference_hole_frames)} torpedo holes, but found {len(detected_holes)}. Skipping.",
            )
            return

        remaining_holes = list(detected_holes)

        left_mid_hole = min(remaining_holes, key=self._bbox_center_x)
        remaining_holes.remove(left_mid_hole)

        bottom_right_hole = max(remaining_holes, key=self._bbox_center_x)
        remaining_holes.remove(bottom_right_hole)

        top_mid_hole = min(remaining_holes, key=self._bbox_center_y)
        bottom_mid_hole = max(remaining_holes, key=self._bbox_center_y)

        assignments = [
            ("torpedo_reference_hole_left_mid_link", left_mid_hole),
            ("torpedo_reference_hole_bottom_right_link", bottom_right_hole),
            ("torpedo_reference_hole_bottom_mid_link", bottom_mid_hole),
            ("torpedo_reference_hole_top_mid_link", top_mid_hole),
        ]

        for child_frame_id, detection in assignments:
            self._publish_hole_transform(detection, child_frame_id, stamp)

        self.reference_holes_ready = self._can_transform_all_reference_holes(stamp)
        if self.reference_holes_ready:
            rospy.loginfo("All torpedo reference hole frames are available in TF.")

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

        return self._publish_odom_transform(
            child_frame_id, offset_x, offset_y, distance, stamp
        )

    def _nearest_reference_frame(self, detection, reference_positions, stamp):
        nearest_reference_frame = None
        nearest_distance = None

        for reference_frame, reference_position in reference_positions.items():
            projected_position = self._project_hole_to_odom(
                detection, stamp, reference_frame
            )
            if projected_position is None:
                continue

            distance_squared = self._distance_squared(
                projected_position, reference_position
            )
            if nearest_distance is None or distance_squared < nearest_distance:
                nearest_reference_frame = reference_frame
                nearest_distance = distance_squared

        return nearest_reference_frame

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

    def _publish_odom_transform(
        self, child_frame_id, offset_x, offset_y, distance, stamp
    ):
        position = self._transform_camera_point_to_odom(
            offset_x, offset_y, distance, stamp
        )
        if position is None:
            return None

        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = stamp
        transform_stamped_msg.header.frame_id = "odom"
        transform_stamped_msg.child_frame_id = child_frame_id
        transform_stamped_msg.transform.translation = Vector3(
            position.x, position.y, position.z
        )
        transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
        self.object_transform_pub.publish(transform_stamped_msg)
        return position

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

    def _lookup_reference_hole_positions(self, stamp):
        reference_positions = {}

        for frame_id in self.reference_hole_frames:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "odom",
                    frame_id,
                    stamp,
                    rospy.Duration(1.0),
                )
                reference_positions[frame_id] = transform.transform.translation
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(5.0, f"Transform error for {frame_id}: {e}")
                continue

        return reference_positions

    def _can_transform_all_reference_holes(self, stamp):
        return all(
            self.tf_buffer.can_transform("odom", frame_id, stamp, rospy.Duration(0.1))
            for frame_id in self.reference_hole_frames
        )

    def _prop_for_hole_frame(self, frame_id):
        prop_name = frame_id
        for prefix in ("torpedo_reference_", "torpedo_close_"):
            if prop_name.startswith(prefix):
                prop_name = prop_name.replace(prefix, "torpedo_", 1)
                break
        return self.props.get(prop_name) or self.props.get("torpedo_hole_link")

    @staticmethod
    def _close_frame_for_reference(reference_frame):
        return reference_frame.replace("torpedo_reference_", "torpedo_close_", 1)

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
