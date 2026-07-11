#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
    check_inside_image_bottom_bin,
)


class BottomCameraHandler:
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

        # IDs that use altitude for distance instead of prop size estimation
        self.altitude_distance_ids = self.id_tf_map.ids_of(
            "bin_blood_link", "bin_fire_link", "octagon_table_link"
        )

        # Bottom camera specific state
        self.active_ids = self.id_tf_map.ids_of("bin_blood_link", "bin_fire_link")
        self.bin_frames = {
            self.id_tf_map.id_of("bin_fire_link"): (
                "bin_fire_link",
                "bin_fire_first",
                "bin_fire_second",
            ),
            self.id_tf_map.id_of("bin_blood_link"): (
                "bin_blood_link",
                "bin_blood_first",
                "bin_blood_second",
            ),
        }

    def set_active_ids(self, ids: list):
        """Called by orchestrator when set_bottom_camera_focus service is triggered."""
        self.active_ids = ids

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp
        bin_detections = {detection_id: [] for detection_id in self.bin_frames}

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id not in self.active_ids:
                continue

            if detection_id not in self.id_tf_map:
                continue

            prop_name = self.id_tf_map[detection_id]
            if prop_name not in self.props:
                continue

            prop = self.props[prop_name]

            if detection_id in self.bin_frames:
                if not check_inside_image_bottom_bin(
                    detection, self.image_width, self.image_height
                ):
                    continue
                bin_detections[detection_id].append(detection)

            if detection_id in self.altitude_distance_ids:
                if not check_inside_image_bottom_bin(
                    detection, self.image_width, self.image_height
                ):
                    continue
                distance = self.shared_state.get("altitude")
                if distance is None:
                    rospy.logwarn_throttle(
                        5, "No altitude data for bottom camera bin detection"
                    )
                    continue
            else:
                if not check_inside_image(
                    detection, self.image_width, self.image_height
                ):
                    continue
                distance = prop.estimate_distance(
                    detection.bbox.size_y,
                    detection.bbox.size_x,
                    self.calibration,
                )

            if distance is None:
                continue

            angles, offset_x, offset_y = calculate_angles_and_offsets(
                self.calibration, detection.bbox.center, distance
            )

            # Publish props yaw
            props_yaw_msg = PropsYaw()
            props_yaw_msg.header.stamp = stamp
            props_yaw_msg.object = prop.name
            props_yaw_msg.angle = -angles[0]
            self.props_yaw_pub.publish(props_yaw_msg)

            # Max distance check
            if (offset_x**2 + offset_y**2 + distance**2) > 30**2:
                rospy.logdebug(f"Detection for {prop_name} is too far away. Skipping.")
                continue

            transform_to_odom_and_publish(
                self.camera_frame,
                prop_name,
                offset_x,
                offset_y,
                distance,
                stamp,
                self.tf_buffer,
                self.object_transform_pub,
            )

        for detection_id, detections in bin_detections.items():
            self._process_bin_detections(detection_id, detections, stamp)

    def _process_bin_detections(self, detection_id, detections, stamp):
        prop_name, first_frame, second_frame = self.bin_frames[detection_id]
        detections = detections[:2]

        if len(detections) == 1:
            child_frame_id = first_frame
            if self.tf_buffer.can_transform(
                "odom", first_frame, stamp, rospy.Duration(0.1)
            ) and self.tf_buffer.can_transform(
                "odom", second_frame, stamp, rospy.Duration(0.1)
            ):
                child_frame_id = min(
                    (first_frame, second_frame),
                    key=lambda frame: self._distance_to_frame(
                        detections[0], frame, stamp
                    ),
                )
            self._publish_bin_detection(detections[0], prop_name, child_frame_id, stamp)
            return

        if len(detections) < 2:
            return

        first_detection, second_detection = detections
        if self.tf_buffer.can_transform(
            "odom", first_frame, stamp, rospy.Duration(0.1)
        ):
            first_distance = self._distance_to_frame(
                first_detection, first_frame, stamp
            )
            second_distance = self._distance_to_frame(
                second_detection, first_frame, stamp
            )
            if second_distance < first_distance:
                first_detection, second_detection = second_detection, first_detection

        self._publish_bin_detection(first_detection, prop_name, first_frame, stamp)
        self._publish_bin_detection(second_detection, prop_name, second_frame, stamp)

    def _distance_to_frame(self, detection, frame_id, stamp):
        detection_position = self._bin_position(detection, stamp)

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", frame_id, stamp, rospy.Duration(1.0)
            )
            frame_position = transform.transform.translation
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Transform error for {frame_id}: {e}")
            return float("inf")

        if detection_position is None:
            return float("inf")

        return (
            (detection_position.x - frame_position.x) ** 2
            + (detection_position.y - frame_position.y) ** 2
            + (detection_position.z - frame_position.z) ** 2
        )

    def _publish_bin_detection(self, detection, prop_name, child_frame_id, stamp):
        offsets = self._bin_offsets(detection)
        if offsets is None:
            return
        angles, offset_x, offset_y, distance = offsets

        props_yaw_msg = PropsYaw()
        props_yaw_msg.header.stamp = stamp
        props_yaw_msg.object = self.props[prop_name].name
        props_yaw_msg.angle = -angles[0]
        self.props_yaw_pub.publish(props_yaw_msg)

        if (offset_x**2 + offset_y**2 + distance**2) > 30**2:
            rospy.logdebug(f"Detection for {prop_name} is too far away. Skipping.")
            return

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

    def _bin_position(self, detection, stamp):
        offsets = self._bin_offsets(detection)
        if offsets is None:
            return None
        _, offset_x, offset_y, distance = offsets

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.camera_frame
        pose_stamped.pose.position = Point(offset_x, offset_y, distance)
        pose_stamped.pose.orientation = Quaternion(0, 0, 0, 1)

        try:
            return self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(4.0)
            ).pose.position
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            return None

    def _bin_offsets(self, detection):
        distance = self.shared_state.get("altitude")
        if distance is None:
            rospy.logwarn_throttle(
                5, "No altitude data for bottom camera bin detection"
            )
            return None

        angles, offset_x, offset_y = calculate_angles_and_offsets(
            self.calibration, detection.bbox.center, distance
        )
        return angles, offset_x, offset_y, distance


def create_handler(
    camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
):
    return BottomCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
