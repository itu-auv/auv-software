#!/usr/bin/env python3

"""Place tetra frames from bottom-camera bboxes and DVL altitude."""

import rospy

from utils.detection_utils import (
    calculate_angles_and_offsets,
    check_inside_image_bottom_bin,
    transform_to_odom_and_publish,
)


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

    def handle(self, detection_msg):
        stamp = detection_msg.header.stamp
        #print("annemi çok seviyorum.")
        for detection in detection_msg.detections.detections:
            if not detection.results:
                continue
            detection_id = detection.results[0].id
            if detection_id not in self.id_tf_map:
                continue

            distance = self.shared_state.get("altitude")
            if distance is None:
                rospy.logwarn_throttle(
                    5.0, "No altitude data for bottom camera tetra detection"
                )
                continue

            _, offset_x, offset_y = calculate_angles_and_offsets(
                self.calibration, detection.bbox.center, distance
            )
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
