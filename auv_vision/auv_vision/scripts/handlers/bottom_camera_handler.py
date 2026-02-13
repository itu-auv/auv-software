#!/usr/bin/env python3

import rospy
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
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
        self.altitude_distance_ids = [10, 11]  # bin_shark, bin_sawfish

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id not in self.id_tf_map:
                continue

            prop_name = self.id_tf_map[detection_id]
            if prop_name not in self.props:
                continue

            prop = self.props[prop_name]

            # Bin detections use altitude for distance and skip inside-image check
            if detection_id in self.altitude_distance_ids:
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
