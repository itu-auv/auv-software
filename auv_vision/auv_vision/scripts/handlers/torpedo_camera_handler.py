#!/usr/bin/env python3

import rospy
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

        # Torpedo hole props for distance estimation
        self.torpedo_hole_props = {
            "torpedo_hole_shark_link": self.props.get("torpedo_hole_link"),
            "torpedo_hole_sawfish_link": self.props.get("torpedo_hole_link"),
        }

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        # First: process torpedo holes (ID=5) with special 2-hole logic
        self._process_torpedo_holes(detection_msg, stamp)

        # Then: standard detection loop for other IDs (e.g., ID=4 torpedo_map)
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            # Skip torpedo holes — already processed above
            if detection_id == 5:
                continue

            if detection_id not in self.id_tf_map:
                continue

            if not check_inside_image(detection, self.image_width, self.image_height):
                continue

            prop_name = self.id_tf_map[detection_id]
            if prop_name not in self.props:
                continue

            prop = self.props[prop_name]
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

    def _process_torpedo_holes(self, detection_msg: YoloResult, stamp):
        """Find exactly 2 torpedo holes, assign upper/bottom and shark/sawfish, publish transforms."""
        detected_holes = []

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id == 5:  # Torpedo hole ID
                if check_inside_image(detection, self.image_width, self.image_height):
                    detected_holes.append(detection)

        if len(detected_holes) != 2:
            rospy.logwarn_throttle(
                5,
                f"Expected 2 torpedo holes, but found {len(detected_holes)}. Skipping.",
            )
            return

        hole1 = detected_holes[0]
        hole2 = detected_holes[1]

        # Determine upper/bottom by Y coordinate (smaller Y = higher in image)
        if hole1.bbox.center.y < hole2.bbox.center.y:
            upper_hole = hole1
            bottom_hole = hole2
        else:
            upper_hole = hole2
            bottom_hole = hole1

        # Assign shark/sawfish by X position relative to each other
        if upper_hole.bbox.center.x > bottom_hole.bbox.center.x:
            upper_child_frame = "torpedo_hole_sawfish_link"
            bottom_child_frame = "torpedo_hole_shark_link"
        else:
            upper_child_frame = "torpedo_hole_shark_link"
            bottom_child_frame = "torpedo_hole_sawfish_link"

        rospy.loginfo_once(
            f"Upper hole assigned to {upper_child_frame}, bottom to {bottom_child_frame}"
        )

        self._publish_hole_transform(upper_hole, upper_child_frame, stamp)
        self._publish_hole_transform(bottom_hole, bottom_child_frame, stamp)

    def _publish_hole_transform(self, detection, child_frame_id, stamp):
        """Estimate distance and publish transform for a single torpedo hole."""
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
