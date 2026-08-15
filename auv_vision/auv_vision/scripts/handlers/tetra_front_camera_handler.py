#!/usr/bin/env python3

"""Project front-camera tetra bboxes onto the DVL-derived floor plane."""

import math

import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped, Quaternion, TransformStamped, Vector3

from utils.detection_utils import (
    calculate_intersection_with_plane,
    check_inside_image,
)


class TetraFrontCameraHandler:
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
        self.image_width = camera_config.get("image_width", 640)
        self.image_height = camera_config.get("image_height", 480)
        self.id_tf_map = id_tf_map
        self.calibration = calibration
        self.tf_buffer = tf_buffer
        self.object_transform_pub = publishers["object_transform"]
        self.shared_state = shared_state

    def handle(self, detection_msg):
        stamp = detection_msg.header.stamp
        for detection in detection_msg.detections.detections:
            if not detection.results:
                continue

            detection_id = detection.results[0].id
            if detection_id not in self.id_tf_map:
                continue
            if not check_inside_image(
                detection, self.image_width, self.image_height
            ):
                continue

            self._publish_floor_projection(
                detection,
                self.id_tf_map[detection_id],
                stamp,
            )

    def _publish_floor_projection(self, detection, child_frame, stamp):
        altitude = self.shared_state.get("altitude")
        if altitude is None:
            rospy.logwarn_throttle(
                5.0, "No altitude data for front camera tetra detection"
            )
            return

        bbox_bottom = (
            detection.bbox.center.x,
            detection.bbox.center.y + detection.bbox.size_y * 0.5,
        )
        angles = self.calibration.calculate_angles(bbox_bottom)

        ray_length = 500.0
        camera_origin = PointStamped()
        camera_origin.header.stamp = stamp
        camera_origin.header.frame_id = self.camera_frame

        camera_ray = PointStamped()
        camera_ray.header = camera_origin.header
        camera_ray.point.x = math.tan(angles[0]) * ray_length
        camera_ray.point.y = math.tan(angles[1]) * ray_length
        camera_ray.point.z = ray_length

        try:
            camera_to_odom = self.tf_buffer.lookup_transform(
                "odom", self.camera_frame, stamp, rospy.Duration(1.0)
            )
            origin_odom = tf2_geometry_msgs.do_transform_point(
                camera_origin, camera_to_odom
            )
            ray_odom = tf2_geometry_msgs.do_transform_point(
                camera_ray, camera_to_odom
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(
                5.0, "Front tetra altitude projection TF error: %s", error
            )
            return

        intersection = calculate_intersection_with_plane(
            origin_odom,
            ray_odom,
            z_plane=origin_odom.point.z - float(altitude),
        )
        if intersection is None:
            rospy.logwarn_throttle(
                5.0, "Front tetra camera ray does not intersect the DVL floor"
            )
            return

        x, y, z = intersection
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = child_frame
        transform.transform.translation = Vector3(x, y, z)
        transform.transform.rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.object_transform_pub.publish(transform)


def create_handler(
    camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
):
    return TetraFrontCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
