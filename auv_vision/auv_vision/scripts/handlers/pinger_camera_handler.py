#!/usr/bin/env python3

"""Front-camera pinger/tetra handler with DVL floor projection for tetra."""

import math

import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped, Quaternion, TransformStamped, Vector3

from auv_msgs.msg import PropsYaw
from utils.detection_utils import (
    calculate_angles_and_offsets,
    calculate_intersection_with_plane,
    check_inside_image,
    transform_to_odom_and_publish,
)


class PingerCameraHandler:
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
        self.active_ids = list(id_tf_map.keys())

    def set_active_ids(self, ids):
        self.active_ids = ids

    def handle(self, detection_msg):
        stamp = detection_msg.header.stamp
        for detection in detection_msg.detections.detections:
            #print("annemi cok seviyorum")
            if not detection.results:
                continue
            detection_id = detection.results[0].id
            if (
                detection_id not in self.active_ids
                or detection_id not in self.id_tf_map
            ):
                continue

            child_frame = self.id_tf_map[detection_id]
            if child_frame == "front_tetra":
                self._publish_front_tetra(detection, stamp)
            else:
                self._publish_sized_object(detection, child_frame, stamp)

    def _publish_front_tetra(self, detection, stamp):
        if not check_inside_image(detection, self.image_width, self.image_height):
            return

        altitude = self.shared_state.get("altitude")
        if altitude is None:
            rospy.logwarn_throttle(
                5.0, "No altitude data for front camera tetra detection"
            )
            return

        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5
        angles = self.calibration.calculate_angles((bbox_bottom_x, bbox_bottom_y))

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
        transform.child_frame_id = "front_tetra"
        transform.transform.translation = Vector3(x, y, z)
        transform.transform.rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.object_transform_pub.publish(transform)

    def _publish_sized_object(self, detection, child_frame, stamp):
        if not check_inside_image(detection, self.image_width, self.image_height):
            return
        prop = self.props.get(child_frame)
        if prop is None:
            return
        distance = prop.estimate_distance(
            detection.bbox.size_y, detection.bbox.size_x, self.calibration
        )
        if distance is None:
            return

        angles, offset_x, offset_y = calculate_angles_and_offsets(
            self.calibration, detection.bbox.center, distance
        )
        props_yaw = PropsYaw()
        props_yaw.header.stamp = stamp
        props_yaw.object = prop.name
        props_yaw.angle = -angles[0]
        self.props_yaw_pub.publish(props_yaw)
        if (offset_x**2 + offset_y**2 + distance**2) > 30**2:
            return
        transform_to_odom_and_publish(
            self.camera_frame,
            child_frame,
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
    return PingerCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
