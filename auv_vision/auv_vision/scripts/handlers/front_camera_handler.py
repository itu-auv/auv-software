#!/usr/bin/env python3

import math
import rospy
from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
import tf2_ros
import tf2_geometry_msgs
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    transform_to_odom_and_publish,
    calculate_intersection_with_plane,
)


class FrontCameraHandler:
    def __init__(
        self, camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
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

        # Front camera specific state
        self.active_ids = list(id_tf_map.keys())  # All IDs active by default

    def set_active_ids(self, ids: list):
        """Called by orchestrator when set_front_camera_focus service is triggered."""
        self.active_ids = ids

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id not in self.active_ids:
                continue

            if detection_id not in self.id_tf_map:
                continue

            # bin_whole (ID=6) uses altitude projection — special path
            if detection_id == 6:
                self._process_altitude_projection(detection, stamp)
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

    def _process_altitude_projection(self, detection, stamp):
        """Project bin_whole (ID=6) onto the pool floor using ray-plane intersection."""
        altitude = self.shared_state.get("altitude")
        pool_depth = self.shared_state.get("pool_depth")
        if altitude is None:
            rospy.logwarn("No altitude data available")
            return

        detection_id = detection.results[0].id

        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5

        angles = self.calibration.calculate_angles((bbox_bottom_x, bbox_bottom_y))

        distance = 500.0
        offset_x = math.tan(angles[0]) * distance
        offset_y = math.tan(angles[1]) * distance

        point1 = PointStamped()
        point1.header.frame_id = self.camera_frame
        point1.point.x = 0
        point1.point.y = 0
        point1.point.z = 0

        point2 = PointStamped()
        point2.header.frame_id = self.camera_frame
        point2.point.x = offset_x
        point2.point.y = offset_y
        point2.point.z = distance

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom",
                self.camera_frame,
                stamp,
                rospy.Duration(1.0),
            )
            point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)
            point2_odom = tf2_geometry_msgs.do_transform_point(point2, transform)

            intersection = calculate_intersection_with_plane(
                point1_odom, point2_odom, z_plane=-pool_depth
            )
            if intersection:
                x, y, z = intersection
                transform_stamped_msg = TransformStamped()
                transform_stamped_msg.header.stamp = stamp
                transform_stamped_msg.header.frame_id = "odom"
                transform_stamped_msg.child_frame_id = self.id_tf_map[detection_id]

                transform_stamped_msg.transform.translation = Vector3(x, y, z)
                transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
                self.object_transform_pub.publish(transform_stamped_msg)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Transform error: {e}")


def create_handler(camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state):
    return FrontCameraHandler(
        camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
    )
