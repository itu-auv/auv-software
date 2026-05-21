#!/usr/bin/env python3

import math
import rospy
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from ultralytics_ros.msg import YoloResult
import tf2_ros
import tf2_geometry_msgs
from utils.detection_utils import (
    check_inside_image,
    calculate_angles_and_offsets,
    calculate_intersection_with_plane,
)


class KdeCameraHandler:
    """Camera handler that publishes all detections as PointStamped messages
    for downstream KDE-based volumetric mapping.

    Unlike the standard front_camera_handler, this handler:
    - Does NOT publish to object_transform_updates
    - Does NOT skip any object classes (all are processed)
    - Publishes odom-frame positions as PointStamped on per-class topics
    """

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
        self.shared_state = shared_state

        # Create per-class PointStamped publishers
        self.kde_publishers = {}
        for _detection_id, link_name in id_tf_map.items():
            if link_name not in self.kde_publishers:
                topic = f"kde_map/points/{link_name}"
                self.kde_publishers[link_name] = rospy.Publisher(
                    topic, PointStamped, queue_size=10
                )
                rospy.loginfo(f"KDE handler: publishing {link_name} on {topic}")

    def handle(self, detection_msg: YoloResult):
        stamp = detection_msg.header.stamp

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id not in self.id_tf_map:
                continue

            prop_name = self.id_tf_map[detection_id]

            # bin_whole uses altitude projection (special path)
            if prop_name == "bin_whole_link":
                self._process_altitude_projection(detection, stamp)
                continue

            if not check_inside_image(detection, self.image_width, self.image_height):
                continue

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

            _angles, offset_x, offset_y = calculate_angles_and_offsets(
                self.calibration, detection.bbox.center, distance
            )

            self._transform_and_publish_point(
                prop_name, offset_x, offset_y, distance, stamp
            )

    def _transform_and_publish_point(
        self, link_name, offset_x, offset_y, distance, stamp
    ):
        """Transform a camera-frame detection to odom and publish as PointStamped."""
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = stamp
            pose_stamped.header.frame_id = self.camera_frame
            pose_stamped.pose.position.x = offset_x
            pose_stamped.pose.position.y = offset_y
            pose_stamped.pose.position.z = distance
            pose_stamped.pose.orientation.w = 1.0

            transformed = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(4.0)
            )

            point_msg = PointStamped()
            point_msg.header.stamp = stamp
            point_msg.header.frame_id = "odom"
            point_msg.point = transformed.pose.position

            if link_name in self.kde_publishers:
                self.kde_publishers[link_name].publish(point_msg)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"KDE transform error for {link_name}: {e}")

    def _process_altitude_projection(self, detection, stamp):
        """Project bin_whole onto the pool floor using ray-plane intersection."""
        altitude = self.shared_state.get("altitude")
        pool_depth = self.shared_state.get("pool_depth")
        if altitude is None:
            return

        detection_id = detection.results[0].id
        link_name = self.id_tf_map[detection_id]

        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5

        angles = self.calibration.calculate_angles((bbox_bottom_x, bbox_bottom_y))

        far_distance = 500.0
        offset_x = math.tan(angles[0]) * far_distance
        offset_y = math.tan(angles[1]) * far_distance

        point1 = PointStamped()
        point1.header.frame_id = self.camera_frame
        point1.point.x = 0
        point1.point.y = 0
        point1.point.z = 0

        point2 = PointStamped()
        point2.header.frame_id = self.camera_frame
        point2.point.x = offset_x
        point2.point.y = offset_y
        point2.point.z = far_distance

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", self.camera_frame, stamp, rospy.Duration(1.0)
            )
            point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)
            point2_odom = tf2_geometry_msgs.do_transform_point(point2, transform)

            intersection = calculate_intersection_with_plane(
                point1_odom, point2_odom, z_plane=-pool_depth
            )
            if intersection:
                x, y, z = intersection
                point_msg = PointStamped()
                point_msg.header.stamp = stamp
                point_msg.header.frame_id = "odom"
                point_msg.point.x = x
                point_msg.point.y = y
                point_msg.point.z = z

                if link_name in self.kde_publishers:
                    self.kde_publishers[link_name].publish(point_msg)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"KDE altitude projection error: {e}")


def create_handler(
    camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
):
    return KdeCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
