#!/usr/bin/env python3

from collections import defaultdict, deque
import re

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import Image
from ultralytics_ros.msg import YoloResult
import tf2_ros
from tf import transformations as tf_transformations

from utils.detection_utils import calculate_angles_and_offsets
from utils.segment_utils import (
    findposes_circle,
    findposes_rect,
    publish_merged_debug_image,
)


class SegmentCameraHandler:
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
        self.shared_state = shared_state
        self.bridge = CvBridge()
        self.debug_segment_pose = rospy.get_param("~debug_segment_pose", True)
        self.debug_image_topic = rospy.get_param(
            "~segment_pose_debug_topic", "segment_pose_debug"
        )
        self.segment_pose_debug_pub = (
            rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
            if self.debug_segment_pose
            else None
        )
        self.last_yaws = {}

    def _mask_to_cv2(self, mask_msg):
        try:
            return self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to decode segmentation mask: {e}")
            return None

    def _estimate_distance(self, prop, detection, geometry):
        if geometry is not None:
            if geometry.get("diameter_px") is not None:
                measured_diameter = geometry["diameter_px"]
                return prop.estimate_distance(
                    measured_diameter if prop.real_height is not None else None,
                    measured_diameter if prop.real_width is not None else None,
                    self.calibration,
                )

            longest_edge, shortest_edge = geometry.get("edges_px")
            if longest_edge is None or shortest_edge is None:
                return None

            # longest  -> height
            # shortest -> width
            return prop.estimate_distance(longest_edge, shortest_edge, self.calibration)
        # just in case geometry fails //No need actually
        return prop.estimate_distance(
            detection.bbox.size_y,
            detection.bbox.size_x,
            self.calibration,
        )

    def _extract_mask_id(self, mask_msg):
        frame_id = (mask_msg.header.frame_id or "").strip()
        if not frame_id:
            return None

        if frame_id.isdigit() or (frame_id.startswith("-") and frame_id[1:].isdigit()):
            return int(frame_id)

        match = re.search(r"-?\d+", frame_id)
        if match is None:
            return None
        return int(match.group(0))

    def _build_rotation(self, stamp, geometry, prop_name):
        if geometry is not None:
            if prop_name == "nutbolt_link" or prop_name == "pill_link":
                return Quaternion(0, 0, 0, 1)

            try:
                transform = self.tf_buffer.lookup_transform(
                    "odom",
                    "taluy/base_link",
                    stamp,
                    rospy.Duration(0.5),
                )
                _, _, robot_yaw = tf_transformations.euler_from_quaternion(
                    [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ]
                )
                object_angle_odom = robot_yaw + geometry["yaw"]
                quat = tf_transformations.quaternion_from_euler(0, 0, object_angle_odom)
                return Quaternion(*quat)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                return Quaternion(0, 0, 0, 1)

        return Quaternion(0, 0, 0, 1)

    def handle(self, detection_msg: YoloResult):
        masks_msgs = list(detection_msg.masks) if detection_msg.masks else []
        stamp = detection_msg.header.stamp
        masks_by_id = defaultdict(deque)

        # Prefer explicit mask IDs from mask headers.
        for mask_msg in masks_msgs:
            mask_id = self._extract_mask_id(mask_msg)
            if mask_id is None:
                continue
            masks_by_id[mask_id].append(mask_msg)

        # Fallback: some publishers leave mask header.frame_id empty.
        # In that case, retain ID-based access by assigning mask to paired detection ID.
        for detection, mask_msg in zip(detection_msg.detections.detections, masks_msgs):
            if len(detection.results) == 0:
                continue
            if self._extract_mask_id(mask_msg) is None:
                masks_by_id[detection.results[0].id].append(mask_msg)

        debug_items_by_id = {}

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            try:
                if detection_id not in self.id_tf_map:
                    continue

                prop_name = self.id_tf_map[detection_id]
                if prop_name not in self.props:
                    continue

                prop = self.props[prop_name]
                mask_msg = (
                    masks_by_id[detection_id].popleft()
                    if masks_by_id.get(detection_id)
                    else None
                )
                geometry = None

                if mask_msg is not None:
                    mask = self._mask_to_cv2(mask_msg)
                    if mask is not None:
                        last_yaw = self.last_yaws.get(detection_id)
                        if prop_name == "electric_link" or prop_name == "bandaid_link":
                            geometry = findposes_rect(
                                mask, last_yaw=last_yaw, debug=self.debug_segment_pose
                            )
                        elif prop_name == "nutbolt_link" or prop_name == "pill_link":
                            geometry = findposes_circle(
                                mask, last_yaw=last_yaw, debug=self.debug_segment_pose
                            )

                        if geometry is not None and not geometry["valid"]:
                            geometry = None

                        if geometry is not None and geometry.get("yaw") is not None:
                            self.last_yaws[detection_id] = geometry["yaw"]

                if geometry is not None and self.debug_segment_pose:
                    debug_items_by_id[detection_id] = {
                        "detection_id": detection_id,
                        "prop_name": prop_name,
                        "geometry": geometry,
                        "bbox_center": detection.bbox.center,
                    }

                distance = self._estimate_distance(prop, detection, geometry)
                if distance is None:
                    distance = self.shared_state.get("altitude")
                    if distance is None:
                        rospy.logwarn_throttle(
                            5,
                            "No distance data for segment detection (no geometry, no altitude)",
                        )
                        continue

                if geometry is not None:
                    center_x, center_y = geometry["center"]
                    center = Point(x=center_x, y=center_y, z=0.0)
                else:
                    center = detection.bbox.center

                angles, offset_x, offset_y = calculate_angles_and_offsets(
                    self.calibration, center, distance
                )

                transform_stamped_msg = TransformStamped()
                transform_stamped_msg.header.stamp = stamp
                transform_stamped_msg.header.frame_id = self.camera_frame
                transform_stamped_msg.child_frame_id = prop_name
                transform_stamped_msg.transform.translation = Vector3(
                    offset_x, offset_y, distance
                )
                transform_stamped_msg.transform.rotation = self._build_rotation(
                    stamp, geometry, prop_name
                )

                try:
                    pose_stamped = PoseStamped()
                    pose_stamped.header = transform_stamped_msg.header
                    pose_stamped.pose.position = (
                        transform_stamped_msg.transform.translation
                    )
                    pose_stamped.pose.orientation = (
                        transform_stamped_msg.transform.rotation
                    )

                    transformed_pose_stamped = self.tf_buffer.transform(
                        pose_stamped, "odom", rospy.Duration(4.0)
                    )

                    final_transform_stamped = TransformStamped()
                    final_transform_stamped.header = transformed_pose_stamped.header
                    final_transform_stamped.header.stamp = stamp
                    final_transform_stamped.child_frame_id = prop_name
                    final_transform_stamped.transform.translation = (
                        transformed_pose_stamped.pose.position
                    )
                    final_transform_stamped.transform.rotation = (
                        transform_stamped_msg.transform.rotation
                    )

                    self.object_transform_pub.publish(final_transform_stamped)
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logerr(f"Transform error for {prop_name}: {e}")
            except Exception as e:
                rospy.logwarn_throttle(
                    2.0,
                    f"Segment detection processing failed for id={detection_id}: {e}",
                )
                continue
        if self.debug_segment_pose and debug_items_by_id:
            publish_merged_debug_image(
                self.segment_pose_debug_pub,
                detection_msg.header,
                [debug_items_by_id[k] for k in sorted(debug_items_by_id.keys())],
                bridge=self.bridge,
            )


def create_handler(
    camera_config, id_tf_map, props, calibration, tf_buffer, publishers, shared_state
):
    return SegmentCameraHandler(
        camera_config,
        id_tf_map,
        props,
        calibration,
        tf_buffer,
        publishers,
        shared_state,
    )
