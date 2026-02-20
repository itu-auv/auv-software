#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import (
    PoseStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
import tf2_ros
from tf import transformations as tf_transformations
from utils.detection_utils import (
    calculate_angles_and_offsets,
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
        self.props_yaw_pub = publishers["props_yaw"]
        self.shared_state = shared_state

    def handle(self, detection_msg: YoloResult):
        seg = self.shared_state.get("last_segment_measurement")

        # Use segment measurement timestamp if available, otherwise detection stamp
        # ı did this beacuse ı do not want to abandon the bottle frame completly.
        if seg is not None and seg.valid:
            stamp = seg.header.stamp
        else:
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

            # --- Distance estimation ---
            # Prefer segment_measurement thickness_px for distance calculation
            if seg is not None and seg.valid and seg.thickness_px > 0:
                distance = prop.estimate_distance(
                    None,
                    seg.thickness_px,
                    self.calibration,
                )
            else:
                # Fallback: use YOLO bbox dimensions
                distance = prop.estimate_distance(
                    detection.bbox.size_y,
                    detection.bbox.size_x,
                    self.calibration,
                )

            # Final fallback: use altitude
            if distance is None:
                distance = self.shared_state.get("altitude")
                if distance is None:
                    rospy.logwarn_throttle(
                        5,
                        "No distance data for bottle detection (no segment, no altitude)",
                    )
                    continue

            # --- Angle and offset calculation ---
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

            # --- Build transform with rotation from segment measurement ---
            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = stamp
            transform_stamped_msg.header.frame_id = self.camera_frame
            transform_stamped_msg.child_frame_id = prop_name

            transform_stamped_msg.transform.translation = Vector3(
                offset_x, offset_y, distance
            )

            # Calculate bottle orientation from segment angle + robot yaw
            if seg is not None and seg.valid:
                dt = abs((stamp - seg.header.stamp).to_sec())
                if dt < 10000:
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            "odom",
                            "taluy/base_link",
                            seg.header.stamp,
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
                        bottle_angle_odom = robot_yaw + seg.angle
                        quat = tf_transformations.quaternion_from_euler(
                            0, 0, bottle_angle_odom
                        )
                        transform_stamped_msg.transform.rotation = Quaternion(*quat)
                    except (
                        tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException,
                    ):
                        transform_stamped_msg.transform.rotation = Quaternion(
                            0, 0, 0, 1
                        )
                else:
                    rospy.logwarn_throttle(5, "Segment measurement is too old")
                    transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
            else:
                transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)

            # --- Transform to odom and publish ---
            try:
                pose_stamped = PoseStamped()
                pose_stamped.header = transform_stamped_msg.header
                pose_stamped.pose.position = transform_stamped_msg.transform.translation
                pose_stamped.pose.orientation = transform_stamped_msg.transform.rotation

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
                # Keep original rotation (not the transformed one)
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
