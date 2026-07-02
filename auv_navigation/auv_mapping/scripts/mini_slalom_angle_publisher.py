#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rospy
import tf2_geometry_msgs  # noqa: F401 - registers geometry_msgs transforms
import tf2_ros
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    PoseStamped,
    Quaternion,
    TransformStamped,
    Vector3,
    Vector3Stamped,
)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray


SLALOM_RED_ID = 2
SLALOM_WHITE_ID = 3
SLALOM_RED_FRAME = "slalom_red_pipe_link"


class MiniSlalomAnglePublisher:
    def __init__(self):
        rospy.init_node("mini_slalom_angle_publisher")

        self.base_link_frame = rospy.get_param(
            "~base_link_frame", "taluy_mini/base_link"
        )
        self.slalom_camera_frame = rospy.get_param(
            "~slalom_camera_frame",
            self.base_link_frame + "/front_camera_optical_link_stabilized",
        )
        self.yolo_result_topic = rospy.get_param(
            "~yolo_result_topic", "/yolo_result_slalom"
        )
        self.image_topic = rospy.get_param(
            "~image_topic", "/taluy_mini/cameras/cam_front/image_corrected"
        )
        self.slalom_real_height = rospy.get_param("~slalom_real_height", 0.9)
        self.slalom_real_width = rospy.get_param("~slalom_real_width", 0.0254)
        self.max_red_frame_distance = rospy.get_param("~max_red_frame_distance", 30.0)

        self.cam = CameraCalibrationFetcher("cameras/cam_front").get_camera_info()
        self.pipe_angle_full_height_ratio = rospy.get_param(
            "~pipe_angle_full_height_ratio", 0.9
        )
        self.pipe_angle_vertical_edge_margin_px = rospy.get_param(
            "~pipe_angle_vertical_edge_margin_px", 5.0
        )
        self.pipe_angle_debug_jpeg_quality = max(
            1, min(100, int(rospy.get_param("~pipe_angle_debug_jpeg_quality", 80)))
        )

        self.cv_bridge = CvBridge()
        self.latest_image_msg = None
        self.last_pipe_angle_data = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pipe_angle_pub = rospy.Publisher(
            "slalom/pipe_angles", Float32MultiArray, queue_size=1
        )
        self.pipe_angle_debug_pub = rospy.Publisher(
            "slalom/pipe_angles_debug/compressed", CompressedImage, queue_size=1
        )
        self.object_transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )

        rospy.Subscriber(
            self.yolo_result_topic, YoloResult, self.yolo_callback, queue_size=1
        )
        rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )

    def image_callback(self, msg: Image):
        self.latest_image_msg = msg

    def yolo_callback(self, msg: YoloResult):
        detections: Detection2DArray = msg.detections
        if len(detections.detections) == 0:
            return

        self.publish_pipe_angles(detections, msg.header.stamp)

    def publish_pipe_angles(self, detections: Detection2DArray, stamp):
        angle_detections = self.collect_angle_detections(detections, stamp)
        red_detections = [x for x in angle_detections if x["id"] == SLALOM_RED_ID]
        white_detections = [x for x in angle_detections if x["id"] == SLALOM_WHITE_ID]

        if not red_detections:
            left_white, right_white = self.select_outer_white_detections(
                white_detections
            )
            self.publish_pipe_angles_debug(
                angle_detections,
                self.build_pipe_angle_debug_points(
                    None, left_white, right_white, stamp
                ),
                stamp,
            )
            self.publish_pipe_angle_data(
                [
                    None,
                    left_white["angle"] if left_white is not None else None,
                    right_white["angle"] if right_white is not None else None,
                    None,
                    left_white["height"] if left_white is not None else None,
                    right_white["height"] if right_white is not None else None,
                ]
            )
            return

        red_debug = self.build_red_debug_detection(red_detections)
        self.publish_red_frames(red_detections, stamp)
        left_white = self.select_side_white_detection(
            [x for x in white_detections if x["center_x"] < red_debug["center_x"]],
            side="left",
        )
        right_white = self.select_side_white_detection(
            [x for x in white_detections if x["center_x"] > red_debug["center_x"]],
            side="right",
        )

        self.publish_pipe_angles_debug(
            angle_detections,
            self.build_pipe_angle_debug_points(
                red_debug, left_white, right_white, stamp
            ),
            stamp,
        )
        self.publish_pipe_angle_data(
            [
                red_debug["angle"],
                left_white["angle"] if left_white is not None else None,
                right_white["angle"] if right_white is not None else None,
                red_debug["height"],
                left_white["height"] if left_white is not None else None,
                right_white["height"] if right_white is not None else None,
            ]
        )

    def collect_angle_detections(self, detections: Detection2DArray, stamp):
        angle_detections = []
        for detection in detections.detections:
            if len(detection.results) == 0:
                continue

            detection_id = detection.results[0].id
            if detection_id not in [SLALOM_RED_ID, SLALOM_WHITE_ID]:
                continue

            bbox = detection.bbox
            if bbox.size_x <= 0 or bbox.size_y <= 0:
                continue

            camera_angle_x = self.pixel_horizontal_angle(bbox.center.x)
            camera_angle_y = self.pixel_vertical_angle(bbox.center.y)
            angle = self.bbox_angle_relative_base(bbox.center.x, bbox.center.y, stamp)
            if angle is None:
                continue

            angle_detections.append(
                {
                    "id": detection_id,
                    "width": bbox.size_x,
                    "height": bbox.size_y,
                    "center_x": bbox.center.x,
                    "center_y": bbox.center.y,
                    "left": bbox.center.x - bbox.size_x * 0.5,
                    "right": bbox.center.x + bbox.size_x * 0.5,
                    "top": bbox.center.y - bbox.size_y * 0.5,
                    "bottom": bbox.center.y + bbox.size_y * 0.5,
                    "angle": angle,
                    "camera_angle_x": camera_angle_x,
                    "camera_angle_y": camera_angle_y,
                }
            )
        return angle_detections

    def build_red_debug_detection(self, red_detections):
        selected_red = self.select_red_angle_detections(red_detections)
        return {
            "id": SLALOM_RED_ID,
            "width": self.mean([x["width"] for x in selected_red]),
            "height": self.mean([x["height"] for x in selected_red]),
            "center_x": self.mean([x["center_x"] for x in selected_red]),
            "center_y": self.mean([x["center_y"] for x in selected_red]),
            "left": min(x["left"] for x in selected_red),
            "right": max(x["right"] for x in selected_red),
            "top": min(x["top"] for x in selected_red),
            "bottom": max(x["bottom"] for x in selected_red),
            "angle": self.average_angles([x["angle"] for x in selected_red]),
            "camera_angle_x": self.average_angles(
                [x["camera_angle_x"] for x in selected_red]
            ),
            "camera_angle_y": self.average_angles(
                [x["camera_angle_y"] for x in selected_red]
            ),
        }

    def publish_pipe_angle_data(self, pipe_angle_data):
        if self.last_pipe_angle_data is not None:
            pipe_angle_data = [
                self.last_pipe_angle_data[i] if value is None else value
                for i, value in enumerate(pipe_angle_data)
            ]

        if any(value is None for value in pipe_angle_data):
            return

        msg = Float32MultiArray()
        msg.data = pipe_angle_data
        self.last_pipe_angle_data = pipe_angle_data
        self.pipe_angle_pub.publish(msg)

    def publish_red_frames(self, red_detections, stamp):
        for red_detection in red_detections:
            distance = self.estimate_red_distance(red_detection)
            if distance is None:
                continue

            offset_x = math.tan(red_detection["camera_angle_x"]) * distance
            offset_y = math.tan(red_detection["camera_angle_y"]) * distance
            if offset_x**2 + offset_y**2 + distance**2 > self.max_red_frame_distance**2:
                continue

            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.slalom_camera_frame
            transform.child_frame_id = SLALOM_RED_FRAME
            transform.transform.translation = Vector3(offset_x, offset_y, distance)
            transform.transform.rotation = Quaternion(0, 0, 0, 1)

            try:
                pose = PoseStamped()
                pose.header = transform.header
                pose.pose.position = transform.transform.translation
                pose.pose.orientation = transform.transform.rotation

                pose_odom = self.tf_buffer.transform(pose, "odom", rospy.Duration(4.0))
                final_transform = TransformStamped()
                final_transform.header = pose_odom.header
                final_transform.header.stamp = stamp
                final_transform.child_frame_id = SLALOM_RED_FRAME
                final_transform.transform.translation = pose_odom.pose.position
                final_transform.transform.rotation = transform.transform.rotation
                self.object_transform_pub.publish(final_transform)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(5, f"Could not publish red slalom frame: {e}")

    def estimate_red_distance(self, red_detection):
        pixel_length = math.sqrt(
            red_detection["height"] ** 2 + red_detection["width"] ** 2
        )
        if pixel_length <= 0:
            return None
        real_length = math.sqrt(self.slalom_real_height**2 + self.slalom_real_width**2)
        return (self.cam.K[4] * real_length) / pixel_length

    def build_pipe_angle_debug_points(
        self, red_detection, left_white, right_white, stamp
    ):
        points = []
        for label, detection, color in [
            ("white L", left_white, (255, 220, 40)),
            ("red", red_detection, (0, 0, 255)),
            ("white R", right_white, (40, 255, 120)),
        ]:
            if detection is None:
                continue

            points.append(
                {
                    "label": label,
                    "center_x": detection["center_x"],
                    "center_y": detection["center_y"],
                    "angle": detection["angle"],
                    "color": color,
                    "label_at_top": False,
                }
            )

        if red_detection is not None:
            for label, white_detection, color in [
                ("L-red mid", left_white, (255, 0, 255)),
                ("red-R mid", right_white, (0, 255, 255)),
            ]:
                if white_detection is None:
                    continue

                center_x = (
                    red_detection["center_x"] + white_detection["center_x"]
                ) * 0.5
                center_y = (
                    red_detection["center_y"] + white_detection["center_y"]
                ) * 0.5
                angle = self.bbox_angle_relative_base(center_x, center_y, stamp)
                if angle is None:
                    angle = self.average_angles(
                        [red_detection["angle"], white_detection["angle"]]
                    )

                points.append(
                    {
                        "label": label,
                        "center_x": center_x,
                        "center_y": center_y,
                        "angle": angle,
                        "color": color,
                        "label_at_top": True,
                    }
                )
        return points

    def publish_pipe_angles_debug(self, angle_detections, debug_points, stamp):
        if self.pipe_angle_debug_pub.get_num_connections() == 0:
            return

        debug_image, frame_id = self.get_pipe_angle_debug_image()
        self.draw_pipe_angle_context(debug_image, angle_detections)
        self.draw_pipe_angle_points(debug_image, debug_points)

        ok, encoded = cv2.imencode(
            ".jpg",
            debug_image,
            [cv2.IMWRITE_JPEG_QUALITY, self.pipe_angle_debug_jpeg_quality],
        )
        if not ok:
            rospy.logwarn_throttle(5, "Could not encode slalom pipe angle debug image")
            return

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id or self.slalom_camera_frame
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.pipe_angle_debug_pub.publish(msg)

    def get_pipe_angle_debug_image(self):
        if self.latest_image_msg is None:
            return (
                np.zeros((self.cam.height, self.cam.width, 3), dtype=np.uint8),
                self.slalom_camera_frame,
            )

        try:
            image = self.cv_bridge.imgmsg_to_cv2(
                self.latest_image_msg, desired_encoding="bgr8"
            ).copy()
            return image, self.latest_image_msg.header.frame_id
        except Exception as e:
            rospy.logwarn_throttle(5, f"Could not convert slalom debug image: {e}")
            return (
                np.zeros((self.cam.height, self.cam.width, 3), dtype=np.uint8),
                self.slalom_camera_frame,
            )

    def draw_pipe_angle_context(self, image, detections):
        for detection in detections:
            color = (0, 0, 160) if detection["id"] == SLALOM_RED_ID else (230, 230, 230)
            x1, y1 = self.scale_debug_point(image, detection["left"], detection["top"])
            x2, y2 = self.scale_debug_point(
                image, detection["right"], detection["bottom"]
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    def draw_pipe_angle_points(self, image, debug_points):
        top_labels = []
        for index, detection in enumerate(debug_points):
            color = detection["color"]
            center = self.scale_debug_point(
                image, detection["center_x"], detection["center_y"]
            )

            cv2.circle(image, center, 8, (0, 0, 0), -1)
            cv2.circle(image, center, 6, color, -1)
            cv2.circle(image, center, 8, (255, 255, 255), 1, cv2.LINE_AA)

            label = f"{detection['label']} {math.degrees(detection['angle']):+.1f} deg"
            if detection.get("label_at_top", False):
                top_labels.append(label)
            else:
                self.draw_debug_label(
                    image,
                    label,
                    (center[0] + 8, center[1] - 8 + index * 14),
                )

        for index, label in enumerate(top_labels):
            self.draw_debug_label(image, label, (12, 24 + index * 24))

    def select_outer_white_detections(self, white_detections):
        candidates = [x for x in white_detections if self.is_vertically_inside_image(x)]
        if not candidates:
            return None, None

        left_white = min(candidates, key=lambda x: x["center_x"])
        right_white = max(candidates, key=lambda x: x["center_x"])
        return left_white, right_white

    def select_red_angle_detections(self, red_detections):
        max_height = max(x["height"] for x in red_detections)
        full_height = self.pipe_angle_full_height_ratio * self.cam.height

        if max_height >= full_height:
            selected = [x for x in red_detections if x["height"] >= full_height]
            if selected:
                return selected

        return [max(red_detections, key=lambda x: x["height"])]

    def select_side_white_detection(self, white_detections, side: str):
        if not white_detections:
            return None

        candidates = [x for x in white_detections if self.is_vertically_inside_image(x)]
        if not candidates:
            return None

        if side == "right":
            return max(candidates, key=lambda x: x["center_x"])
        return min(candidates, key=lambda x: x["center_x"])

    def is_vertically_inside_image(self, detection):
        margin = self.pipe_angle_vertical_edge_margin_px
        return (
            detection["top"] > margin and detection["bottom"] < self.cam.height - margin
        )

    def bbox_angle_relative_base(self, u: float, v: float, stamp):
        fx = self.cam.K[0]
        fy = self.cam.K[4]
        cx = self.cam.K[2]
        cy = self.cam.K[5]

        ray = Vector3Stamped()
        ray.header.frame_id = self.slalom_camera_frame
        ray.header.stamp = stamp if stamp != rospy.Time(0) else rospy.Time(0)
        ray.vector.x = (u - cx) / fx
        ray.vector.y = (v - cy) / fy
        ray.vector.z = 1.0

        try:
            if not self.tf_buffer.can_transform(
                self.base_link_frame,
                self.slalom_camera_frame,
                ray.header.stamp,
                rospy.Duration(0.05),
            ):
                rospy.logwarn_throttle(
                    5,
                    "No transform from %s to %s for slalom pipe angle",
                    self.slalom_camera_frame,
                    self.base_link_frame,
                )
                return None

            ray_in_base = self.tf_buffer.transform(
                ray, self.base_link_frame, rospy.Duration(1.0)
            )
            return math.atan2(ray_in_base.vector.y, ray_in_base.vector.x)
        except Exception as e:
            rospy.logwarn_throttle(5, f"slalom pipe angle transform error: {e}")
            return None

    def pixel_horizontal_angle(self, u: float):
        return math.atan((u - self.cam.K[2]) / self.cam.K[0])

    def pixel_vertical_angle(self, v: float):
        return math.atan((v - self.cam.K[5]) / self.cam.K[4])

    def scale_debug_point(self, image, x, y):
        height, width = image.shape[:2]
        scale_x = width / float(self.cam.width)
        scale_y = height / float(self.cam.height)
        px = int(round(x * scale_x))
        py = int(round(y * scale_y))
        return (
            max(0, min(width - 1, px)),
            max(0, min(height - 1, py)),
        )

    @staticmethod
    def draw_debug_label(image, text, origin):
        color = (255, 255, 255)
        x = max(4, min(image.shape[1] - 4, origin[0]))
        y = max(18, min(image.shape[0] - 6, origin[1]))
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def average_angles(angles):
        return math.atan2(
            sum(math.sin(angle) for angle in angles),
            sum(math.cos(angle) for angle in angles),
        )

    @staticmethod
    def mean(values):
        return sum(values) / float(len(values))

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = MiniSlalomAnglePublisher()
    node.spin()
