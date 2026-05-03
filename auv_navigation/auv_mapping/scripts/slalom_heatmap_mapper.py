#!/usr/bin/env python3

import rospy
import math
import time
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as transformations
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from ultralytics_ros.msg import YoloResult
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client
from auv_mapping.cfg import SlalomHeatmapConfig
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import (
    PoseStamped,
    TransformStamped,
    Quaternion,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from dataclasses import dataclass, field
from typing import List


# copy paste from auv_vision.utils
def check_inside_image(
    detection, image_width: int = 640, image_height: int = 480, deadzone: int = 5
) -> bool:
    """Check if a detection bounding box is fully inside the image."""
    center = detection.bbox.center
    half_size_x = detection.bbox.size_x * 0.5
    half_size_y = detection.bbox.size_y * 0.5
    if (
        center.x + half_size_x >= image_width - deadzone
        or center.x - half_size_x <= deadzone
    ):
        return False
    if (
        center.y + half_size_y >= image_height - deadzone
        or center.y - half_size_y <= deadzone
    ):
        return False
    return True


@dataclass
class Point:
    x: float
    y: float


@dataclass
class SlalomRow:
    white_left: Point
    red: Point
    white_right: Point


@dataclass
class SlalomTFGroup:
    object_tfs: List[TransformStamped] = field(default_factory=list)
    waypoint_tfs: List[TransformStamped] = field(default_factory=list)


class SlalomHeatmapMapper:
    def __init__(self):
        rospy.init_node("slalom_heatmap_mapper")

        self.frequency = rospy.get_param("~frequency", 10.0)
        self.rate = rospy.Rate(self.frequency)

        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.search_frame = rospy.get_param("~search_frame", "slalom_search_start")
        self.red_pipe_frame = "slalom_red_pipe_link"
        self.white_pipe_frame = "slalom_white_pipe_link"
        self.search_start_distance = rospy.get_param("~search_start_distance", 2.0)
        self.search_side_offset = rospy.get_param("~search_side_offset", 1.0)
        self.search_side_yaw_offset_deg = rospy.get_param(
            "~search_side_yaw_offset_deg", 30.0
        )

        self.slalom_width = rospy.get_param("~slalom_width", 0.0254)
        self.slalom_height = rospy.get_param("~slalom_height", 0.9)
        self.ratio_threshold = rospy.get_param("~ratio_threshold", 20)
        self.ratio = self.slalom_height / self.slalom_width

        self.cam = CameraCalibrationFetcher("cameras/cam_front").get_camera_info()
        self.yolo_res = rospy.Subscriber(
            "/yolo_result_front", YoloResult, self.yolo_callback
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        self.slalom_rows = []
        self.object_tfs = []
        self.waypoint_tfs = []
        self.points = []
        self.collecting = False
        self.i = 0
        self.heatmap_vis = None
        self.heatmap_build_interval = rospy.get_param("~heatmap_build_interval", 0.2)
        self.last_heatmap_build_time = 0.0
        self.heatmap_color_sample_step = rospy.get_param(
            "~heatmap_color_sample_step", 10
        )
        self.heatmap_image_width_px = rospy.get_param("~heatmap_image_width_px", 640)
        self.heatmap_image_height_px = rospy.get_param("~heatmap_image_height_px", 480)
        self.heatmap_padding_px = rospy.get_param("~heatmap_padding_px", 50)
        self.heatmap_detection_deadzone_px = rospy.get_param(
            "~heatmap_detection_deadzone_px", 5
        )
        self.heatmap_point_radius_px = rospy.get_param("~heatmap_point_radius_px", 3)
        self.heatmap_opening_kernel_px = rospy.get_param(
            "~heatmap_opening_kernel_px", 6
        )
        self.heatmap_gaussian_sigma_px = rospy.get_param(
            "~heatmap_gaussian_sigma_px", 8.0
        )
        self.heatmap_max_centers = rospy.get_param("~heatmap_max_centers", 9)
        self.heatmap_peak_threshold = rospy.get_param("~heatmap_peak_threshold", 0.01)
        self.heatmap_suppression_radius_px = rospy.get_param(
            "~heatmap_suppression_radius_px", 50
        )
        self.slalom_direction = rospy.get_param("~slalom_direction", "left")
        self.cv_bridge = CvBridge()
        self.heatmap_pub = rospy.Publisher("slalom/heatmap_vis", Image, queue_size=1)
        self.reconfigure_server = Server(SlalomHeatmapConfig, self.reconfigure_callback)
        self.smach_params_client = Client(
            "smach_parameters_server",
            timeout=10,
            config_callback=self.smach_params_callback,
        )

        self.srv_publish_search_points = rospy.Service(
            "slalom/publish_search_points", Trigger, self.publish_search_points_callback
        )
        self.srv_start_point_search = rospy.Service(
            "slalom/start_point_search", SetBool, self.start_point_search_callback
        )
        self.srv_stop_point_search = rospy.Service(
            "slalom/stop_point_search", SetBool, self.stop_point_search_callback
        )
        self.srv_publish_waypoints = rospy.Service(
            "slalom/publish_waypoints", SetBool, self.publish_waypoints_callback
        )

    def reconfigure_callback(self, config, level):
        self.heatmap_image_width_px = config.heatmap_image_width_px
        self.heatmap_image_height_px = config.heatmap_image_height_px
        self.heatmap_padding_px = config.heatmap_padding_px
        self.heatmap_detection_deadzone_px = config.heatmap_detection_deadzone_px
        self.heatmap_point_radius_px = config.heatmap_point_radius_px
        self.heatmap_opening_kernel_px = config.heatmap_opening_kernel_px
        self.heatmap_gaussian_sigma_px = config.heatmap_gaussian_sigma_px
        self.heatmap_max_centers = config.heatmap_max_centers
        self.heatmap_peak_threshold = config.heatmap_peak_threshold
        self.heatmap_suppression_radius_px = config.heatmap_suppression_radius_px
        return config

    def publish_search_points_callback(self, req):
        try:
            red_pipes = self.get_all_pipe_positions(self.red_pipe_frame)
            white_pipes = self.get_all_pipe_positions(self.white_pipe_frame)
            if len(red_pipes) == 0 or len(white_pipes) == 0:
                return TriggerResponse(
                    success=False,
                    message=(
                        "Could not find red/white pipe frames "
                        f"(red={len(red_pipes)} white={len(white_pipes)})"
                    ),
                )

            robot_trans = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_link_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            robot_pos = np.array(
                [
                    robot_trans.transform.translation.x,
                    robot_trans.transform.translation.y,
                ]
            )

            closest_red = min(red_pipes, key=lambda p: np.linalg.norm(robot_pos - p))
            closest_white = min(
                white_pipes, key=lambda p: np.linalg.norm(robot_pos - p)
            )

            pipe_line = closest_red - closest_white
            pipe_line_norm = np.linalg.norm(pipe_line)
            if pipe_line_norm <= 1e-6:
                return TriggerResponse(
                    success=False, message="Closest red/white pipe frames overlap"
                )

            pipe_line_unit = pipe_line / pipe_line_norm
            search_forward = np.array([-pipe_line_unit[1], pipe_line_unit[0]])

            q_robot = robot_trans.transform.rotation
            robot_yaw = transformations.euler_from_quaternion(
                [q_robot.x, q_robot.y, q_robot.z, q_robot.w]
            )[2]
            robot_forward = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
            if np.dot(search_forward, robot_forward) < 0:
                search_forward = -search_forward

            target_pos = closest_red - search_forward * self.search_start_distance
            target_yaw = math.atan2(search_forward[1], search_forward[0])
            side_vector = np.array([-math.sin(target_yaw), math.cos(target_yaw)])
            side_angle = math.radians(self.search_side_yaw_offset_deg)
            stamp = rospy.Time.now()

            for name, side_offset, yaw_offset in [
                ("start", 0.0, 0.0),
                ("left", self.search_side_offset, side_angle),
                ("right", -self.search_side_offset, -side_angle),
            ]:
                pos = target_pos + side_vector * side_offset
                q = transformations.quaternion_from_euler(0, 0, target_yaw + yaw_offset)

                t_odom = TransformStamped()
                t_odom.header.stamp = stamp
                t_odom.header.frame_id = self.odom_frame
                t_odom.child_frame_id = f"slalom_search_{name}"
                t_odom.transform.translation.x = pos[0]
                t_odom.transform.translation.y = pos[1]
                t_odom.transform.translation.z = 0.0
                t_odom.transform.rotation = Quaternion(*q)

                req_obj = SetObjectTransformRequest()
                req_obj.transform = t_odom
                self.set_object_transform_service.call(req_obj)

            return TriggerResponse(
                success=True,
                message=(
                    "Published search frames from closest red/white pipes "
                    f"(distance={self.search_start_distance:.2f}m)"
                ),
            )
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def start_point_search_callback(self, req):
        if req.data:
            self.points = []
            self.slalom_rows = []
            self.object_tfs = []
            self.waypoint_tfs = []
            self.collecting = True
            self.i = 0
            self.last_heatmap_build_time = 0.0
            self.heatmap_vis = np.zeros(
                (self.heatmap_image_height_px, self.heatmap_image_width_px, 3),
                dtype=np.uint8,
            )
            self.publish_heatmap()
            rospy.loginfo("Started point collection")
            return SetBoolResponse(success=True, message="Started point search")
        else:
            self.collecting = False
            return SetBoolResponse(
                success=True, message="Stopped point search (collection paused)"
            )

    def stop_point_search_callback(self, req):
        self.collecting = False
        rospy.loginfo(
            f"Stopped point collection. Collected {len(self.points)} points. Filtering..."
        )
        self.filter_points()

        self.publish_transform_group(self.object_tfs)

        return SetBoolResponse(
            success=True,
            message=f"Processed and published {len(self.object_tfs)} slalom objects",
        )

    def publish_waypoints_callback(self, req):
        if self.slalom_rows:
            self.waypoint_tfs = self.build_waypoint_transforms(self.slalom_rows)

        if not self.waypoint_tfs:
            return SetBoolResponse(success=False, message="No waypoints available")

        self.publish_transform_group(self.waypoint_tfs)
        return SetBoolResponse(
            success=True, message=f"Published {len(self.waypoint_tfs)} waypoints"
        )

    def smach_params_callback(self, config):
        if config is None:
            rospy.logwarn("Could not get parameters from smach_parameters_server")
            return
        self.slalom_direction = config.slalom_direction
        if self.slalom_rows:
            self.waypoint_tfs = self.build_waypoint_transforms(self.slalom_rows)
        rospy.loginfo(f"Slalom direction updated to: {self.slalom_direction}")

    def get_all_pipe_positions(self, frame_prefix):
        pipe_positions = []

        if self.tf_buffer.can_transform(
            self.odom_frame, frame_prefix, rospy.Time(0), rospy.Duration(0.05)
        ):
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame, frame_prefix, rospy.Time(0), rospy.Duration(1.0)
            )
            pipe_positions.append(
                np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                    ]
                )
            )

        index = 0
        while self.tf_buffer.can_transform(
            self.odom_frame,
            f"{frame_prefix}_{index}",
            rospy.Time(0),
            rospy.Duration(0.05),
        ):
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                f"{frame_prefix}_{index}",
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            pipe_positions.append(
                np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                    ]
                )
            )
            index += 1

        return pipe_positions

    def yolo_callback(self, msg):
        if not self.collecting:
            return

        detections: Detection2DArray = msg.detections
        if len(detections.detections) == 0:
            return

        for x in detections.detections:
            if not check_inside_image(
                x,
                self.cam.width,
                self.cam.height,
                self.heatmap_detection_deadzone_px,
            ):
                continue

            if len(x.results) == 0:
                continue
            if not x.results[0].id in [2, 3]:
                continue

            self.i += 1
            bbox = x.bbox
            if bbox.size_x <= 0 or bbox.size_y <= 0:
                rospy.logwarn_throttle(
                    5,
                    "Skipping slalom detection with non-positive bbox size: size_x=%s size_y=%s",
                    bbox.size_x,
                    bbox.size_y,
                )
                continue

            # slalom pipes might be inclined, if that's the case use diognal length instead of direct height
            pipe_length = bbox.size_y
            detection_ratio = bbox.size_y / bbox.size_x
            if abs(detection_ratio - self.ratio) > self.ratio_threshold:
                pipe_length = math.sqrt(bbox.size_y**2 + bbox.size_x**2)
            off_x, off_y, off_z = self.world_pos_from_height(
                self.slalom_height, pipe_length, bbox.center.x, bbox.center.y
            )
            # Too far
            if off_z > 10:
                continue

            try:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = detections.header.stamp
                pose_stamped.header.frame_id = (
                    self.base_link_frame + "/front_camera_optical_link_stabilized"
                )
                pose_stamped.pose.position.x = off_x
                pose_stamped.pose.position.y = off_y
                pose_stamped.pose.position.z = off_z
                pose_stamped.pose.orientation = Quaternion(0, 0, 0, 1)

                stamp = pose_stamped.header.stamp
                if stamp == rospy.Time(0):
                    rospy.logwarn_throttle(
                        5,
                        "Detection timestamp is zero; using latest transform to %s",
                        self.search_frame,
                    )

                if not self.tf_buffer.can_transform(
                    self.search_frame,
                    pose_stamped.header.frame_id,
                    stamp,
                    rospy.Duration(0.05),
                ):
                    rospy.logwarn_throttle(
                        5,
                        "No transform from %s to %s at detection stamp %.3f",
                        pose_stamped.header.frame_id,
                        self.search_frame,
                        stamp.to_sec(),
                    )
                    continue

                pose_in_search = self.tf_buffer.transform(
                    pose_stamped, self.search_frame, rospy.Duration(1.0)
                )

                wx = pose_in_search.pose.position.x
                wy = pose_in_search.pose.position.y
                self.points.append([wx, wy, x.results[0].id])
                rospy.loginfo_throttle(
                    2.0,
                    "Collected %d slalom points in %s",
                    len(self.points),
                    self.search_frame,
                )
                self.update_heatmap_vis()

            except Exception as e:
                rospy.logwarn_throttle(5, f"transformation error: {e}")

    def build_heatmap_data(self):
        if not self.points:
            self.heatmap_vis = np.zeros(
                (self.heatmap_image_height_px, self.heatmap_image_width_px, 3),
                dtype=np.uint8,
            )
            return None

        pts_with_ids = np.array(self.points)
        pts = pts_with_ids[:, :2]
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0

        padding = self.heatmap_padding_px
        target_w = max(1, self.heatmap_image_width_px - 2 * padding)
        target_h = max(1, self.heatmap_image_height_px - 2 * padding)

        scale = min(target_w / height, target_h / width)

        img = np.zeros(
            (self.heatmap_image_height_px, self.heatmap_image_width_px),
            dtype=np.uint8,
        )

        pixel_points = []
        for center, det_id in zip(pts, pts_with_ids[:, 2].astype(int)):
            u = int(
                (y_max - center[1]) * scale
                + (self.heatmap_image_width_px - height * scale) / 2
            )
            v = int(
                (x_max - center[0]) * scale
                + (self.heatmap_image_height_px - width * scale) / 2
            )

            u = max(0, min(self.heatmap_image_width_px - 1, u))
            v = max(0, min(self.heatmap_image_height_px - 1, v))

            cv2.circle(img, (u, v), self.heatmap_point_radius_px, 255, -1)
            pixel_points.append((u, v, det_id))

        binary_img = img
        if self.heatmap_opening_kernel_px > 0:
            kernel = np.ones(
                (
                    self.heatmap_opening_kernel_px,
                    self.heatmap_opening_kernel_px,
                ),
                np.uint8,
            )
            binary_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

        binary = binary.astype(np.float32) / 255.0
        heatmap = cv2.GaussianBlur(
            binary,
            (0, 0),
            sigmaX=self.heatmap_gaussian_sigma_px,
            sigmaY=self.heatmap_gaussian_sigma_px,
        )

        return {
            "heatmap": heatmap,
            "pixel_points": pixel_points,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width": width,
            "height": height,
            "scale": scale,
        }

    def update_heatmap_vis(self):
        now = time.monotonic()
        if now - self.last_heatmap_build_time < self.heatmap_build_interval:
            return
        self.last_heatmap_build_time = now

        heatmap_data = self.build_heatmap_data()
        if heatmap_data is None:
            return

        self.heatmap_vis = self.build_heatmap_visualization(
            heatmap_data["heatmap"], heatmap_data["pixel_points"]
        )
        self.publish_heatmap()

    def filter_points(self):
        heatmap_data = self.build_heatmap_data()
        if heatmap_data is None:
            self.slalom_rows = []
            self.object_tfs = []
            self.waypoint_tfs = []
            return

        heatmap = heatmap_data["heatmap"]
        pixel_points = heatmap_data["pixel_points"]
        x_min = heatmap_data["x_min"]
        x_max = heatmap_data["x_max"]
        y_min = heatmap_data["y_min"]
        y_max = heatmap_data["y_max"]
        width = heatmap_data["width"]
        height = heatmap_data["height"]
        scale = heatmap_data["scale"]
        self.heatmap_vis = self.build_heatmap_visualization(heatmap, pixel_points)

        heatmap_copy = heatmap.copy()
        pixel_centers = []

        for _ in range(self.heatmap_max_centers):
            _, maxVal, _, maxLoc = cv2.minMaxLoc(heatmap_copy)
            if maxVal < self.heatmap_peak_threshold:
                break
            pixel_centers.append(maxLoc)
            cv2.circle(heatmap_copy, maxLoc, self.heatmap_suppression_radius_px, 0, -1)

        pixel_center_colors = {}
        if pixel_centers:
            center_counts = [{2: 0, 3: 0} for _ in pixel_centers]
            for u, v, det_id in pixel_points:
                closest_idx = min(
                    range(len(pixel_centers)),
                    key=lambda i: (pixel_centers[i][0] - u) ** 2
                    + (pixel_centers[i][1] - v) ** 2,
                )
                center_counts[closest_idx][det_id] += 1
            for center, counts in zip(pixel_centers, center_counts):
                pixel_center_colors[center] = (
                    "red" if counts[2] >= counts[3] else "white"
                )
            for i, counts in enumerate(center_counts):
                rospy.loginfo(
                    f"Cluster {i}: red={counts[2]}, white={counts[3]}, selected={pixel_center_colors[pixel_centers[i]]}"
                )

        def pixel_to_world(p):
            u, v = p.x, p.y
            return Point(
                ((v - (self.heatmap_image_height_px - width * scale) / 2) / scale) * -1
                + x_max,
                ((u - (self.heatmap_image_width_px - height * scale) / 2) / scale) * -1
                + y_max,
            )

        world_points = [
            pixel_to_world(Point(center[0], center[1])) for center in pixel_centers
        ]
        colors = [pixel_center_colors.get(center, "white") for center in pixel_centers]
        self.slalom_rows = self.build_slalom_rows(world_points, colors)
        tf_group = SlalomTFGroup(
            object_tfs=self.build_object_transforms(self.slalom_rows),
            waypoint_tfs=self.build_waypoint_transforms(self.slalom_rows),
        )
        self.object_tfs = tf_group.object_tfs
        self.waypoint_tfs = tf_group.waypoint_tfs

        self.publish_heatmap()

    def build_slalom_rows(
        self, world_points: List[Point], colors: List[str]
    ) -> List[SlalomRow]:
        reds = []
        whites = []
        for point, color in zip(world_points, colors):
            if color == "red":
                reds.append(point)
            else:
                whites.append(point)

        if not reds or len(whites) < 2:
            rospy.logwarn(
                "Not enough colored slalom detections to build rows. red=%d white=%d",
                len(reds),
                len(whites),
            )
            return []

        whites_sorted_by_y = sorted(whites, key=lambda point: point.y, reverse=True)
        split_index = len(whites_sorted_by_y) // 2
        white_left = sorted(whites_sorted_by_y[:split_index], key=lambda point: point.x)
        white_right = sorted(
            whites_sorted_by_y[split_index:], key=lambda point: point.x
        )
        reds = sorted(reds, key=lambda point: point.x)

        row_count = min(len(reds), len(white_left), len(white_right))
        if row_count == 0:
            rospy.logwarn("No complete semantic slalom rows could be built")
            return []

        if (
            row_count < len(reds)
            or row_count < len(white_left)
            or row_count < len(white_right)
        ):
            rospy.logwarn(
                "Truncating semantic slalom rows to %d entries (red=%d left=%d right=%d)",
                row_count,
                len(reds),
                len(white_left),
                len(white_right),
            )

        return [
            SlalomRow(
                white_left=white_left[i],
                red=reds[i],
                white_right=white_right[i],
            )
            for i in range(row_count)
        ]

    def build_object_transforms(self, rows: List[SlalomRow]) -> List[TransformStamped]:
        transforms = []
        for i, row in enumerate(rows):
            transforms.append(
                self.make_position_transform(f"pipe_white_left_{i}", row.white_left)
            )
            transforms.append(self.make_position_transform(f"pipe_red_{i}", row.red))
            transforms.append(
                self.make_position_transform(f"pipe_white_right_{i}", row.white_right)
            )
        return transforms

    def build_waypoint_transforms(
        self, rows: List[SlalomRow]
    ) -> List[TransformStamped]:
        transforms = []
        side = self.get_selected_waypoint_side()

        for i, row in enumerate(rows):
            side_point = row.white_left if side == "left" else row.white_right
            side_vec = np.array([side_point.x, side_point.y])
            red_vec = np.array([row.red.x, row.red.y])

            pos_wp = (side_vec + red_vec) / 2.0
            v_pipe = side_vec - red_vec
            v_pipe_norm = np.linalg.norm(v_pipe)
            if v_pipe_norm <= 1e-6:
                rospy.logwarn(
                    "Skipping waypoint %d because selected pipe points overlap", i
                )
                continue
            v_pipe = v_pipe / v_pipe_norm
            v_forward = np.array([-v_pipe[1], v_pipe[0]])
            try:
                v_forward = self.align_forward_vector_with_base(v_forward)
            except Exception as e:
                rospy.logwarn(
                    "Failed to align waypoint %d with base_link forward axis: %s", i, e
                )
                continue
            yaw = math.atan2(v_forward[1], v_forward[0])
            q = transformations.quaternion_from_euler(0, 0, yaw)

            try:
                transforms.append(
                    self.make_transform_in_odom(
                        f"slalom_wp_{i}",
                        pos_wp[0],
                        pos_wp[1],
                        Quaternion(*q),
                    )
                )
            except Exception as e:
                rospy.logwarn("Failed to transform waypoint %d to odom: %s", i, e)

        return transforms

    def get_selected_waypoint_side(self):
        if self.slalom_direction in ["left", "right"]:
            return self.slalom_direction

        rospy.logwarn(
            "Unsupported slalom_direction '%s'. Falling back to 'left'",
            self.slalom_direction,
        )
        return "left"

    def align_forward_vector_with_base(self, v_forward):
        trans = self.tf_buffer.lookup_transform(
            self.search_frame,
            self.base_link_frame,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        q_base = [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w,
        ]
        matrix_base = transformations.quaternion_matrix(q_base)
        fwd_base = matrix_base[:2, 0]

        if np.dot(v_forward, fwd_base) < 0:
            v_forward = -v_forward
        return v_forward

    def make_position_transform(self, child_frame_id: str, point: Point):
        return self.make_transform_in_odom(
            child_frame_id,
            point.x,
            point.y,
            Quaternion(0, 0, 0, 1),
        )

    def make_transform_in_odom(self, child_frame_id, x, y, orientation):
        pose_in_search = PoseStamped()
        pose_in_search.header.stamp = rospy.Time(0)
        pose_in_search.header.frame_id = self.search_frame
        pose_in_search.pose.position.x = x
        pose_in_search.pose.position.y = y
        pose_in_search.pose.position.z = 0.0
        pose_in_search.pose.orientation = orientation

        pose_in_odom = self.tf_buffer.transform(
            pose_in_search, self.odom_frame, rospy.Duration(1.0)
        )

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation.x = pose_in_odom.pose.position.x
        t.transform.translation.y = pose_in_odom.pose.position.y
        t.transform.translation.z = pose_in_odom.pose.position.z
        t.transform.rotation = pose_in_odom.pose.orientation
        return t

    def build_heatmap_visualization(self, heatmap, pixel_points):
        heatmap_vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        density_view = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        self.overlay_sampled_detection_rings(density_view, pixel_points)
        self.draw_heatmap_axes_indicator(density_view)
        return density_view

    def get_sampled_pixel_points(self, pixel_points):
        sample_step = max(1, int(self.heatmap_color_sample_step))
        return [
            point
            for index, point in enumerate(pixel_points)
            if index % sample_step == 0
        ]

    def overlay_sampled_detection_points(self, heatmap_vis, pixel_points):
        color_map = {
            2: (0, 0, 255),
            3: (255, 255, 255),
        }

        for u, v, det_id in self.get_sampled_pixel_points(pixel_points):
            color = color_map.get(det_id)
            if color is None:
                continue

            cv2.circle(heatmap_vis, (u, v), 3, color, -1)
            cv2.circle(heatmap_vis, (u, v), 3, (0, 0, 0), 1)

    def overlay_sampled_detection_rings(self, heatmap_vis, pixel_points):
        color_map = {
            2: ((0, 0, 255), (255, 245, 245)),
            3: ((255, 255, 255), (30, 30, 30)),
        }

        for u, v, det_id in self.get_sampled_pixel_points(pixel_points):
            colors = color_map.get(det_id)
            if colors is None:
                continue

            ring_color, center_color = colors
            cv2.circle(heatmap_vis, (u, v), 7, (0, 0, 0), 2)
            cv2.circle(heatmap_vis, (u, v), 6, ring_color, 2)
            cv2.circle(heatmap_vis, (u, v), 2, center_color, -1)

    def draw_heatmap_axes_indicator(self, heatmap_vis):
        origin = (heatmap_vis.shape[1] - 28, 48)
        x_end = (origin[0], origin[1] - 36)
        y_end = (origin[0] - 36, origin[1])
        axis_color = (235, 235, 235)
        outline_color = (15, 15, 15)

        cv2.circle(heatmap_vis, origin, 3, outline_color, -1)
        cv2.arrowedLine(
            heatmap_vis,
            origin,
            x_end,
            outline_color,
            4,
            tipLength=0.25,
        )
        cv2.arrowedLine(
            heatmap_vis,
            origin,
            y_end,
            outline_color,
            4,
            tipLength=0.25,
        )
        cv2.arrowedLine(
            heatmap_vis,
            origin,
            x_end,
            axis_color,
            2,
            tipLength=0.25,
        )
        cv2.arrowedLine(
            heatmap_vis,
            origin,
            y_end,
            axis_color,
            2,
            tipLength=0.25,
        )
        cv2.putText(
            heatmap_vis,
            "x",
            (x_end[0] + 6, x_end[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            outline_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_vis,
            "x",
            (x_end[0] + 6, x_end[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            axis_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_vis,
            "y",
            (y_end[0] - 14, y_end[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            outline_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_vis,
            "y",
            (y_end[0] - 14, y_end[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            axis_color,
            1,
            cv2.LINE_AA,
        )

    def publish_transform_group(self, transforms: List[TransformStamped]):
        for transform in transforms:
            req_obj = SetObjectTransformRequest()
            req_obj.transform = transform
            self.set_object_transform_service.call(req_obj)

    def publish_heatmap(self):
        if self.heatmap_vis is not None:
            img_msg = self.cv_bridge.cv2_to_imgmsg(self.heatmap_vis, "bgr8")
            self.heatmap_pub.publish(img_msg)

    def world_pos_from_height(self, real_height, pixel_height, u, v):
        fx = self.cam.K[0]
        fy = self.cam.K[4]
        cx = self.cam.K[2]
        cy = self.cam.K[5]

        Z = (fy * real_height) / pixel_height
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return X, Y, Z

    def spin(self):
        while not rospy.is_shutdown():
            self.publish_heatmap()
            self.rate.sleep()


if __name__ == "__main__":
    node = SlalomHeatmapMapper()
    node.spin()
