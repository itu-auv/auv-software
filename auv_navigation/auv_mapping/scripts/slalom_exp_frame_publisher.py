#!/usr/bin/env python3

import rospy
import math
import itertools
from collections import Counter
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as transformations
from dynamic_reconfigure.server import Server
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from auv_mapping.cfg import SlalomExpFrameConfig
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import (
    PointStamped,
    PoseArray,
    PoseStamped,
    Pose,
    TransformStamped,
    Transform,
    Vector3,
    Quaternion,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from dataclasses import dataclass, field
from typing import List, Optional


# copy paste from auv_vision.utils
def check_inside_image(
    detection, image_width: int = 640, image_height: int = 480
) -> bool:
    """Check if a detection bounding box is fully inside the image."""
    center = detection.bbox.center
    half_size_x = detection.bbox.size_x * 0.5
    half_size_y = detection.bbox.size_y * 0.5
    deadzone = 5  # pixels
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
    pipe_id: Optional[int] = None


@dataclass
class SlalomGroup:
    left: Point
    right: Point
    mid: Point


@dataclass
class Slalom:
    groups: List[SlalomGroup] = field(default_factory=list)


class SlalomExpFramePublisher:
    def __init__(self):
        rospy.init_node("slalom_exp_frame_publisher")

        self.frequency = rospy.get_param("~frequency", 10.0)
        self.rate = rospy.Rate(self.frequency)

        self.base_link_frame = rospy.get_param("~base_link_frame", "taluy/base_link")

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
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        self.tfs = None
        self.points = []
        self.collecting = False
        self.i = 0
        self.heatmap_vis = None
        self.cv_bridge = CvBridge()
        self.heatmap_pub = rospy.Publisher("slalom/heatmap_vis", Image, queue_size=1)

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

    @staticmethod
    def pipe_id_to_label(pipe_id: Optional[int]) -> str:
        if pipe_id == 2:
            return "red"
        if pipe_id == 3:
            return "white"
        return "unknown"

    def reconfigure_callback(self, config, level):
        self.search_lateral_offset_m = config.search_lateral_offset_m
        self.search_yaw_deg = config.search_yaw_deg
        self.pipe_height_m = config.pipe_height_m
        self.max_detection_distance_m = config.max_detection_distance_m
        self.waypoint_forward_offset_m = config.waypoint_forward_offset_m

        self.image_width_px = max(2, int(config.image_width_px))
        self.image_height_px = max(2, int(config.image_height_px))
        self.image_padding_px = max(0, int(config.image_padding_px))
        self.opening_kernel_size_px = max(1, int(config.opening_kernel_size_px))
        self.heatmap_sigma_px = max(0.1, float(config.heatmap_sigma_px))
        self.max_peak_count = max(1, int(config.max_peak_count))
        self.peak_min_value = max(0.0, float(config.peak_min_value))
        self.suppression_radius_px = max(1, int(config.suppression_radius_px))
        self.triplet_angle_spread_max_deg = max(
            0.0, float(config.triplet_angle_spread_max_deg)
        )
        self.triplet_target_angle_deg = float(config.triplet_target_angle_deg)
        self.triplet_target_tolerance_deg = max(
            0.0, float(config.triplet_target_tolerance_deg)
        )
        return config

    def publish_search_points_callback(self, req):
        try:
            for c in ["start", "left", "right"]:
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.base_link_frame
                t.child_frame_id = f"slalom_search_{c}"
                t.transform.rotation.w = 1.0

                if c == "start":
                    pass
                elif c == "left":
                    t.transform.translation.y = self.search_lateral_offset_m
                    q = transformations.quaternion_from_euler(
                        0, 0, math.radians(self.search_yaw_deg)
                    )
                    t.transform.rotation = Quaternion(*q)
                elif c == "right":
                    t.transform.translation.y = -self.search_lateral_offset_m
                    q = transformations.quaternion_from_euler(
                        0, 0, math.radians(-self.search_yaw_deg)
                    )
                    t.transform.rotation = Quaternion(*q)

                pose_in_base = PoseStamped()
                pose_in_base.header.frame_id = self.base_link_frame
                pose_in_base.pose.position = t.transform.translation
                pose_in_base.pose.orientation = t.transform.rotation

                pose_in_odom = self.tf_buffer.transform(pose_in_base, "odom")

                t_odom = TransformStamped()
                t_odom.header.stamp = rospy.Time.now()
                t_odom.header.frame_id = "odom"
                t_odom.child_frame_id = t.child_frame_id
                t_odom.transform.translation = Vector3(
                    pose_in_odom.pose.position.x,
                    pose_in_odom.pose.position.y,
                    pose_in_odom.pose.position.z,
                )
                t_odom.transform.rotation = pose_in_odom.pose.orientation

                req_obj = SetObjectTransformRequest()
                req_obj.transform = t_odom
                self.set_object_transform_service.call(req_obj)

            return TriggerResponse(success=True, message="Published all search frames")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def start_point_search_callback(self, req):
        if req.data:
            self.points = []
            self.collecting = True
            self.i = 0
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

        if self.tfs:
            for t in self.tfs:
                req_obj = SetObjectTransformRequest()
                req_obj.transform = t
                self.set_object_transform_service.call(req_obj)

        return SetBoolResponse(
            success=True,
            message=f"Processed and published {len(self.tfs) if self.tfs else 0} centroids",
        )

    def publish_waypoints_callback(self, req):
        if not self.tfs:
            return SetBoolResponse(success=False, message="No centroids available")

        # easier
        groups = {}
        for t in self.tfs:
            parts = t.child_frame_id.split("_")
            idx = int(parts[-1])
            pt_type = parts[-2]
            if idx not in groups:
                groups[idx] = {}
            groups[idx][pt_type] = np.array(
                [t.transform.translation.x, t.transform.translation.y]
            )

        # TODO: what if we couldn't find all 9 pipes??
        for idx, g in groups.items():
            for w in ["left", "right"]:
                pos_wp = (g[w] + g["mid"]) / 2.0
                v_pipe = g[w] - g["mid"]
                v_pipe = v_pipe / np.linalg.norm(v_pipe)
                v_forward = np.array([-v_pipe[1], v_pipe[0]])

                trans = self.tf_buffer.lookup_transform(
                    "odom", self.base_link_frame, rospy.Time(0), rospy.Duration(1.0)
                )
                q_base = [
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ]
                # ????
                matrix_base = transformations.quaternion_matrix(q_base)
                fwd_base = matrix_base[:2, 0]

                if np.dot(v_forward, fwd_base) < 0:
                    v_forward = -v_forward

                yaw = math.atan2(v_forward[1], v_forward[0])
                q = transformations.quaternion_from_euler(0, 0, yaw)

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "odom"
                t.child_frame_id = f"slalom_wp_{w}_{idx}"
                t.transform.translation.x = pos_wp[0]
                t.transform.translation.y = pos_wp[1]
                t.transform.rotation = Quaternion(*q)

                req_obj = SetObjectTransformRequest()
                req_obj.transform = t
                self.set_object_transform_service.call(req_obj)

        return SetBoolResponse(success=True, message="Published waypoints")

    def yolo_callback(self, msg):
        if not self.collecting:
            return

        detections: Detection2DArray = msg.detections
        if len(detections.detections) == 0:
            return

        for x in detections.detections:
            # TODO: hardcoded width-height
            if not check_inside_image(x, 640, 480):
                continue

            if len(x.results) == 0:
                continue
            if not x.results[0].id in [2, 3]:
                continue

            self.i += 1
            bbox = x.bbox

            # slalom pipes might be inclined, if that's the case use diognal length instead of direct height
            pipe_length = bbox.size_y
            detection_ratio = bbox.size_y / bbox.size_x
            if abs(detection_ratio - self.ratio) > self.ratio_threshold:
                pipe_length = math.sqrt(bbox.size_y**2 + bbox.size_x**2)
            off_x, off_y, off_z = self.world_pos_from_height(
                self.slalom_height, pipe_length, bbox.center.x, bbox.center.y
            )
            # Too far
            if off_z > self.max_detection_distance_m:
                continue

            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = detections.header.stamp
            transform_stamped_msg.header.frame_id = (
                self.base_link_frame + "/front_camera_optical_link_stabilized"
            )
            transform_stamped_msg.child_frame_id = f"pipe_{self.i}"
            transform_stamped_msg.transform.translation = Vector3(off_x, off_y, off_z)
            transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)

            try:
                pose_stamped = PoseStamped()
                pose_stamped.header = transform_stamped_msg.header
                pose_stamped.pose.position = transform_stamped_msg.transform.translation
                pose_stamped.pose.orientation = transform_stamped_msg.transform.rotation

                transformed_pose_stamped = self.tf_buffer.transform(
                    pose_stamped, "odom", rospy.Duration(4.0)
                )

                wx = transformed_pose_stamped.pose.position.x
                wy = transformed_pose_stamped.pose.position.y
                self.points.append(Point(wx, wy, pipe_id=int(x.results[0].id)))

            except Exception as e:
                rospy.logwarn_throttle(5, f"transformation error: {e}")

    def filter_points(self):
        self.heatmap_vis = None
        if not self.points:
            self.tfs = []
            return

        pts = np.array([[point.x, point.y] for point in self.points], dtype=np.float32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0

        image_w = max(2, int(self.image_width_px))
        image_h = max(2, int(self.image_height_px))
        padding = max(0, int(self.image_padding_px))
        target_w = max(1, image_w - 2 * padding)
        target_h = max(1, image_h - 2 * padding)

        scale = min(target_w / width, target_h / height)

        img = np.zeros((image_h, image_w), dtype=np.uint8)
        point_pixels = []

        for point in self.points:
            u = int((point.x - x_min) * scale + (image_w - width * scale) / 2)
            v = int((point.y - y_min) * scale + (image_h - height * scale) / 2)

            u = max(0, min(image_w - 1, u))
            v = max(0, min(image_h - 1, v))

            cv2.circle(img, (u, v), 3, 255, -1)
            point_pixels.append((u, v, point.pipe_id))

        kernel_size = max(1, int(self.opening_kernel_size_px))
        opening = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8)
        )
        _, binary = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)

        binary = binary.astype(np.float32) / 255.0
        heatmap = cv2.GaussianBlur(
            binary,
            (0, 0),
            sigmaX=max(0.1, float(self.heatmap_sigma_px)),
            sigmaY=max(0.1, float(self.heatmap_sigma_px)),
        )

        heatmap_copy = heatmap.copy()
        pixel_centers = []

        for _ in range(max(1, int(self.max_peak_count))):
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap_copy)
            if maxVal < self.peak_min_value:
                break
            pixel_centers.append(maxLoc)
            suppression_radius = max(1, int(self.suppression_radius_px))
            cv2.circle(heatmap_copy, maxLoc, suppression_radius, 0, -1)

        center_pipe_ids = {}
        if pixel_centers:
            center_votes = {center: Counter() for center in pixel_centers}
            for u, v, pipe_id in point_pixels:
                nearest_center = min(
                    pixel_centers,
                    key=lambda center: (center[0] - u) ** 2 + (center[1] - v) ** 2,
                )
                center_votes[nearest_center][pipe_id] += 1

            for center, votes in center_votes.items():
                if votes:
                    center_pipe_ids[center] = votes.most_common(1)[0][0]

        heatmap_visa = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        self.heatmap_vis = cv2.applyColorMap(heatmap_visa, cv2.COLORMAP_JET)

        def get_line_error(pts):
            pts = np.array(pts)
            doubles = list(itertools.combinations(pts, 2))
            errors = [
                (
                    math.atan2(abs(pt[0][1] - pt[1][1]), abs(pt[0][0] - pt[1][0]))
                    * 180
                    / math.pi
                )
                for pt in doubles
            ]
            error_dif = max(errors) - min(errors)
            if error_dif > self.triplet_angle_spread_max_deg:
                return None
            error = sum(errors) / len(errors)
            return error

        pixel_slalom = Slalom()
        # TODO: this shouldn't be the final approach
        all_triplets = list(itertools.combinations(pixel_centers, 3))
        for tri in all_triplets:
            err = get_line_error(tri)
            if (
                err
                and abs(err - self.triplet_target_angle_deg)
                < self.triplet_target_tolerance_deg
            ):
                a = np.array(list(tri)).reshape(-1, 1, 2)
                vx, vy, x0, y0 = cv2.fitLine(a, cv2.DIST_L2, 0, 0.01, 0.01)
                m_x, m_y = sorted(
                    tri, key=lambda x: np.linalg.norm(np.array([x0[0], y0[0]]) - x)
                )[0]
                l_x, l_y = sorted(tri, key=lambda x: -x[1])[0]
                r_x, r_y = sorted(tri, key=lambda x: x[1])[0]
                pixel_slalom.groups.append(
                    SlalomGroup(
                        left=Point(l_x, l_y, center_pipe_ids.get((l_x, l_y))),
                        right=Point(r_x, r_y, center_pipe_ids.get((r_x, r_y))),
                        mid=Point(m_x, m_y, center_pipe_ids.get((m_x, m_y))),
                    )
                )

        def pixel_to_world(p):
            u, v = p.x, p.y
            return Point(
                ((u - (image_w - width * scale) / 2) / scale) + x_min,
                ((v - (image_h - height * scale) / 2) / scale) + y_min,
                p.pipe_id,
            )

        world_slalom = Slalom()
        for ps in pixel_slalom.groups:
            world_slalom.groups.append(
                SlalomGroup(
                    left=pixel_to_world(ps.left),
                    right=pixel_to_world(ps.right),
                    mid=pixel_to_world(ps.mid),
                )
            )

        world_slalom.groups.sort(key=lambda g: g.mid.x)

        self.tfs = []
        for i, g in enumerate(world_slalom.groups):
            left_label = self.pipe_id_to_label(g.left.pipe_id)
            right_label = self.pipe_id_to_label(g.right.pipe_id)
            mid_label = self.pipe_id_to_label(g.mid.pipe_id)

            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_{left_label}_left_{i}"
            t.transform.translation.x = g.left.x
            t.transform.translation.y = g.left.y
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.tfs.append(t)
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_{right_label}_right_{i}"
            t.transform.translation.x = g.right.x
            t.transform.translation.y = g.right.y
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.tfs.append(t)
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_{mid_label}_mid_{i}"
            t.transform.translation.x = g.mid.x
            t.transform.translation.y = g.mid.y
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.tfs.append(t)

        self.publish_heatmap()

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
    node = SlalomExpFramePublisher()
    node.spin()
