#!/usr/bin/env python3

import rospy
import math
import itertools
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as transformations
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
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
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from dataclasses import dataclass, field
from typing import List


@dataclass
class Point:
    x: float
    y: float


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

        self.base_link_frame = rospy.get_param("~namespace", "taluy/base_link")

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
                    t.transform.translation.y = 1.0
                    q = transformations.quaternion_from_euler(0, 0, math.radians(30))
                    t.transform.rotation = Quaternion(*q)
                elif c == "right":
                    t.transform.translation.y = -1.0
                    q = transformations.quaternion_from_euler(0, 0, math.radians(-30))
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
                t.transform.translation.x = pos_wp[0] + v_forward[0] * 0.3
                t.transform.translation.y = pos_wp[1] + v_forward[1] * 0.3
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
            # TODO: check_inside_image
            self.i += 1
            bbox = x.bbox
            off_x, off_y, off_z = self.world_pos_from_height(
                0.9, bbox.size_y, bbox.center.x, bbox.center.y
            )
            # Too far
            if off_z > 10:
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
                self.points.append([wx, wy])

            except Exception as e:
                rospy.logwarn_throttle(5, f"transformation error: {e}")

    def filter_points(self):
        if not self.points:
            self.tfs = []
            return

        pts = np.array(self.points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0

        # TODO: hardcoded
        padding = 50
        target_w = 640 - 2 * padding
        target_h = 480 - 2 * padding

        scale = min(target_w / width, target_h / height)

        img = np.zeros((480, 640), dtype=np.uint8)

        for center in pts:
            u = int((center[0] - x_min) * scale + (640 - width * scale) / 2)
            v = int((center[1] - y_min) * scale + (480 - height * scale) / 2)

            u = max(0, min(639, u))
            v = max(0, min(479, v))

            cv2.circle(img, (u, v), 3, 255, -1)

        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        _, binary = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)

        binary = binary.astype(np.float32) / 255.0
        # TODO: hardcoded
        heatmap = cv2.GaussianBlur(binary, (0, 0), sigmaX=15, sigmaY=15)

        heatmap_copy = heatmap.copy()
        X = 9
        pixel_centers = []

        for i in range(X):
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap_copy)
            if maxVal < 0.01:
                break
            pixel_centers.append(maxLoc)
            # TODO: hardcoded
            suppression_radius = int(25 * 2)
            cv2.circle(heatmap_copy, maxLoc, suppression_radius, 0, -1)

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
            if error_dif > 10:
                return None
            error = sum(errors) / len(errors)
            return error

        pixel_slalom = Slalom()
        # TODO: this shouldn't be the final approach
        all_triplets = list(itertools.combinations(pixel_centers, 3))
        for tri in all_triplets:
            err = get_line_error(tri)
            if err and abs(err - 90) < 20:
                a = np.array(list(tri)).reshape(-1, 1, 2)
                vx, vy, x0, y0 = cv2.fitLine(a, cv2.DIST_L2, 0, 0.01, 0.01)
                m_x, m_y = sorted(
                    tri, key=lambda x: np.linalg.norm(np.array([x0[0], y0[0]]) - x)
                )[0]
                l_x, l_y = sorted(tri, key=lambda x: -x[1])[0]
                r_x, r_y = sorted(tri, key=lambda x: x[1])[0]
                pixel_slalom.groups.append(
                    SlalomGroup(
                        left=Point(l_x, l_y),
                        right=Point(r_x, r_y),
                        mid=Point(m_x, m_y),
                    )
                )

        def pixel_to_world(p):
            u, v = p.x, p.y
            return Point(
                ((u - (640 - width * scale) / 2) / scale) + x_min,
                ((v - (480 - height * scale) / 2) / scale) + y_min,
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
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_left_{i}"
            t.transform.translation.x = g.left.x
            t.transform.translation.y = g.left.y
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.tfs.append(t)
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_right_{i}"
            t.transform.translation.x = g.right.x
            t.transform.translation.y = g.right.y
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.tfs.append(t)
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_mid_{i}"
            t.transform.translation.x = g.mid.x
            t.transform.translation.y = g.mid.y
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.tfs.append(t)

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
        rospy.spin()


if __name__ == "__main__":
    node = SlalomExpFramePublisher()
    node.spin()
