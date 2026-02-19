#!/usr/bin/env python3

import rospy
import math
import itertools
import cv2
import numpy as np
import tf2_ros
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray
from camera_detection_pose_estimator import CameraCalibration
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

class SlalomKmeansCluster:
    def __init__(self):
        rospy.init_node("SlalomKmeansCluster")
        self.cam = CameraCalibration("taluy/cameras/cam_front")
        self.yolo_res = rospy.Subscriber("/yolo_result_front", YoloResult, self.yolo_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.tfs = None
        
        self.points = []
        self.centers = None
        self.start_time = None
        self.collecting = True
        self.i = 0

    def yolo_callback(self, msg):
        if not self.collecting:
            return

        detections: Detection2DArray = msg.detections
        if len(detections.detections) == 0:
            return

        if self.start_time is None:
            self.start_time = rospy.Time.now()

        for x in detections.detections:
            # TODO: check_inside_image
            self.i += 1
            bbox = x.bbox
            off_x, off_y, off_z = self.world_pos_from_height(0.9, bbox.size_y, bbox.center.x, bbox.center.y)
            print(f"distance = {off_z}")
            # Too far
            if off_z > 10:
                continue

            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = detections.header.stamp
            transform_stamped_msg.header.frame_id = "taluy/base_link/front_camera_optical_link_stabilized"
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
                rospy.logwarn(f"transformation error: {e}")

        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed_time >= 20.0:
            self.collecting = False
            rospy.loginfo(f"collected {len(self.points)} points. filtering...")
            self.point_filter()

    def point_filter(self):
        pts = np.array(self.points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0: width = 1.0
        if height == 0: height = 1.0

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

        img_copy = img.copy()

        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        _, binary = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)

        binary = binary.astype(np.float32) / 255.0
        heatmap = cv2.GaussianBlur(binary, (0,0), sigmaX=15, sigmaY=15)

        heatmap_copy = heatmap.copy()
        # TODO: we don't have any protection in case we can't properly see all 9 pipes
        X = 9
        pixel_centers = []

        for i in range(X):
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap_copy)
            if maxVal < 0.01:
                break
            pixel_centers.append(maxLoc)
            suppression_radius = int(25*2)
            cv2.circle(heatmap_copy, maxLoc, suppression_radius, 0, -1)

        def get_line_error(pts):
            pts = np.array(pts)

            doubles = list(itertools.combinations(pts, 2))
            errors = [(math.atan2(abs(pt[0][1] - pt[1][1]), abs(pt[0][0] - pt[1][0])) * 180 / math.pi) for pt in doubles]

            error_dif = max(errors) - min(errors)
            if error_dif > 10:
                return None

            error = sum(errors)/len(errors)
            return error

        pixel_slalom = Slalom()

        all_triplets = list(itertools.combinations(pixel_centers, 3))
        valid_triplets = []
        for tri in all_triplets:
            err = get_line_error(tri)
            # TODO: this shouldn't be the final approach
            if err and abs(err-90) < 20:
                a = np.array(list(tri)).reshape(-1, 1, 2)
                vx, vy, x0, y0 = cv2.fitLine(a, cv2.DIST_L2, 0, 0.01, 0.01)
                #mid
                m_x, m_y = sorted(tri, key=lambda x: np.linalg.norm(np.array([x0[0], y0[0]]) - x))[0]
                #left
                l_x, l_y = sorted(tri, key=lambda x: x[1])[0]
                #right
                r_x, r_y = sorted(tri, key=lambda x: -x[1])[0]
                pixel_slalom.groups.append(SlalomGroup(
                    left=Point(l_x, l_y),
                    right=Point(r_x, r_y),
                    mid=Point(m_x, m_y),
                ))

        print(f"pixel_slalom = {pixel_slalom}")

        def pixel_to_world(p):
            u, v = p.x, p.y
            return Point(((u - (640 - width * scale) / 2) / scale) + x_min, ((v - (480 - height * scale) / 2) / scale) + y_min)

        world_slalom = Slalom()
        for ps in pixel_slalom.groups:
            world_slalom.groups.append(SlalomGroup(
                    left=pixel_to_world(ps.left),
                    right=pixel_to_world(ps.right),
                    mid=pixel_to_world(ps.mid),
            ))

        save_path = "slalom_centroids.png"
        cv2.imwrite(save_path, opening)

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
        fx = self.cam.calibration.K[0]
        fy = self.cam.calibration.K[4]
        cx = self.cam.calibration.K[2]
        cy = self.cam.calibration.K[5]

        Z = (fy * real_height) / pixel_height
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return X, Y, Z

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.tfs is not None:
                for t in self.tfs:
                    t.header.stamp = rospy.Time.now()
                    self.tf_broadcaster.sendTransform(t)
            rate.sleep()

if __name__ == "__main__":
    node = SlalomKmeansCluster()
    node.spin()
