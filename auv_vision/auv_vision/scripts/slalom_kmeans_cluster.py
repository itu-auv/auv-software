#!/usr/bin/env python3

import rospy
import math
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

class SlalomKmeansCluster:
    def __init__(self):
        rospy.init_node("SlalomKmeansCluster")
        self.cam = CameraCalibration("taluy/cameras/cam_front")
        self.yolo_res = rospy.Subscriber("/yolo_result_front", YoloResult, self.yolo_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
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
            rospy.loginfo(f"collected {len(self.points)} points. k-means...")
            self.perform_kmeans()

    def perform_kmeans(self):
        pts = np.array(self.points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0: width = 1.0
        if height == 0: height = 1.0

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

        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
        _, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
        points = np.column_stack(np.where(thresh == 255))
        points = np.float32(points)

        K = 9
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1)

        _, _, pixel_centers = cv2.kmeans(
            points,
            K,
            None,
            criteria,
            100,
            cv2.KMEANS_RANDOM_CENTERS
        )

        print(f"pixel_centers = {pixel_centers}")

        world_centers = []
        for center in pixel_centers:
            v = float(center[0])
            u = float(center[1])
            world_centers.append([((u - (640 - width * scale) / 2) / scale) + x_min, ((v - (480 - height * scale) / 2) / scale) + y_min])

        # TODO: work with the generated (debug?) image instead of pure locations then convert them to locations again
        # benefits:
        # more easy to filter (slalom coordinates are generally small floating points so it's hard to filter them)

        save_path = "slalom_centroids.png"
        cv2.imwrite(save_path, opening)
        rospy.loginfo(f"Image saved to {save_path}")

        self.centers = []
        for i, center in enumerate(world_centers):
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = f"slalom_pipe_{i}"
            t.transform.translation.x = center[0]
            t.transform.translation.y = center[1]
            t.transform.translation.z = 0
            t.transform.rotation.w = 1.0
            self.centers.append(t)

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
            if self.centers is not None:
                for t in self.centers:
                    t.header.stamp = rospy.Time.now()
                    self.tf_broadcaster.sendTransform(t)
            rate.sleep()

if __name__ == "__main__":
    node = SlalomKmeansCluster()
    node.spin()
