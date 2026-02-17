#!/usr/bin/env python3

import rospy
import math
import tf2_ros
import tf2_geometry_msgs
from ultralytics_ros.msg import YoloResult
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
        self.yolo_res = rospy.Subscriber("/yolo_result_front", YoloResult, self.yolo_callback)
        self.cam = CameraCalibration("taluy/cameras/cam_front")
        print(self.cam)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.i = 0

    def yolo_callback(self, msg):
        detections: Detection2DArray = msg.detections
        masks_msgs = list(msg.masks) if msg.masks else []
        if len(detections.detections) == 0: return
        for x in detections.detections:
            self.i += 1
            bbox = x.bbox
            height = math.sqrt(bbox.size_x ** 2 + bbox.size_y ** 2)
            off_x, off_y, off_z = self.world_pos_from_height(0.9, bbox.size_y, bbox.center.x, bbox.center.y)
            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = detections.header.stamp
            transform_stamped_msg.header.frame_id = "taluy/base_link/front_camera_optical_link_stabilized"
            transform_stamped_msg.child_frame_id = f"pipe_{self.i}"
            transform_stamped_msg.transform.translation = Vector3(
                off_x, off_y, off_z
            )
            transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
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
                final_transform_stamped.child_frame_id = f"pipe_{self.i}"
                final_transform_stamped.transform.translation = (
                    transformed_pose_stamped.pose.position
                )
                final_transform_stamped.transform.rotation = (
                    transform_stamped_msg.transform.rotation
                )

                self.tf_broadcaster.sendTransform(final_transform_stamped)
            except Exception as e:
                rospy.logwarn(f"ERROR: {e}")


    def world_pos_from_height(self, real_height, pixel_height, u, v):
        fx = self.cam.calibration.K[0]
        fy = self.cam.calibration.K[4]
        cx = self.cam.calibration.K[2]
        cy = self.cam.calibration.K[5]

        Z = (fy * real_height) / pixel_height

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return X, Y, Z

if __name__ == "__main__":
    SlalomKmeansCluster()
    rospy.spin()
