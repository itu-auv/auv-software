#!/usr/bin/env python3
"""
Test script for MegaPose ROS integration.
Publishes Image + CameraInfo + Bbox and listens for Pose.
"""

import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, PoseArray
from cv_bridge import CvBridge

# Configuration
IMAGE_PATH = "/home/agxorin/megapose/meshes/Robosub_objects/Bottle/frame_1090.jpg"
BBOX = [273.0, 325.0, 367.0, 400.0]  # x1, y1, x2, y2
TOPIC_IMAGE = "/taluy/camera/color/image_raw"
TOPIC_INFO = "/taluy/camera/color/camera_info"
TOPIC_BBOX = "/taluy/megapose_node/bbox"
TOPIC_POSE = "/taluy/megapose_node/pose"

class MegaPoseTester:
    def __init__(self):
        rospy.init_node("megapose_tester")
        
        self.bridge = CvBridge()
        
        # Publishers
        self.pub_image = rospy.Publisher(TOPIC_IMAGE, Image, queue_size=1)
        self.pub_info = rospy.Publisher(TOPIC_INFO, CameraInfo, queue_size=1)
        self.pub_bbox = rospy.Publisher(TOPIC_BBOX, Float32MultiArray, queue_size=1)
        
        # Subscriber
        # rospy.Subscriber(TOPIC_POSE, PoseStamped, self.pose_callback)
        
        # Load image
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            rospy.logerr(f"Failed to load image: {IMAGE_PATH}")
            return
    
    # def pose_callback(self, msg):
    #     pos = msg.pose.position
    #     rot = msg.pose.orientation
    #     rospy.loginfo(f"\n[RECEIVED POSE]\n  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}\n  Orientation: w={rot.w:.3f}, x={rot.x:.3f}, y={rot.y:.3f}, z={rot.z:.3f}")
    
    def publish(self):
        # Publish Camera Info
        info_msg = CameraInfo()
        info_msg.header.stamp = rospy.Time.now()
        info_msg.header.frame_id = "taluy/base_link/front_camera_optical_link"
        info_msg.width = self.image.shape[1]
        info_msg.height = self.image.shape[0]
        # Example K (fx=600, fy=600, cx=320, cy=240)
        info_msg.K = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
        self.pub_info.publish(info_msg)
        
        # Publish Image
        img_msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8")
        img_msg.header = info_msg.header
        self.pub_image.publish(img_msg)
        
        # Publish BBox
        bbox_msg = Float32MultiArray()
        bbox_msg.data = BBOX
        self.pub_bbox.publish(bbox_msg)
        
        rospy.loginfo("Published Image + Info + Bbox")

    def run(self):
        rate = rospy.Rate(1) # 1 Hz
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

if __name__ == "__main__":
    tester = MegaPoseTester()
    tester.run()
