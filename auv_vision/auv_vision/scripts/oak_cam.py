#!/usr/bin/env python3

import depthai as dai
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OakCameraNode:
    def __init__(self):
        rospy.init_node('oak_camera_node')

        self.hz = 30
        self.width = 1280
        self.height = 720
        self.topic_name = '/taluy_mini/cameras/cam_front/image_raw'

        self.image_pub = rospy.Publisher(self.topic_name, Image, queue_size=10)
        self.bridge = CvBridge()

        self.pipeline = dai.Pipeline()

        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camRgb.setPreviewSize(self.width, self.height) 
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setInterleaved(False)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setFps(self.hz)

        self.camRgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        self.videoQueue = self.camRgb.video.createOutputQueue()
        self.pipeline.start()

        self.rate = rospy.Rate(self.hz)

    def spin(self):
        while not rospy.is_shutdown():
            if not self.pipeline.isRunning():
                self.rate.sleep()
                continue
            videoIn = self.videoQueue.get()
            if videoIn is not None:
                frame = videoIn.getCvFrame()
                image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                
                image_msg.header.stamp = rospy.Time.now()
                image_msg.header.frame_id = "cam_front_link"

                self.image_pub.publish(image_msg)

            self.rate.sleep()

if __name__ == '__main__':
    node = OakCameraNode()
    node.spin()