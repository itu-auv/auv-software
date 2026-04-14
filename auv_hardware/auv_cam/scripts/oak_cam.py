#!/usr/bin/env python3

import depthai as dai
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


class OakCameraNode:
    def __init__(self):
        rospy.init_node("oak_camera_node")

        self.name = rospy.get_param("~name", "cam_front")

        self.width = rospy.get_param("width", 1280)
        self.height = rospy.get_param("height", 720)
        self.hz = rospy.get_param("fps", 30)

        self.distortion_model = rospy.get_param("distortion_model", "plumb_bob")
        self.D = rospy.get_param("distortion_coefficients/data", [])
        self.K = rospy.get_param("camera_matrix/data", [])
        self.R = rospy.get_param("rectification_matrix/data", [])
        self.P = rospy.get_param("projection_matrix/data", [])

        self.frame_id = f"{self.name}_optical_frame"
        self.bridge = CvBridge()

        self.pub_raw = rospy.Publisher("image_raw", Image, queue_size=1)
        self.pub_info = rospy.Publisher("camera_info", CameraInfo, queue_size=1)

        self.cam_info_msg = self._build_camera_info()

        self.device = dai.Device()
        self.pipeline = dai.Pipeline(self.device)

        self.camRgb = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )
        self.camRgb.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
        )

        self.videoOut = self.camRgb.requestOutput(
            (self.width, self.height), type=dai.ImgFrame.Type.BGR888p
        )
        self.videoQueue = self.videoOut.createOutputQueue(maxSize=4, blocking=False)

        self.pipeline.start()
        self.rate = rospy.Rate(self.hz)

    def _build_camera_info(self):
        msg = CameraInfo()
        msg.header.frame_id = self.frame_id
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = self.distortion_model
        msg.D = list(self.D)
        msg.K = list(self.K)
        msg.R = list(self.R)
        msg.P = list(self.P)
        return msg

    def spin(self):
        while not rospy.is_shutdown():
            videoIn = self.videoQueue.get()
            frame = np.ascontiguousarray(videoIn.getCvFrame())
            stamp = rospy.Time.now()

            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.frame_id

            self.cam_info_msg.stamp = stamp

            self.pub_info.publish(self.cam_info_msg)
            self.pub_raw.publish(img_msg)
            self.rate.sleep()


if __name__ == "__main__":
    OakCameraNode().spin()
