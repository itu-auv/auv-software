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
        self.width = rospy.get_param("~width", 1920)
        self.height = rospy.get_param("~height", 1080)
        self.hz = rospy.get_param("~fps", 30)

        self.frame_id = f"{self.name}_optical_frame"
        self.bridge = CvBridge()

        self.pub_raw = rospy.Publisher("image_raw", Image, queue_size=1)
        self.pub_info = rospy.Publisher("camera_info", CameraInfo, queue_size=1)

        self.cam_info_msg = self._build_default_camera_info()

        self.device = dai.Device()
        self.pipeline = dai.Pipeline(self.device)

        self.camRgb = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )
        self.camRgb.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
        )

        self.videoOut = self.camRgb.requestOutput(
            (self.width, self.height),
            type=dai.ImgFrame.Type.BGR888p
        )
        self.videoQueue = self.videoOut.createOutputQueue(maxSize=4, blocking=False)

        self.pipeline.start()
        self.rate = rospy.Rate(self.hz)

    def _build_default_camera_info(self):
        fx = float(self.width)
        fy = float(self.width)
        cx = self.width / 2.0
        cy = self.height / 2.0

        msg = CameraInfo()
        msg.header.frame_id = self.frame_id
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = "plumb_bob"
        msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.K = [fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.P = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    def spin(self):
        while not rospy.is_shutdown():
            videoIn = self.videoQueue.get()
            frame = np.ascontiguousarray(videoIn.getCvFrame())
            stamp = rospy.Time.now()

            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.frame_id

            info_msg = CameraInfo()
            info_msg.header.stamp = stamp
            info_msg.header.frame_id = self.frame_id
            info_msg.width = self.cam_info_msg.width
            info_msg.height = self.cam_info_msg.height
            info_msg.distortion_model = self.cam_info_msg.distortion_model
            info_msg.D = list(self.cam_info_msg.D)
            info_msg.K = list(self.cam_info_msg.K)
            info_msg.R = list(self.cam_info_msg.R)
            info_msg.P = list(self.cam_info_msg.P)

            self.pub_info.publish(info_msg)
            self.pub_raw.publish(img_msg)
            self.rate.sleep()


if __name__ == "__main__":
    OakCameraNode().spin()