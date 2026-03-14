#!/usr/bin/env python3

import numpy as np
import depthai as dai
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge


class OakCameraNode:
    def __init__(self):
        rospy.init_node("oak_camera_node")

        self.namespace = rospy.get_param("~namespace", "taluy_mini")
        self.name = rospy.get_param("~name", "cam_front")

        self.width = 1280
        self.height = 720
        self.hz = 30

        self.distortion_model = rospy.get_param(
            f"~{self.name}/distortion_model", "plumb_bob"
        )
        D = rospy.get_param(f"~{self.name}/distortion_coefficients/data", [])
        K = rospy.get_param(f"~{self.name}/camera_matrix/data", [])
        R = rospy.get_param(f"~{self.name}/rectification_matrix/data", [])
        P = rospy.get_param(f"~{self.name}/projection_matrix/data", [])

        self.D = D
        self.K_list = K
        self.R_list = R
        self.P_list = P

        self.K_mat = (
            np.array(K, dtype=np.float64).reshape(3, 3) if len(K) == 9 else None
        )
        self.D_mat = np.array(D, dtype=np.float64) if D else None
        self.R_mat = (
            np.array(R, dtype=np.float64).reshape(3, 3) if len(R) == 9 else None
        )
        self.P_mat = (
            np.array(P, dtype=np.float64).reshape(3, 4) if len(P) == 12 else None
        )

        self.can_rectify = (
            self.K_mat is not None
            and self.D_mat is not None
            and self.R_mat is not None
            and self.P_mat is not None
        )
        if self.can_rectify:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K_mat, self.D_mat, (self.width, self.height), alpha=0
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K_mat,
                self.D_mat,
                self.R_mat,
                new_K,
                (self.width, self.height),
                cv2.CV_16SC2,
            )
        else:
            rospy.logwarn("invalid calibration parameters")

        self.frame_id = f"{self.name}_optical_frame"
        self.bridge = CvBridge()

        self.pub_raw = rospy.Publisher("image_raw", Image, queue_size=5)
        self.pub_raw_compressed = rospy.Publisher(
            "image_raw/compressed", CompressedImage, queue_size=5
        )
        self.pub_info = rospy.Publisher("camera_info", CameraInfo, queue_size=5)
        self.pub_rect = rospy.Publisher("image_rect_color", Image, queue_size=5)
        self.pub_rect_compressed = rospy.Publisher(
            "image_rect_color/compressed", CompressedImage, queue_size=5
        )

        self.cam_info_msg = self._build_camera_info()

        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setVideoSize(self.width, self.height)
        self.camRgb.setInterleaved(False)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setFps(self.hz)
        self.camRgb.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
        )

        self.videoQueue = self.camRgb.video.createOutputQueue()
        self.pipeline.start()

        self.rate = rospy.Rate(self.hz)
        rospy.loginfo("OAK camera node started")

    def _build_camera_info(self):
        msg = CameraInfo()
        msg.header.frame_id = self.frame_id
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = self.distortion_model
        msg.D = self.D
        msg.K = self.K_list
        msg.R = self.R_list
        msg.P = self.P_list
        msg.binning_x = 1
        msg.binning_y = 1
        msg.roi.width = self.width
        msg.roi.height = self.height
        msg.roi.do_rectify = False
        return msg

    def _make_compressed(self, frame):
        msg = CompressedImage()
        msg.format = "jpeg"
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        msg.data = buf.tobytes()
        return msg

    def spin(self):
        while not rospy.is_shutdown():
            if not self.pipeline.isRunning():
                continue

            videoIn = self.videoQueue.get()
            if videoIn is None:
                continue

            now = rospy.Time.now()
            frame = videoIn.getCvFrame()

            header_stamp = now
            header_frame = self.frame_id
            raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            raw_msg.header.stamp = header_stamp
            raw_msg.header.frame_id = header_frame

            raw_compressed = self._make_compressed(frame)
            raw_compressed.header.stamp = header_stamp
            raw_compressed.header.frame_id = header_frame

            self.cam_info_msg.header.stamp = header_stamp

            self.pub_raw.publish(raw_msg)
            self.pub_raw_compressed.publish(raw_compressed)
            self.pub_info.publish(self.cam_info_msg)

            if self.can_rectify:
                rectified = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                rect_msg = self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8")
                rect_msg.header.stamp = header_stamp
                rect_msg.header.frame_id = header_frame

                rect_compressed = self._make_compressed(rectified)
                rect_compressed.header.stamp = header_stamp
                rect_compressed.header.frame_id = header_frame

                self.pub_rect.publish(rect_msg)
                self.pub_rect_compressed.publish(rect_compressed)

            self.rate.sleep()


if __name__ == "__main__":
    node = OakCameraNode()
    node.spin()
