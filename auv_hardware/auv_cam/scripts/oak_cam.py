#!/usr/bin/env python3

import depthai as dai
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class OakCameraNode:
    def __init__(self):
        rospy.init_node("oak_camera_node")

        self.name = rospy.get_param("~name", "cam_front")

        self.width = rospy.get_param(f"{self.name}/width", 1280)
        self.height = rospy.get_param(f"{self.name}/height", 720)
        self.hz = rospy.get_param(f"{self.name}/fps", 30)

        self.distortion_model = rospy.get_param(
            f"{self.name}/distortion_model", "plumb_bob"
        )
        D = rospy.get_param(f"{self.name}/distortion_coefficients/data", [])
        K = rospy.get_param(f"{self.name}/camera_matrix/data", [])
        R = rospy.get_param(f"{self.name}/rectification_matrix/data", [])
        P = rospy.get_param(f"{self.name}/projection_matrix/data", [])

        self.D = D
        self.K_list = K
        self.R_list = R
        self.P_list = P

        self.frame_id = f"{self.name}_optical_frame"
        self.bridge = CvBridge()

        self.pub_raw = rospy.Publisher("image_raw", Image, queue_size=5)
        self.pub_info = rospy.Publisher("camera_info", CameraInfo, queue_size=5)

        self.cam_info_msg = self._build_camera_info()
        self.publish_camera_info = self.cam_info_msg is not None

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
        if len(self.K_list) != 9 or len(self.R_list) != 9 or len(self.P_list) != 12:
            rospy.logwarn(
                "Incomplete camera calibration for %s. "
                "Expected K=9, R=9, P=12 values but got K=%d, R=%d, P=%d. "
                "Skipping camera_info publication.",
                self.name,
                len(self.K_list),
                len(self.R_list),
                len(self.P_list),
            )
            return None

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

            self.pub_raw.publish(raw_msg)
            if self.publish_camera_info:
                self.cam_info_msg.header.stamp = header_stamp
                self.pub_info.publish(self.cam_info_msg)

            self.rate.sleep()


if __name__ == "__main__":
    node = OakCameraNode()
    node.spin()
