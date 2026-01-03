#!/usr/bin/env python3
"""
Depth Anything 3 ROS Node - ZeroMQ Client.
Sends images to ZMQ inference server and publishes depth + colorized visualization.
"""
import sys

import cv2
import numpy as np
import rospy
import zmq
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import auv_common_lib.vision.camera_calibrations as camera_calibrations

ZMQ_TIMEOUT_MS = 30000


class DepthAnythingClient:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self._load_parameters()
        self._init_camera_intrinsics()
        self._init_zmq()
        self._init_ros()

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo(f"Depth client ready: {self.zmq_host}:{self.zmq_port}")

    def _load_parameters(self) -> None:
        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.process_res = rospy.get_param("~process_res", 504)
        self.camera_namespace = rospy.get_param("~camera_namespace", "cameras/cam_front")
        self.max_batch_size = rospy.get_param("~max_batch_size", 1)

    def _init_camera_intrinsics(self) -> None:
        try:
            fetcher = camera_calibrations.CameraCalibrationFetcher(
                self.camera_namespace, wait_for_camera_info=False
            )
            cam_info = fetcher.get_camera_info()
            if cam_info:
                self.intrinsics = np.array(cam_info.K).reshape(3, 3).astype(np.float32)
                fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
                rospy.loginfo(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}")
            else:
                rospy.logwarn("No camera_info available yet")
                self.intrinsics = None
        except Exception as e:
            rospy.logwarn(f"No camera intrinsics: {e}")
            self.intrinsics = None

    def _init_zmq(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.LINGER, 0)

        if not self._ping_server():
            rospy.logerr("ZMQ server not responding. Start zmq_server.py first.")
            sys.exit(1)

    def _ping_server(self) -> bool:
        try:
            self.socket.send_pyobj({"command": "ping"})
            resp = self.socket.recv_pyobj()
            return resp.get("status") == "success"
        except zmq.error.Again:
            return False

    def _init_ros(self) -> None:
        self.latest_image = None
        self.image_sub = rospy.Subscriber(
            "image_raw", Image, self._image_cb, queue_size=1
        )
        self.depth_pub = rospy.Publisher("~depth", Image, queue_size=1)
        self.colorized_pub = rospy.Publisher("~colorized", Image, queue_size=1)

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = msg

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        d_u8 = (d_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    def _infer(self, cv_img: np.ndarray) -> dict:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        request = {
            "command": "inference",
            "image": rgb,
            "process_res": self.process_res,
        }

        self.socket.send_pyobj(request)
        response = self.socket.recv_pyobj()

        if response["status"] != "success":
            raise RuntimeError(response.get("error", "Unknown error"))

        return response

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            msg = self.latest_image
            self.latest_image = None

            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                result = self._infer(cv_img)

                depth = result["depth"]

                depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)

                colorized = self._colorize_depth(depth)
                color_msg = self.bridge.cv2_to_imgmsg(colorized, "bgr8")
                color_msg.header = msg.header
                self.colorized_pub.publish(color_msg)

            except zmq.error.Again:
                rospy.logwarn_throttle(5.0, "ZMQ timeout")
            except Exception as e:
                rospy.logerr_throttle(5.0, f"Inference failed: {e}")

            rate.sleep()

    def cleanup(self) -> None:
        rospy.loginfo("Shutting down depth client...")
        self.socket.close(linger=0)
        self.context.term()


if __name__ == "__main__":
    rospy.init_node("depth_anything_node")
    try:
        node = DepthAnythingClient()
        node.run()
    except rospy.ROSInterruptException:
        pass
