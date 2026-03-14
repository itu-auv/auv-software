#!/usr/bin/env python3
"""MetricAnything DepthMap ROS Node - ZeroMQ client."""

import cv2
import numpy as np
import rospy
import zmq
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

ZMQ_TIMEOUT_MS = 30000


class MetricDepthNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )
        self.frame_id = rospy.get_param(
            "~frame_id", "taluy/base_link/front_camera_optical_link"
        )

        camera_info_fetcher = CameraCalibrationFetcher(
            self.camera_namespace, wait_for_camera_info=True
        )
        self.camera_info = camera_info_fetcher.get_camera_info()
        self.scaled_intrinsics = None

        self.context = zmq.Context()
        self.socket = None
        self._reset_zmq()

        self.latest_image = None
        self.image_sub = rospy.Subscriber(
            "image_raw", Image, self._image_cb, queue_size=1
        )
        self.depth_pub = rospy.Publisher("raw_depth", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher(
            "scaled_camera_info", CameraInfo, queue_size=1, latch=True
        )

        rospy.on_shutdown(self.cleanup)

        if not self._ping_server():
            rospy.logwarn("[MetricDepth] Server not responding. Will retry in loop.")
        rospy.loginfo(f"[MetricDepth] Ready: {self.zmq_host}:{self.zmq_port}")

    def _reset_zmq(self) -> None:
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.LINGER, 0)

    def _ping_server(self) -> bool:
        try:
            self.socket.send_pyobj({"command": "ping"})
            return self.socket.recv_pyobj().get("status") == "success"
        except zmq.error.Again:
            return False

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = msg

    def _compute_scaled_intrinsics(self, h: int, w: int):
        if self.camera_info is None:
            return None

        orig_w = self.camera_info.width
        orig_h = self.camera_info.height
        if orig_w == 0 or orig_h == 0:
            rospy.logerr_throttle(
                5.0, "[MetricDepth] Camera resolution is 0, cannot scale intrinsics."
            )
            return None

        scale_w = w / float(orig_w)
        scale_h = h / float(orig_h)
        return {
            "width": w,
            "height": h,
            "fx": self.camera_info.K[0] * scale_w,
            "fy": self.camera_info.K[4] * scale_h,
            "cx": self.camera_info.K[2] * scale_w,
            "cy": self.camera_info.K[5] * scale_h,
        }

    def _infer(self, cv_img: np.ndarray) -> dict:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        req_intrinsics = self._compute_scaled_intrinsics(h, w)

        payload = {
            "command": "inference",
            "image": rgb,
        }
        if req_intrinsics is not None:
            payload["f_px"] = (req_intrinsics["fx"] + req_intrinsics["fy"]) / 2.0
        else:
            rospy.logwarn_throttle(
                10.0,
                "[MetricDepth] Camera intrinsics unavailable; server will fallback to image width.",
            )

        self.socket.send_pyobj(payload)
        response = self.socket.recv_pyobj()
        if response["status"] != "success":
            raise RuntimeError(response.get("error", "Unknown error"))
        return response

    def _update_intrinsics(self, h: int, w: int) -> None:
        if self.camera_info is None:
            return

        if (
            self.scaled_intrinsics is not None
            and self.scaled_intrinsics["width"] == w
            and self.scaled_intrinsics["height"] == h
        ):
            return

        scaled = self._compute_scaled_intrinsics(h, w)
        if scaled is None:
            return

        self.scaled_intrinsics = scaled
        rospy.loginfo(
            f"[MetricDepth] Scaled intrinsics from {self.camera_info.width}x{self.camera_info.height} to {w}x{h}"
        )
        self._publish_scaled_camera_info()

    def _publish_scaled_camera_info(self) -> None:
        if self.scaled_intrinsics is None or self.camera_info is None:
            return

        msg = CameraInfo()
        msg.header = self.camera_info.header
        msg.width = self.scaled_intrinsics["width"]
        msg.height = self.scaled_intrinsics["height"]
        msg.distortion_model = self.camera_info.distortion_model
        msg.D = self.camera_info.D

        fx = self.scaled_intrinsics["fx"]
        fy = self.scaled_intrinsics["fy"]
        cx = self.scaled_intrinsics["cx"]
        cy = self.scaled_intrinsics["cy"]

        msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.camera_info_pub.publish(msg)

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            msg = self.latest_image
            self.latest_image = None

            msg.header.frame_id = self.frame_id
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                result = self._infer(cv_img)
                depth = np.asarray(result["depth"], dtype=np.float32)

                h, w = depth.shape[:2]
                self._update_intrinsics(h, w)

                depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)

            except (zmq.error.ZMQError, zmq.error.Again) as exc:
                rospy.logwarn_throttle(
                    5.0, f"[MetricDepth] ZMQ error: {exc}. Reconnecting..."
                )
                self._reset_zmq()
            except Exception as exc:
                rospy.logerr_throttle(5.0, f"[MetricDepth] Inference failed: {exc}")

            rate.sleep()

    def cleanup(self) -> None:
        rospy.loginfo("[MetricDepth] Shutting down...")
        try:
            if self.socket is not None:
                self.socket.close(linger=0)
        finally:
            self.context.term()


if __name__ == "__main__":
    rospy.init_node("metric_depth_node")
    try:
        node = MetricDepthNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
