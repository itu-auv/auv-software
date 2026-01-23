#!/usr/bin/env python3
"""Depth Anything 3 ROS Node - ZeroMQ Client"""

import cv2
import numpy as np
import rospy
import zmq
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2

ZMQ_TIMEOUT_MS = 30000


class DepthAnythingNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.process_res = rospy.get_param("~process_res", 504)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )

        camera_info_fetcher = CameraCalibrationFetcher(
            self.camera_namespace, wait_for_camera_info=True
        )
        self.camera_info = camera_info_fetcher.get_camera_info()

        self.context = zmq.Context()
        self.socket = None
        self._reset_zmq()

        self.latest_image = None
        self.image_sub = rospy.Subscriber(
            "image_raw", Image, self._image_cb, queue_size=1
        )
        self.depth_pub = rospy.Publisher("raw_depth", Image, queue_size=1)
        self.colorized_pub = rospy.Publisher("colorized", Image, queue_size=1)
        self.points_pub = rospy.Publisher("points", PointCloud2, queue_size=1)

        rospy.on_shutdown(self.cleanup)

        if not self._ping_server():
            rospy.logwarn("[DA3] Server not responding. Will retry in loop.")
        rospy.loginfo(f"[DA3] Ready: {self.zmq_host}:{self.zmq_port}")

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

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        d_u8 = (d_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    def _infer(self, cv_img: np.ndarray) -> dict:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.socket.send_pyobj(
            {"command": "inference", "image": rgb, "process_res": self.process_res}
        )
        response = self.socket.recv_pyobj()
        if response["status"] != "success":
            raise RuntimeError(response.get("error", "Unknown error"))
        return response

    def _depth_to_pointcloud(self, depth: np.ndarray, header) -> PointCloud2:
        if self.camera_info is None:
            rospy.logwarn_throttle(
                5.0, "[DA3] CameraInfo not available yet. Skipping pointcloud."
            )
            return None

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        if fx == 0.0 or fy == 0.0:
            raise ValueError("Invalid camera intrinsics: fx or fy is zero")

        h, w = depth.shape[:2]
        if self.camera_info.width and self.camera_info.height:
            if self.camera_info.width != w or self.camera_info.height != h:
                rospy.logwarn_throttle(
                    5.0,
                    (
                        "[DA3] CameraInfo size (%dx%d) does not match depth (%dx%d). "
                        "Projection may be inaccurate."
                    )
                    % (self.camera_info.width, self.camera_info.height, w, h),
                )

        depth_f = depth.astype(np.float32)
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        z = depth_f
        valid = np.isfinite(z) & (z > 0)
        if not np.any(valid):
            return None

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)
        cloud = pc2.create_cloud_xyz32(header, points)
        return cloud

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

                cloud_msg = self._depth_to_pointcloud(depth, msg.header)
                if cloud_msg is not None:
                    self.points_pub.publish(cloud_msg)
                else:
                    rospy.logwarn_throttle(5.0, "[DA3] No valid points to publish.")

            except (zmq.error.ZMQError, zmq.error.Again) as e:
                rospy.logwarn_throttle(5.0, f"[DA3] ZMQ error: {e}. Reconnecting...")
                self._reset_zmq()
            except Exception as e:
                rospy.logerr_throttle(5.0, f"[DA3] Inference failed: {e}")

            rate.sleep()

    def cleanup(self) -> None:
        rospy.loginfo("[DA3] Shutting down...")
        self.socket.close(linger=0)
        self.context.term()


if __name__ == "__main__":
    rospy.init_node("depth_anything_node")
    try:
        node = DepthAnythingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
