#!/usr/bin/env python3
"""Depth Anything 3 ROS Node - ZeroMQ Client"""

import cv2
import numpy as np
import rospy
import zmq
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

ZMQ_TIMEOUT_MS = 30000


class DepthAnythingNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.process_res = rospy.get_param("~process_res", 640)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )
        self.frame_id = rospy.get_param(
            "~frame_id", "taluy/base_link/front_camera_optical_link"
        )
        self.publish_rgb_point_cloud = rospy.get_param(
            "~publish_rgb_point_cloud", False
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
        self.point_cloud_pub = None
        self.point_cloud_fields = None
        self.point_cloud_pixels = None
        if self.publish_rgb_point_cloud:
            self.point_cloud_pub = rospy.Publisher(
                "rgb_point_cloud", PointCloud2, queue_size=1
            )
            self.point_cloud_fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

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

    def _infer(self, cv_img: np.ndarray) -> dict:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.socket.send_pyobj(
            {"command": "inference", "image": rgb, "process_res": self.process_res}
        )
        response = self.socket.recv_pyobj()
        if response["status"] != "success":
            raise RuntimeError(response.get("error", "Unknown error"))
        return response

    def _update_intrinsics(self, h: int, w: int) -> None:
        if self.camera_info is None:
            return

        if (
            self.scaled_intrinsics is None
            or self.scaled_intrinsics["width"] != w
            or self.scaled_intrinsics["height"] != h
        ):
            orig_w = self.camera_info.width
            orig_h = self.camera_info.height

            if orig_w == 0 or orig_h == 0:
                rospy.logerr_throttle(
                    5.0, "[DA3] Camera resolution is 0, cannot scale intrinsics."
                )
                return

            scale_w = w / float(orig_w)
            scale_h = h / float(orig_h)

            self.scaled_intrinsics = {
                "width": w,
                "height": h,
                "fx": self.camera_info.K[0] * scale_w,
                "fy": self.camera_info.K[4] * scale_h,
                "cx": self.camera_info.K[2] * scale_w,
                "cy": self.camera_info.K[5] * scale_h,
            }
            rospy.loginfo(f"[DA3] Scaled intrinsics from {orig_w}x{orig_h} to {w}x{h}")
            self._publish_scaled_camera_info()

    def _publish_scaled_camera_info(self) -> None:
        """Publish the scaled camera info as a CameraInfo message."""
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

        # Intrinsic matrix K (3x3 row-major)
        msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        # Rectification matrix R (identity for monocular)
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix P (3x4 row-major)
        msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.camera_info_pub.publish(msg)

    def _update_point_cloud_pixels(self, h: int, w: int) -> None:
        if (
            self.point_cloud_pixels is None
            or self.point_cloud_pixels["width"] != w
            or self.point_cloud_pixels["height"] != h
        ):
            v_coords, u_coords = np.indices((h, w), dtype=np.float32)
            self.point_cloud_pixels = {
                "width": w,
                "height": h,
                "u": u_coords.reshape(-1),
                "v": v_coords.reshape(-1),
            }

    def _publish_rgb_point_cloud(
        self, depth: np.ndarray, bgr_img: np.ndarray, header
    ) -> None:
        if (
            self.point_cloud_pub is None
            or self.point_cloud_pub.get_num_connections() == 0
        ):
            return

        if self.scaled_intrinsics is None:
            rospy.logwarn_throttle(
                5.0, "[DA3] Cannot publish RGB point cloud without intrinsics."
            )
            return

        fx = self.scaled_intrinsics["fx"]
        fy = self.scaled_intrinsics["fy"]
        if fx == 0.0 or fy == 0.0:
            rospy.logwarn_throttle(
                5.0, "[DA3] Cannot publish RGB point cloud with zero focal length."
            )
            return

        h, w = depth.shape[:2]
        self._update_point_cloud_pixels(h, w)

        if bgr_img.shape[:2] != (h, w):
            bgr_img = cv2.resize(bgr_img, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_flat = depth.reshape(-1)
        valid_mask = np.isfinite(depth_flat) & (depth_flat > 0.0)
        point_count = int(np.count_nonzero(valid_mask))

        points = np.zeros(
            point_count,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
            ],
        )

        if point_count > 0:
            z = depth_flat[valid_mask].astype(np.float32, copy=False)
            u = self.point_cloud_pixels["u"][valid_mask]
            v = self.point_cloud_pixels["v"][valid_mask]
            cx = self.scaled_intrinsics["cx"]
            cy = self.scaled_intrinsics["cy"]

            points["x"] = (u - cx) * z / fx
            points["y"] = (v - cy) * z / fy
            points["z"] = z

            colors = bgr_img.reshape(-1, 3)[valid_mask]
            # Pack RGB bytes into the PCL-compatible float32 "rgb" field.
            rgb = (
                (colors[:, 2].astype(np.uint32) << 16)
                | (colors[:, 1].astype(np.uint32) << 8)
                | colors[:, 0].astype(np.uint32)
            )
            points["rgb"] = rgb.view(np.float32)

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = point_count
        cloud_msg.fields = self.point_cloud_fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = points.dtype.itemsize
        cloud_msg.row_step = cloud_msg.point_step * point_count
        cloud_msg.is_dense = True
        cloud_msg.data = points.tobytes()
        self.point_cloud_pub.publish(cloud_msg)

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
                raw_depth = result["depth"]

                h, w = raw_depth.shape[:2]
                self._update_intrinsics(h, w)

                if self.scaled_intrinsics:
                    fx = self.scaled_intrinsics["fx"]
                    fy = self.scaled_intrinsics["fy"]
                    # metric depth: depth = focal * net_output / 300 (as per DA3 paper)
                    focal = (fx + fy) / 2.0
                    depth = raw_depth * focal / 300.0
                else:
                    depth = raw_depth

                depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)

                if self.point_cloud_pub is not None:
                    self._publish_rgb_point_cloud(depth, cv_img, msg.header)

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
