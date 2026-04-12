#!/usr/bin/env python3
"""Depth Anything 3 ROS Node - TensorRT Inference

This node runs Depth Anything 3 metric depth inference using a TensorRT engine,
replacing the previous ZMQ-based architecture that required Python 3.10.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rospy
import tensorrt as trt
import pycuda.driver as cuda

# Initialize CUDA manually (pycuda.autoinit uses Python 3.10+ syntax)
cuda.init()
_cuda_device = cuda.Device(0)
_cuda_context = _cuda_device.make_context()

from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

# ============================================================================
# Preprocessing Constants (Depth Anything 3 defaults)
# ============================================================================
# Model input dimensions (fixed at ONNX export time)
MODEL_INPUT_HEIGHT = 476
MODEL_INPUT_WIDTH = 644

# Normalization: ImageNet mean/std (DA3 uses standard ImageNet normalization)
# Applied as: normalized = (pixel / 255.0 - mean) / std
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Metric depth conversion factor (from DA3 paper)
# metric_depth = raw_depth * focal_length / DEPTH_SCALE_FACTOR
DEPTH_SCALE_FACTOR = 300.0


class TensorRTEngine:
    """TensorRT engine wrapper for inference."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

        return inputs, outputs, bindings, stream

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        # Copy input to host buffer
        np.copyto(self.inputs[0]["host"], input_data.ravel())

        # Transfer to device
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer output from device
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])


class DepthAnythingTRTNode:
    """ROS node for Depth Anything 3 using TensorRT."""

    def __init__(self) -> None:
        self.bridge = CvBridge()

        # Parameters
        engine_path = rospy.get_param("~engine_path")
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_torpedo"
        )
        self.frame_id = rospy.get_param(
            "~frame_id", "taluy/base_link/torpedo_camera_optical_link"
        )
        self.publish_rgb_point_cloud = rospy.get_param(
            "~publish_rgb_point_cloud", False
        )

        # Validate engine path
        if not Path(engine_path).exists():
            rospy.logfatal(f"[DA3-TRT] Engine file not found: {engine_path}")
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        # Load TensorRT engine
        rospy.loginfo(f"[DA3-TRT] Loading engine: {engine_path}")
        self.engine = TensorRTEngine(engine_path)
        rospy.loginfo("[DA3-TRT] Engine loaded successfully")

        # Camera intrinsics
        camera_info_fetcher = CameraCalibrationFetcher(
            self.camera_namespace, wait_for_camera_info=True
        )
        self.camera_info = camera_info_fetcher.get_camera_info()
        self.scaled_intrinsics = self._compute_scaled_intrinsics()

        # ROS interface
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

        # Publish scaled camera info once
        if self.scaled_intrinsics:
            self._publish_scaled_camera_info()

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo(f"[DA3-TRT] Ready, rate={self.rate_hz}Hz")

    def _compute_scaled_intrinsics(self) -> dict:
        """Pre-compute scaled intrinsics for fixed model input size."""
        if self.camera_info is None:
            return None

        orig_w = self.camera_info.width
        orig_h = self.camera_info.height

        if orig_w == 0 or orig_h == 0:
            rospy.logerr("[DA3-TRT] Camera resolution is 0, cannot scale intrinsics.")
            return None

        scale_w = MODEL_INPUT_WIDTH / float(orig_w)
        scale_h = MODEL_INPUT_HEIGHT / float(orig_h)

        scaled = {
            "width": MODEL_INPUT_WIDTH,
            "height": MODEL_INPUT_HEIGHT,
            "fx": self.camera_info.K[0] * scale_w,
            "fy": self.camera_info.K[4] * scale_h,
            "cx": self.camera_info.K[2] * scale_w,
            "cy": self.camera_info.K[5] * scale_h,
        }
        rospy.loginfo(
            f"[DA3-TRT] Scaled intrinsics: {orig_w}x{orig_h} -> "
            f"{MODEL_INPUT_WIDTH}x{MODEL_INPUT_HEIGHT}"
        )
        return scaled

    def _publish_scaled_camera_info(self) -> None:
        """Publish scaled camera info as CameraInfo message."""
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

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = msg

    def _preprocess(self, cv_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for TensorRT inference.

        Pipeline:
        1. Resize to model input size (644x476)
        2. Convert BGR -> RGB
        3. Normalize to [0,1] and apply ImageNet mean/std
        4. Transpose HWC -> CHW
        5. Add batch dimension
        """
        # Resize to model input size
        resized = cv2.resize(
            cv_img,
            (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize: [0,255] -> [0,1] -> apply mean/std
        normalized = (rgb.astype(np.float32) / 255.0 - NORMALIZE_MEAN) / NORMALIZE_STD

        # HWC -> CHW
        chw = normalized.transpose(2, 0, 1)

        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        return np.expand_dims(chw, axis=0).astype(np.float32), resized

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
                5.0, "[DA3-TRT] Cannot publish RGB point cloud without intrinsics."
            )
            return

        fx = self.scaled_intrinsics["fx"]
        fy = self.scaled_intrinsics["fy"]
        if fx == 0.0 or fy == 0.0:
            rospy.logwarn_throttle(
                5.0,
                "[DA3-TRT] Cannot publish RGB point cloud with zero focal length.",
            )
            return

        h, w = depth.shape[:2]
        self._update_point_cloud_pixels(h, w)

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
        """Main loop."""
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

                # Preprocess and run inference
                input_tensor, resized_bgr = self._preprocess(cv_img)
                raw_output = self.engine.infer(input_tensor)

                # Output shape: (1, 1, H, W) -> (H, W)
                raw_depth = raw_output.squeeze()

                # Convert to metric depth
                if self.scaled_intrinsics:
                    fx = self.scaled_intrinsics["fx"]
                    fy = self.scaled_intrinsics["fy"]
                    focal = (fx + fy) / 2.0
                    depth = raw_depth * focal / DEPTH_SCALE_FACTOR
                else:
                    depth = raw_depth

                # Publish depth
                depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)

                if self.point_cloud_pub is not None:
                    self._publish_rgb_point_cloud(depth, resized_bgr, msg.header)

            except Exception as e:
                rospy.logerr_throttle(5.0, f"[DA3-TRT] Inference failed: {e}")

            rate.sleep()

    def cleanup(self) -> None:
        """Release CUDA and TensorRT resources."""
        rospy.loginfo("[DA3-TRT] Shutting down...")
        try:
            if hasattr(self, "engine") and self.engine is not None:
                del self.engine
        finally:
            try:
                _cuda_context.pop()
            except Exception:
                pass


if __name__ == "__main__":
    rospy.init_node("depth_anything_trt_node")
    try:
        node = DepthAnythingTRTNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
