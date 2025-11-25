#!/usr/bin/env python3
"""
Depth Anything 3 ROS Node - ZeroMQ Client.
This node acts as a ROS interface to a Depth Anything 3 inference server.
It receives image messages, batches them for efficient processing, and publishes
depth maps and point clouds based on the inference results.
"""
import collections
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
import tf2_geometry_msgs
import tf2_ros
import zmq
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2, PointField

import auv_common_lib.vision.camera_calibrations as camera_calibrations

# Configuration constants
ZMQ_RECV_TIMEOUT_MS = 30000
ZMQ_SEND_TIMEOUT_MS = 5000
ZMQ_LINGER_MS = 0
DEFAULT_PROCESS_RESOLUTION = 504
DEFAULT_MAX_BATCH_SIZE = 5
DEFAULT_CONTEXT_WINDOW_SEC = 0.5
IMAGE_BUFFER_MAXLEN = 100
IMAGE_QUEUE_SIZE = 10
IMAGE_BUFFER_SIZE = 2**24
PCL_QUEUE_SIZE = 1
DEPTH_QUEUE_SIZE = 1
TF_LOOKUP_TIMEOUT_SEC = 0.1
INITIAL_INFERENCE_TIME_SEC = 0.1

# Type aliases
ImageBatch = List[Tuple[float, np.ndarray, rospy.Header, Optional[np.ndarray]]]


class DepthAnythingClient:
    """ROS node that interfaces with a Depth Anything 3 inference server via ZeroMQ."""

    def __init__(self) -> None:
        """Initialize the depth estimation client node."""
        rospy.init_node("depth_anything_node", anonymous=True)

        self._load_parameters()
        self._log_configuration()
        self._initialize_camera_intrinsics()
        self._initialize_zmq_connection()
        self._initialize_ros_components()

        rospy.loginfo("Depth Anything client node initialized successfully")

    def _load_parameters(self) -> None:
        """Load ROS parameters for node configuration."""
        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.input_topic = rospy.get_param("~input_topic", "/camera/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/depth_anything/points")
        self.depth_topic = rospy.get_param("~depth_topic", "/depth_anything/depth")
        self.process_res = rospy.get_param("~process_res", DEFAULT_PROCESS_RESOLUTION)

        self.use_external_odom = rospy.get_param("~use_external_odom", False)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )
        self.camera_frame = rospy.get_param(
            "~camera_frame", "taluy/base_link/front_camera_optical_link"
        )
        self.odom_frame = rospy.get_param("~odom_frame", "odom")

        self.max_batch_size = rospy.get_param("~max_batch_size", DEFAULT_MAX_BATCH_SIZE)
        self.context_window = rospy.get_param(
            "~context_window", DEFAULT_CONTEXT_WINDOW_SEC
        )

    def _log_configuration(self) -> None:
        """Log the node configuration for debugging purposes."""
        rospy.loginfo("Depth Anything 3 ZeroMQ Client Node")
        rospy.loginfo(f"ZeroMQ Server: {self.zmq_host}:{self.zmq_port}")
        rospy.loginfo(f"Input topic: {self.input_topic}")
        rospy.loginfo(f"Max batch size: {self.max_batch_size}")
        rospy.loginfo(f"External odometry enabled: {self.use_external_odom}")
        if self.use_external_odom:
            rospy.loginfo(f"TF lookup: {self.odom_frame} -> {self.camera_frame}")

    def _initialize_camera_intrinsics(self) -> None:
        """Fetch camera intrinsics from camera_info topic."""
        rospy.loginfo(
            f"Fetching camera intrinsics from {self.camera_namespace}/camera_info"
        )
        try:
            camera_info = camera_calibrations.CameraCalibrationFetcher(
                self.camera_namespace, wait_for_camera_info=False
            ).get_camera_info()

            if camera_info is not None:
                self.intrinsics = (
                    np.array(camera_info.K).reshape(3, 3).astype(np.float32)
                )
                fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
                cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
                rospy.loginfo(
                    f"Camera intrinsics loaded: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
                )
            else:
                rospy.logwarn(
                    "Could not fetch camera intrinsics. Will use model-estimated values."
                )
                self.intrinsics = None
        except Exception as e:
            rospy.logwarn(
                f"Failed to fetch camera intrinsics: {e}. Using model-estimated values."
            )
            self.intrinsics = None

    def _initialize_zmq_connection(self) -> None:
        """Initialize ZeroMQ connection to inference server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")

        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_SEND_TIMEOUT_MS)
        self.socket.setsockopt(zmq.LINGER, ZMQ_LINGER_MS)

        if not self._check_server_connection():
            rospy.logerr("ZeroMQ server not available. Please start the server first.")
            sys.exit(1)

    def _initialize_ros_components(self) -> None:
        """Initialize ROS subscribers, publishers, and internal buffers."""
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.image_buffer = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.latest_extrinsics = None

        self.image_sub = rospy.Subscriber(
            self.input_topic,
            RosImage,
            self.image_callback,
            queue_size=IMAGE_QUEUE_SIZE,
            buff_size=IMAGE_BUFFER_SIZE,
        )

        self.pcl_pub = rospy.Publisher(
            self.output_topic, PointCloud2, queue_size=PCL_QUEUE_SIZE
        )
        self.depth_pub = rospy.Publisher(
            self.depth_topic, RosImage, queue_size=DEPTH_QUEUE_SIZE
        )

        self.last_inference_time = INITIAL_INFERENCE_TIME_SEC
        self.processed_count = 0

    def _check_server_connection(self) -> bool:
        """Verify ZeroMQ server is available and responsive.
        Returns:
            True if server responds successfully, False otherwise.
        """
        try:
            rospy.loginfo("Checking ZeroMQ server connection...")
            self.socket.send_pyobj({"command": "ping"})
            response = self.socket.recv_pyobj()
            return response.get("status") == "success"
        except Exception as e:
            rospy.logerr(f"Connection check failed: {e}")
            return False

    def image_callback(self, msg: RosImage) -> None:
        """Process incoming image messages and buffer them with camera extrinsics.
        Args:
            msg: ROS Image message containing the camera frame.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = msg.header.stamp.to_sec()

            extrinsics = None
            if self.use_external_odom:
                extrinsics = self._lookup_camera_extrinsics(msg.header.stamp)

            self.image_buffer.append((timestamp, cv_image, msg.header, extrinsics))

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def _lookup_camera_extrinsics(self, timestamp: rospy.Time) -> Optional[np.ndarray]:
        """Look up camera extrinsics from TF tree.
        Args:
            timestamp: Time at which to look up the transform.
        Returns:
            4x4 transformation matrix from odom to camera frame, or None if lookup fails.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.camera_frame,
                timestamp,
                rospy.Duration(TF_LOOKUP_TIMEOUT_SEC),
            )

            trans = transform.transform.translation
            rot = transform.transform.rotation

            quat = [rot.x, rot.y, rot.z, rot.w]
            mat = tf_trans.quaternion_matrix(quat)
            mat[0, 3] = trans.x
            mat[1, 3] = trans.y
            mat[2, 3] = trans.z

            extrinsics = mat.astype(np.float32)

            rospy.loginfo_once("Camera extrinsics matrix (odom to camera) acquired")

            return extrinsics

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
            return None

    def select_batch(self) -> ImageBatch:
        """Select optimal batch of images from buffer for processing.
        Returns:
            List of tuples containing (timestamp, cv_image, header, extrinsics).
        """
        if not self.image_buffer:
            return []

        current_time = rospy.Time.now().to_sec()
        available_frames = list(self.image_buffer)

        valid_frames = [
            f for f in available_frames if (current_time - f[0]) <= self.context_window
        ]

        if not valid_frames:
            valid_frames = [available_frames[-1]]
            rospy.logwarn(
                f"No frames in context window ({self.context_window}s). Using latest available frame."
            )

        batch_size = min(len(valid_frames), self.max_batch_size)

        indices = np.linspace(0, len(valid_frames) - 1, batch_size, dtype=int)
        selected_batch = [valid_frames[i] for i in indices]

        self._log_batch_selection(
            available_frames, valid_frames, selected_batch, batch_size, current_time
        )

        last_timestamp = selected_batch[-1][0]
        while self.image_buffer and self.image_buffer[0][0] <= last_timestamp:
            self.image_buffer.popleft()

        return selected_batch

    def _log_batch_selection(
        self,
        available_frames: ImageBatch,
        valid_frames: ImageBatch,
        selected_batch: ImageBatch,
        batch_size: int,
        current_time: float,
    ) -> None:
        """Log information about batch selection for debugging."""
        buffer_duration = (
            available_frames[-1][0] - available_frames[0][0]
            if len(available_frames) > 1
            else 0
        )
        delays = [round(current_time - f[0], 3) for f in selected_batch]

        rospy.loginfo(
            f"Batch selection: {len(available_frames)} frames available "
            f"(span: {buffer_duration:.2f}s), {len(valid_frames)} valid frames "
            f"in {self.context_window}s window, selected {batch_size} frames "
            f"with ages {delays}s"
        )

    def run(self) -> None:
        """Main processing loop that continuously processes incoming image batches."""
        while not rospy.is_shutdown():
            if not self.image_buffer:
                rospy.sleep(0.01)
                continue

            self._process_batch()

    def _process_batch(self) -> None:
        """Process a single batch of images through the inference server."""
        batch = self.select_batch()

        if not batch:
            return

        start_time = time.time()

        images_rgb = [cv2.cvtColor(item[1], cv2.COLOR_BGR2RGB) for item in batch]

        intrinsics_batch = None
        if self.intrinsics is not None:
            intrinsics_batch = [self.intrinsics for _ in batch]

        extrinsics_batch = [item[3] for item in batch]

        if not self.use_external_odom or any(x is None for x in extrinsics_batch):
            extrinsics_batch = None

        if intrinsics_batch is None or extrinsics_batch is None:
            if self.use_external_odom:
                rospy.logwarn_throttle(
                    10.0,
                    "Pose-conditioned mode disabled: intrinsics or extrinsics unavailable",
                )
            intrinsics_batch = None
            extrinsics_batch = None

        try:
            request = {
                "command": "inference_batch",
                "images": images_rgb,
                "process_res": self.process_res,
                "intrinsics": intrinsics_batch,
                "extrinsics": extrinsics_batch,
            }

            self.socket.send_pyobj(request, protocol=2)
            response = self.socket.recv_pyobj()

            if response["status"] != "success":
                rospy.logerr(f"Server error: {response.get('error')}")
                return

            self.last_inference_time = time.time() - start_time
            self.processed_count += 1

            rospy.loginfo(
                f"Processed {len(batch)} images in {self.last_inference_time:.4f}s"
            )

            depth = response["depth"][-1]
            intrinsics = response["intrinsics"][-1]
            last_header = batch[-1][2]
            last_image = batch[-1][1]

            self._publish_results(depth, last_image, intrinsics, last_header)

        except zmq.error.Again:
            rospy.logerr("Server timeout")
        except Exception as e:
            rospy.logerr(f"Processing failed: {e}")
            import traceback

            traceback.print_exc()

    def _publish_results(
        self, depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, header: rospy.Header
    ) -> None:
        """Publish depth map and point cloud results.
        Args:
            depth: Depth map array.
            rgb: BGR image array.
            K: Camera intrinsics matrix.
            header: ROS message header.
        """
        try:
            depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)

            rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            self._publish_pointcloud(depth, rgb_rgb, K, header)

        except Exception as e:
            rospy.logerr(f"Publishing failed: {e}")

    def _publish_pointcloud(
        self, depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, header: rospy.Header
    ) -> None:
        """Generate and publish point cloud from depth map and RGB image.
        Args:
            depth: Depth map array.
            rgb: RGB image array.
            K: Camera intrinsics matrix.
            header: ROS message header.
        """
        height, width = depth.shape
        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb = cv2.resize(rgb, (width, height))

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        r = rgb[:, :, 0].reshape(-1).astype(np.uint32)
        g = rgb[:, :, 1].reshape(-1).astype(np.uint32)
        b = rgb[:, :, 2].reshape(-1).astype(np.uint32)

        valid = (z > 0) & np.isfinite(z)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        r = r[valid]
        g = g[valid]
        b = b[valid]

        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = rgb_int.view(np.float32)

        points = np.column_stack((x, y, z, rgb_float))

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points)
        self.pcl_pub.publish(pc_msg)

    def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        rospy.loginfo("Cleaning up resources...")
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    node = None
    try:
        node = DepthAnythingClient()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node crashed: {e}")
    finally:
        if node:
            node.cleanup()
