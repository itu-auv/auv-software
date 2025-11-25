#!/usr/bin/env python3
"""
Depth Anything 3 ROS Node - ZeroMQ Client.
"""
import sys
import rospy
import numpy as np
import cv2
import zmq
import collections
import time
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
import tf2_ros
import tf2_geometry_msgs
import auv_common_lib.vision.camera_calibrations as camera_calibrations


class DepthAnythingClient:
    def __init__(self):
        rospy.init_node("depth_anything_node", anonymous=True)

        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.camera_input_topic = rospy.get_param("~input_topic", "/camera/image_raw")
        self.points_topic = rospy.get_param("~output_topic", "/depth_anything/points")
        #! what is the kind of this?
        self.depth_topic = rospy.get_param("~depth_topic", "/depth_anything/depth")

        self.process_res = rospy.get_param("~process_res", 504)
        # External Data Parameters
        self.use_external_odom = rospy.get_param("~use_external_odom", False)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )
        self.camera_frame = rospy.get_param(
            "~camera_frame", "taluy/base_link/front_camera_optical_link"
        )
        self.odom_frame = rospy.get_param("~odom_frame", "odom")

        # Batching Parameters
        self.max_batch_size = rospy.get_param("~max_batch_size", 5)
        self.context_window = rospy.get_param(
            "~context_window", 0.5
        )  # seconds: Time window to look back for context frames

        rospy.loginfo("=" * 60)
        rospy.loginfo("🚀 Depth Anything 3 - ZeroMQ Client Node")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"🔌 ZeroMQ Server: {self.zmq_host}:{self.zmq_port}")
        rospy.loginfo(f"📥 Input topic: {self.camera_input_topic}")
        rospy.loginfo(f"📦 Max Batch Size: {self.max_batch_size}")
        rospy.loginfo(f"🔄 External Odom: {self.use_external_odom}")
        if self.use_external_odom:
            rospy.loginfo(f"🧭 TF Lookup: {self.odom_frame} → {self.camera_frame}")

        # Initialize camera calibration fetcher (keeps subscription active for retries)
        rospy.loginfo(
            f"📷 Subscribing to camera intrinsics: {self.camera_namespace}/camera_info"
        )
        self.camera_calibration_fetcher = camera_calibrations.CameraCalibrationFetcher(
            self.camera_namespace, wait_for_camera_info=False
        )
        self.intrinsics = None
        self._try_fetch_intrinsics()

        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")

        # Set timeouts
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close

        # Check connection
        if not self.check_server():
            rospy.logerr("❌ ZeroMQ server not available! Please start it first.")
            sys.exit(1)

        self.bridge = CvBridge()

        # Initialize TF2 buffer and listener for transform lookups
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Buffer for incoming images
        # Stores tuples of (timestamp, cv_image, header, extrinsics)
        self.image_buffer = collections.deque(maxlen=100)

        self.latest_extrinsics = None

        # Subscribers and Publishers
        self.image_sub = rospy.Subscriber(
            self.camera_input_topic,
            RosImage,
            self.image_callback,
            queue_size=10,  # Increased queue size to allow buffering
            buff_size=2**24,
        )

        self.pcl_pub = rospy.Publisher(self.points_topic, PointCloud2, queue_size=1)
        self.depth_pub = rospy.Publisher(self.depth_topic, RosImage, queue_size=1)

        # Performance metrics
        self.last_inference_time = 0.1  # Initial guess (100ms)
        self.processed_count = 0

        rospy.loginfo("✅ Node started successfully!")
        rospy.loginfo("=" * 60)

    def check_server(self):
        """Check if ZeroMQ server is available."""
        try:
            rospy.loginfo("🔍 Checking server connection...")
            self.socket.send_pyobj({"command": "ping"})
            response = self.socket.recv_pyobj()
            return response.get("status") == "success"
        except Exception as e:
            rospy.logerr(f"❌ Connection check failed: {e}")
            return False

    def _try_fetch_intrinsics(self):
        """Try to fetch camera intrinsics from the camera_info topic."""
        if self.intrinsics is not None:
            return True  # Already have intrinsics

        camera_info = self.camera_calibration_fetcher.get_camera_info()
        if camera_info is not None:
            # Extract intrinsics matrix (3x3) from camera_info.K
            self.intrinsics = np.array(camera_info.K).reshape(3, 3).astype(np.float32)
            rospy.loginfo(f"✅ Camera intrinsics loaded:")
            rospy.loginfo(
                f"   fx={self.intrinsics[0,0]:.2f}, fy={self.intrinsics[1,1]:.2f}"
            )
            rospy.loginfo(
                f"   cx={self.intrinsics[0,2]:.2f}, cy={self.intrinsics[1,2]:.2f}"
            )
            return True
        return False

    def image_callback(self, msg):
        """
        Buffer incoming images with their corresponding camera extrinsics.

        For each image, we look up the transform from odom to camera frame using TF2.
        This gives us the world-to-camera extrinsics needed by Depth Anything 3.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = msg.header.stamp.to_sec()

            # Look up camera extrinsics if external odometry is enabled
            extrinsics = None
            if self.use_external_odom:
                try:
                    # First try to get transform at exact image timestamp
                    transform = self.tf_buffer.lookup_transform(
                        self.odom_frame,
                        self.camera_frame,
                        msg.header.stamp,
                        rospy.Duration(0.05),
                    )
                except tf2_ros.ExtrapolationException:
                    # If image timestamp is in the future, use latest available transform
                    # This happens when camera publishes faster than odometry
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            self.odom_frame,
                            self.camera_frame,
                            rospy.Time(0),  # Use latest available
                            rospy.Duration(0.05),
                        )
                        rospy.logdebug_throttle(
                            5.0, "Using latest TF (image timestamp ahead of TF data)"
                        )
                    except (
                        tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException,
                    ) as e:
                        rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
                        transform = None
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                ) as e:
                    rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
                    transform = None

                if transform is not None:
                    # Convert TransformStamped to 4x4 matrix (world-to-camera)
                    trans = transform.transform.translation
                    rot = transform.transform.rotation

                    # Build the transformation matrix
                    quat = [rot.x, rot.y, rot.z, rot.w]
                    mat = tf_trans.quaternion_matrix(quat)
                    mat[0, 3] = trans.x
                    mat[1, 3] = trans.y
                    mat[2, 3] = trans.z

                    extrinsics = mat.astype(np.float32)

                    # Log the extrinsics matrix in ASCII format (throttled)
                    rospy.loginfo_once("📐 Camera Extrinsics Matrix (odom → camera):")
                    rospy.loginfo_once("   ┌                                      ┐")
                    rospy.loginfo_once(
                        f"   │ {extrinsics[0,0]:7.4f} {extrinsics[0,1]:7.4f} {extrinsics[0,2]:7.4f} │ {extrinsics[0,3]:7.4f} │  ← [R | t]"
                    )
                    rospy.loginfo_once(
                        f"   │ {extrinsics[1,0]:7.4f} {extrinsics[1,1]:7.4f} {extrinsics[1,2]:7.4f} │ {extrinsics[1,3]:7.4f} │  ← Rotation (3×3) | Translation (3×1)"
                    )
                    rospy.loginfo_once(
                        f"   │ {extrinsics[2,0]:7.4f} {extrinsics[2,1]:7.4f} {extrinsics[2,2]:7.4f} │ {extrinsics[2,3]:7.4f} │  ← World→Camera"
                    )
                    rospy.loginfo_once(
                        f"   │ {extrinsics[3,0]:7.4f} {extrinsics[3,1]:7.4f} {extrinsics[3,2]:7.4f} │ {extrinsics[3,3]:7.4f} │  ← Homogeneous"
                    )
                    rospy.loginfo_once("   └                                      ┘")

            # Store image with extrinsics
            self.image_buffer.append((timestamp, cv_image, msg.header, extrinsics))

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def select_batch(self):
        """Select optimal batch of images from buffer."""
        if not self.image_buffer:
            return []

        current_time = rospy.Time.now().to_sec()
        available_frames = list(self.image_buffer)

        # Filter frames within context window
        valid_frames = [
            f for f in available_frames if (current_time - f[0]) <= self.context_window
        ]

        if not valid_frames:
            # If no recent frames, take the latest one available
            valid_frames = [available_frames[-1]]
            rospy.logwarn(
                f"⚠️ No frames in context window ({self.context_window}s). Using latest available frame."
            )

        # Use max batch size but don't exceed available frames
        batch_size = min(len(valid_frames), self.max_batch_size)

        # Uniform sampling (Stride)
        indices = np.linspace(0, len(valid_frames) - 1, batch_size, dtype=int)
        selected_batch = [valid_frames[i] for i in indices]

        # --- Improved Logging ---
        buffer_duration = (
            available_frames[-1][0] - available_frames[0][0]
            if len(available_frames) > 1
            else 0
        )
        delays = [round(current_time - f[0], 3) for f in selected_batch]

        rospy.loginfo(f"📦 Batch Decision:")
        rospy.loginfo(
            f"   ├── Buffer Status: {len(available_frames)} frames (Span: {buffer_duration:.2f}s)"
        )
        rospy.loginfo(
            f"   ├── Context Window: {self.context_window}s -> {len(valid_frames)} valid frames"
        )
        rospy.loginfo(
            f"   ├── Selection: {batch_size} frames (Indices: {indices.tolist()})"
        )
        rospy.loginfo(f"   └── Frame Ages: {delays} sec (Oldest -> Newest)")
        # ------------------------

        # Clear buffer up to the last selected frame to avoid reprocessing old data
        # (Optional: keep some overlap if needed, but for now we clear)
        last_timestamp = selected_batch[-1][0]
        while self.image_buffer and self.image_buffer[0][0] <= last_timestamp:
            self.image_buffer.popleft()

        return selected_batch

    def run(self):
        """Main processing loop."""
        while not rospy.is_shutdown():
            if not self.image_buffer:
                rospy.sleep(0.01)  # Wait for images
                continue

            self.processing_loop()

    def processing_loop(self):
        """Process a single batch."""
        batch = self.select_batch()

        if not batch:
            return

        # Try to fetch intrinsics if not yet available
        self._try_fetch_intrinsics()

        start_time = time.time()

        # Prepare images for inference
        # Convert to RGB and stack
        images_rgb = [cv2.cvtColor(item[1], cv2.COLOR_BGR2RGB) for item in batch]

        # Prepare intrinsics - use the fixed intrinsics loaded at startup
        intrinsics_batch = None
        if self.intrinsics is not None:
            # Replicate the same intrinsics for each image in the batch
            intrinsics_batch = [self.intrinsics for _ in batch]

        # Prepare extrinsics from odometry
        extrinsics_batch = [item[3] for item in batch]

        # If disabled or any frame is missing extrinsics, send None
        if not self.use_external_odom or any(x is None for x in extrinsics_batch):
            extrinsics_batch = None

        # IMPORTANT: Both intrinsics and extrinsics must be provided together
        # If either is None, disable pose-conditioned mode
        if intrinsics_batch is None or extrinsics_batch is None:
            if self.use_external_odom:
                rospy.logwarn_throttle(
                    10.0,
                    "⚠️ Pose-conditioned mode disabled: intrinsics or extrinsics not available",
                )
            intrinsics_batch = None
            extrinsics_batch = None

        # Send to ZeroMQ server
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

            # Update performance metrics
            self.last_inference_time = time.time() - start_time
            self.processed_count += 1

            rospy.loginfo(
                f"⚡ {len(batch)} images took {self.last_inference_time:.4f} seconds to process."
            )

            # Publish result (using the last frame in the batch)
            # The server should return a list of depths, we take the last one
            # corresponding to the most recent image
            depth = response["depth"][-1]
            intrinsics = response["intrinsics"][-1]
            last_header = batch[-1][2]
            last_image = batch[-1][1]  # BGR

            self.publish_results(depth, last_image, intrinsics, last_header)

        except zmq.error.Again:
            rospy.logerr("⏱️ Server timeout")
        except Exception as e:
            rospy.logerr(f"❌ Processing failed: {e}")
            import traceback

            traceback.print_exc()

    def publish_results(self, depth, rgb, K, header):
        """Publish depth map and point cloud."""
        try:
            # Publish Depth Image
            depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)

            # Publish Point Cloud
            rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            self.publish_pointcloud(depth, rgb_rgb, K, header)

        except Exception as e:
            rospy.logerr(f"Publishing failed: {e}")

    def publish_pointcloud(self, depth, rgb, K, header):
        """Generate and publish point cloud."""
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

    def cleanup(self):
        rospy.loginfo("🧹 Cleaning up...")
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
