#!/usr/bin/env python3
"""
ROS client node for Depth-Anything-3 inference via ZMQ.
Sends images to the ZMQ server and publishes metric depth results.
"""

import io
import sys
from typing import Optional

import cv2
import numpy as np
import rospy
import zmq
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import auv_common_lib.vision.camera_calibrations as camera_calibrations


class DepthAnythingClient:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5555)
        self.rate = rospy.get_param("~rate", 10.0)

        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )

        # Get camera calibration for intrinsics
        self.camera_calibration = camera_calibrations.CameraCalibrationFetcher(
            self.camera_namespace, True
        ).get_camera_info()

        # Extract focal length from intrinsics (average of fx and fy)
        K = np.array(self.camera_calibration.K).reshape(3, 3)
        fx = K[0, 0]
        fy = K[1, 1]
        self.focal_length = (fx + fy) / 2.0
        rospy.loginfo(
            f"Using focal length: {self.focal_length:.2f} pixels (fx={fx:.2f}, fy={fy:.2f})"
        )

        self.latest_image: Optional[Image] = None
        self.processing = False

        self.image_sub = rospy.Subscriber(
            "image_raw", Image, self.image_callback, queue_size=1
        )

        self.depth_pub = rospy.Publisher("depth", Image, queue_size=10)

        # Setup ZMQ client
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")

        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.setsockopt(zmq.LINGER, 0)

        if not self._check_server_connection():
            rospy.logerr("ZeroMQ server not available. Please start the server first.")
            sys.exit(1)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("Depth Anything client node initialized successfully")

    def _check_server_connection(self) -> bool:
        """Check if the ZMQ server is available by sending a test image."""
        try:
            # Create a small test image
            test_img = np.zeros((64, 64, 3), dtype=np.uint8)
            _, img_encoded = cv2.imencode(".jpg", test_img)
            image_bytes = img_encoded.tobytes()

            self.socket.send(image_bytes)
            response = self.socket.recv()

            # Check if response is an error message
            if response.startswith(b"ERROR"):
                rospy.logwarn(f"Server returned error: {response.decode('utf-8')}")
                return False

            rospy.loginfo("ZMQ server connection verified")
            return True

        except zmq.error.Again:
            rospy.logerr("ZMQ server connection timeout")
            return False
        except Exception as e:
            rospy.logerr(f"ZMQ server connection failed: {e}")
            return False

    def image_callback(self, msg: Image) -> None:
        """Store the latest image for processing."""
        if not self.processing:
            self.latest_image = msg

    def _encode_image(self, cv_img: np.ndarray) -> bytes:
        """Encode OpenCV image to JPEG bytes for transmission."""
        _, img_encoded = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return img_encoded.tobytes()

    def _deserialize_depth(self, data: bytes) -> Optional[np.ndarray]:
        """Deserialize depth array from numpy save format."""
        try:
            buffer = io.BytesIO(data)
            depth = np.load(buffer)
            return depth
        except Exception as e:
            rospy.logerr(f"Failed to deserialize depth: {e}")
            return None

    def _convert_to_metric_depth(self, net_output: np.ndarray) -> np.ndarray:
        """
        Convert network output to metric depth in meters.
        Formula: metric_depth = focal * net_output / 300.
        """
        metric_depth = self.focal_length * net_output / 300.0
        return metric_depth.astype(np.float32)

    def _process_image(self, msg: Image) -> None:
        """Send image to server, receive depth, convert to metric, and publish."""
        try:
            # Convert ROS image to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Encode image as JPEG bytes
            image_bytes = self._encode_image(cv_img)

            # Send to ZMQ server
            self.socket.send(image_bytes)

            # Receive depth result
            response = self.socket.recv()

            # Check for error response
            if response.startswith(b"ERROR"):
                rospy.logerr(f"Server error: {response.decode('utf-8')}")
                return

            # Deserialize depth array
            depth = self._deserialize_depth(response)
            if depth is None:
                return

            # Convert to metric depth (meters)
            metric_depth = self._convert_to_metric_depth(depth)

            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(metric_depth, "32FC1")
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)

        except zmq.error.Again:
            rospy.logwarn("ZMQ timeout during inference")
        except Exception as e:
            rospy.logerr(f"Processing failed: {e}")

    def cleanup(self) -> None:
        """Clean up ZMQ resources."""
        rospy.loginfo("Cleaning up ZMQ resources...")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close(linger=0)
        self.context.term()

    def run(self) -> None:
        """Main processing loop."""
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            # Mark as processing to avoid overwriting current image
            self.processing = True
            image_to_process = self.latest_image
            self.latest_image = None

            self._process_image(image_to_process)

            self.processing = False
            rate.sleep()


if __name__ == "__main__":
    node = None
    try:
        rospy.init_node("depth_anything_node")
        node = DepthAnythingClient()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received, shutting down...")
    finally:
        if node:
            node.cleanup()
