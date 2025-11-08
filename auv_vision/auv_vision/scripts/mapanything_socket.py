#!/usr/bin/env python3

import pickle
from typing import Optional, List

import cv2
import numpy as np
import zmq
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

import auv_common_lib.vision.camera_calibrations as camera_calibrations


class MapAnythingROSNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.capture_interval_sec = float(rospy.get_param("~capture_interval_sec", 1.0))
        self.batch_size = int(rospy.get_param("~batch_size", 10))
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )
        self.zmq_endpoint = rospy.get_param(
            "~zmq_endpoint", "ipc:///tmp/mapanything.ipc"
        )
        self.zmq_timeout_ms = int(rospy.get_param("~zmq_timeout_ms", 30000))

        self.collected_images: List[np.ndarray] = []
        self.last_capture_time: Optional[float] = None

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = None

        # Load camera intrinsics
        self.intrinsics = self.load_intrinsics()

        # Connect to inference worker
        self.connect_to_worker()

        # Subscribe to camera images
        self.image_subscriber = rospy.Subscriber(
            "camera/image_raw",
            Image,
            self.image_callback,
            queue_size=1,
        )

        # Publisher for results
        self.publisher = rospy.Publisher("map_anything/result", Bool, queue_size=10)

        rospy.loginfo(
            "MapAnythingROSNode: Started (capture interval %.2fs, batch size %d, endpoint %s).",
            self.capture_interval_sec,
            self.batch_size,
            self.zmq_endpoint,
        )

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def load_intrinsics(self) -> np.ndarray:
        calibration_fetcher = camera_calibrations.CameraCalibrationFetcher(
            self.camera_namespace, True
        )
        camera_info = calibration_fetcher.get_camera_info()

        if camera_info is None:
            raise RuntimeError(
                f"MapAnythingROSNode: Failed to fetch camera intrinsics from {self.camera_namespace}"
            )

        intrinsics = np.array(camera_info.K, dtype=np.float32).reshape(3, 3)
        rospy.loginfo(
            "MapAnythingROSNode: Loaded intrinsics for %s",
            self.camera_namespace,
        )
        return intrinsics

    def connect_to_worker(self):
        """Connect to the inference worker via ZeroMQ."""
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.zmq_timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.zmq_timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(self.zmq_endpoint)
            rospy.loginfo(
                "MapAnythingROSNode: Connected to inference worker at %s",
                self.zmq_endpoint,
            )
        except zmq.ZMQError as exc:
            raise RuntimeError(
                f"MapAnythingROSNode: Could not connect to inference worker at {self.zmq_endpoint}: {exc}"
            )

    def image_callback(self, msg: Image):
        if self.socket is None:
            rospy.logwarn_throttle(5.0, "MapAnythingROSNode: Not connected to worker.")
            return

        if not self.should_capture():
            return

        image_rgb = self.convert_image(msg)
        if image_rgb is None:
            return

        self.collected_images.append(image_rgb)
        self.last_capture_time = rospy.get_time()

        rospy.loginfo(
            "MapAnythingROSNode: Captured image %d/%d",
            len(self.collected_images),
            self.batch_size,
        )

        if len(self.collected_images) < self.batch_size:
            return

        # Send batch to inference worker
        success = self.send_batch_and_receive_results()

        if success:
            self.publisher.publish(Bool(data=True))
            rospy.loginfo(
                "MapAnythingROSNode: Successfully processed batch of %d images.",
                len(self.collected_images),
            )

        self.collected_images.clear()

    def convert_image(self, msg: Image) -> Optional[np.ndarray]:
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("MapAnythingROSNode: Failed to convert image: %s", exc)
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def send_batch_and_receive_results(self) -> bool:
        """Send batch of images with intrinsics to worker and receive results."""
        try:
            # Prepare data
            batch_data = {
                "images": self.collected_images,
                "intrinsics": self.intrinsics,
            }

            # Serialize with pickle
            serialized_data = pickle.dumps(batch_data)

            rospy.loginfo(
                "MapAnythingROSNode: Sending batch of %d images (%d bytes)",
                len(self.collected_images),
                len(serialized_data),
            )

            # Send request (ZeroMQ handles framing automatically)
            self.socket.send(serialized_data)

            # Receive response (with timeout)
            response_data = self.socket.recv()

            # Deserialize response
            response = pickle.loads(response_data)

            if response.get("status") == "success":
                rospy.loginfo("MapAnythingROSNode: Received predictions from worker")
                # TODO: Process predictions as needed
                # predictions = response.get("predictions")
                self.publisher.publish(Bool(data=True))
                return True
            else:
                error = response.get("error", "Unknown error")
                rospy.logerr("MapAnythingROSNode: Worker returned error: %s", error)
                return False

        except zmq.Again:
            rospy.logerr("MapAnythingROSNode: Request timeout (worker not responding)")
            # Reset socket after timeout (REQ/REP state machine requirement)
            self.socket.close()
            self.connect_to_worker()
            return False
        except zmq.ZMQError as exc:
            rospy.logerr("MapAnythingROSNode: ZMQ error: %s", exc)
            return False
        except Exception as exc:
            rospy.logerr("MapAnythingROSNode: Error during communication: %s", exc)
            return False

    def should_capture(self) -> bool:
        current_time = rospy.get_time()
        if self.last_capture_time is None:
            return True
        return (current_time - self.last_capture_time) >= self.capture_interval_sec

    def cleanup(self):
        """Clean up ZeroMQ resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        rospy.loginfo("MapAnythingROSNode: Cleaned up ZeroMQ resources")


if __name__ == "__main__":
    rospy.init_node("map_anything_ros_node")
    node = MapAnythingROSNode()

    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.cleanup()
