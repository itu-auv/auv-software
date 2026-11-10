#!/usr/bin/env python3

import rclpy
from auv_common_lib import ros
from auv_common_lib.ros import get_param
from rclpy.node import Node
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time


class ImageSaver(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__("image_saver")

        # Parameters
        self.namespace = get_param(self, "~namespace", "taluy")
        self.camera = get_param(
            self, "~camera", "front"
        )  # front|bottom|torpedo|realsense
        self.images_per_folder = int(get_param(self, "~images_per_folder", 150))
        self.save_interval = float(get_param(self, "~interval_seconds", 1.0))

        # Save path: prefer explicit ~save_path, else build from ~base_dir + ~folder_name
        save_path_param = get_param(self, "~save_path", None)
        if save_path_param:
            self.base_path = os.path.expanduser(save_path_param)
        else:
            base_dir = os.path.expanduser(get_param(self, "~base_dir", "~/Desktop"))
            folder_name = get_param(self, "~folder_name", "yolo_bag")
            self.base_path = os.path.join(base_dir, folder_name)

        # Derive image topic from camera + namespace
        self.topic_name = self.resolve_image_topic(self.camera, self.namespace)

        # State
        self.bridge = CvBridge()
        self.image_count = 0
        self.folder_count = 1
        self.current_folder = ""
        self.last_save_time = 0.0
        self.latest_image = None

        # Prepare directories
        self.create_base_directory()
        self.create_new_folder()

        # ROS subscriber
        self.create_subscription(Image, self.topic_name, self.image_callback, 10)

        # Timer: check frequently and save according to interval
        self.create_timer(0.1, self.save_timer_callback)

        self.get_logger().info("Image saver node started.")
        self.get_logger().info(f"Namespace: {self.namespace}")
        self.get_logger().info(f"Camera: {self.camera}")
        self.get_logger().info(f"Topic: {self.topic_name}")
        self.get_logger().info(f"Save path: {self.base_path}")
        self.get_logger().info(
            f"Create a new folder after {self.images_per_folder} images"
        )
        self.get_logger().info(f"Save interval: {self.save_interval:.2f} seconds")

    @staticmethod
    def resolve_image_topic(camera: str, namespace: str) -> str:
        """Map camera name to image topic based on namespace."""
        cam = (camera or "").strip().lower()
        ns = ("/" + namespace.strip("/")).strip()

        mapping = {
            "front": f"{ns}/cameras/cam_front/image_rect_color",
            "bottom": f"{ns}/cameras/cam_bottom/image_rect_color",
            "torpedo": f"{ns}/cameras/cam_torpedo/image_rect_color",
            "realsense": f"{ns}/camera/color/image_raw",
        }

        if cam not in mapping:
            rclpy.logging.get_logger("image_saver").warning(
                f"Unknown camera '{camera}'. Falling back to 'front'."
            )
            cam = "front"
        return mapping[cam]

    def create_base_directory(self):
        """Create base output directory if missing."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            self.get_logger().info(f"Created base directory: {self.base_path}")

    def create_new_folder(self):
        """Create a new subfolder for outputs."""
        self.current_folder = os.path.join(self.base_path, f"output{self.folder_count}")
        if not os.path.exists(self.current_folder):
            os.makedirs(self.current_folder)
            self.get_logger().info(f"Created folder: output{self.folder_count}")

    def image_callback(self, msg: Image):
        """Receive ROS Image and convert to OpenCV format."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def save_timer_callback(self):
        """Periodic timer callback to save images at requested interval."""
        current_time = time.time()
        if (
            self.latest_image is not None
            and (current_time - self.last_save_time) >= self.save_interval
        ):
            self.save_image()
            self.last_save_time = current_time

    def save_image(self):
        """Save the latest image to disk."""
        try:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            filename = f"image_{self.image_count + 1}_{timestamp:.3f}.jpg"
            filepath = os.path.join(self.current_folder, filename)

            cv2.imwrite(filepath, self.latest_image)
            self.image_count += 1

            self.get_logger().info(
                f"Saved image: {filename} (Total: {self.image_count})"
            )

            if self.image_count % self.images_per_folder == 0:
                self.folder_count += 1
                self.create_new_folder()
                self.get_logger().info(
                    f"{self.images_per_folder} images saved. Switching to folder: output{self.folder_count}"
                )

        except Exception as e:
            self.get_logger().error(f"Image save error: {e}")

    def run(self):
        """Spin ROS node."""
        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            self.get_logger().info("Stopping image saver node...")
            self.get_logger().info(f"Total images saved: {self.image_count}")


def main():
    ros.init()
    node = ImageSaver()
    ros.spin(node, node.run)


if __name__ == "__main__":
    main()
