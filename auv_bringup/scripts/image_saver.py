#!/usr/bin/env python3

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse
from datetime import datetime
import time


class ImageSaver:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("image_saver", anonymous=True)

        # Parameters
        self.namespace = rospy.get_param("~namespace", "taluy")
        self.camera = rospy.get_param(
            "~camera", "front"
        )  # front|bottom|torpedo|realsense
        self.images_per_folder = int(rospy.get_param("~images_per_folder", 150))
        self.save_interval = float(rospy.get_param("~interval_seconds", 1.0))
        self.continuous = bool(rospy.get_param("~continuous", True))

        # Save path: prefer explicit ~save_path, else build from ~base_dir + ~folder_name
        save_path_param = rospy.get_param("~save_path", None)
        if save_path_param:
            self.base_path = os.path.expanduser(save_path_param)
        else:
            base_dir = os.path.expanduser(rospy.get_param("~base_dir", "~/Desktop"))
            folder_name = rospy.get_param("~folder_name", "yolo_bag")
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
        rospy.Subscriber(self.topic_name, Image, self.image_callback)
        rospy.Service("~capture", Trigger, self.capture_image)

        # Timer: check frequently and save according to interval
        if self.continuous:
            rospy.Timer(rospy.Duration(0.1), self.save_timer_callback)

        rospy.loginfo("Image saver node started.")
        rospy.loginfo(f"Namespace: {self.namespace}")
        rospy.loginfo(f"Camera: {self.camera}")
        rospy.loginfo(f"Topic: {self.topic_name}")
        rospy.loginfo(f"Save path: {self.base_path}")
        rospy.loginfo(f"Create a new folder after {self.images_per_folder} images")
        rospy.loginfo(f"Save interval: {self.save_interval:.2f} seconds")

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
            rospy.logwarn(f"Unknown camera '{camera}'. Falling back to 'front'.")
            cam = "front"
        return mapping[cam]

    def create_base_directory(self):
        """Create base output directory if missing."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            rospy.loginfo(f"Created base directory: {self.base_path}")

    def create_new_folder(self):
        """Create a new subfolder for outputs."""
        self.current_folder = os.path.join(self.base_path, f"output{self.folder_count}")
        if not os.path.exists(self.current_folder):
            os.makedirs(self.current_folder)
            rospy.loginfo(f"Created folder: output{self.folder_count}")

    def image_callback(self, msg: Image):
        """Receive ROS Image and convert to OpenCV format."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")

    def save_timer_callback(self, _):
        """Periodic timer callback to save images at requested interval."""
        current_time = time.time()
        if (
            self.latest_image is not None
            and (current_time - self.last_save_time) >= self.save_interval
        ):
            self.save_image()
            self.last_save_time = current_time

    def capture_image(self, _request):
        """Save one current frame when a waypoint has camera enabled."""
        if self.latest_image is None:
            return TriggerResponse(False, "No camera image has been received yet.")
        waypoint_folder = rospy.get_param("~waypoint_folder", "manual_capture")
        waypoint_folder = os.path.basename(str(waypoint_folder).strip())
        if not waypoint_folder:
            waypoint_folder = "manual_capture"
        capture_folder = os.path.join(self.base_path, waypoint_folder)
        os.makedirs(capture_folder, exist_ok=True)

        if not self.save_image(capture_folder):
            return TriggerResponse(False, "Could not save the camera image.")
        return TriggerResponse(True, "One image saved.")

    def save_image(self, folder=None):
        """Save the latest image to disk."""
        try:
            timestamp = rospy.Time.now().to_sec()
            filename = f"image_{self.image_count + 1}_{timestamp:.3f}.jpg"
            filepath = os.path.join(folder or self.current_folder, filename)
            image = self.latest_image.copy()
            capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                image,
                capture_time,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                capture_time,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if not cv2.imwrite(filepath, image):
                rospy.logerr("Could not save image: %s", filepath)
                return False

            self.image_count += 1
            rospy.loginfo(f"Saved image: {filename} (Total: {self.image_count})")
            if self.image_count % self.images_per_folder == 0:
                self.folder_count += 1
                self.create_new_folder()
                rospy.loginfo(
                    f"{self.images_per_folder} images saved. Switching to folder: output{self.folder_count}"
                )
            return True
        except Exception as e:
            rospy.logerr(f"Image save error: {e}")
            return False

    def run(self):
        """Spin ROS node."""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Stopping image saver node...")
            rospy.loginfo(f"Total images saved: {self.image_count}")


if __name__ == "__main__":
    try:
        image_saver = ImageSaver()
        image_saver.run()
    except rospy.ROSInterruptException:
        pass
