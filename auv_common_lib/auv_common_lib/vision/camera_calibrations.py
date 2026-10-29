from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
import threading
import copy


class CameraCalibrationFetcher:
    def __init__(self, node, camera_namespace: str, wait_for_camera_info: bool = True):
        self.node = node
        self.camera_info_topic = f"{camera_namespace}/camera_info"
        self.camera_info = None
        self.camera_info_lock = threading.Lock()
        self.subscription = node.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        if wait_for_camera_info:
            wait_for_message(CameraInfo, node, self.camera_info_topic)

    def camera_info_callback(self, data):
        with self.camera_info_lock:
            self.camera_info = data

    def is_received(self) -> bool:
        return self.camera_info is not None

    def get_camera_info(self):
        if not self.is_received():
            self.node.get_logger().warning(f"Camera info not received on {self.camera_info_topic}")
            return None

        with self.camera_info_lock:
            camera_info = copy.deepcopy(self.camera_info)

        return camera_info
