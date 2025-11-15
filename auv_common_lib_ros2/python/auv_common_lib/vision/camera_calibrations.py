import rospy
from sensor_msgs.msg import CameraInfo
import threading
import copy


class CameraCalibrationFetcher:
    def __init__(self, camera_namespace: str, wait_for_camera_info: bool = True):
        self.camera_info_topic = f"{camera_namespace}/camera_info"
        self.camera_info = None
        self.camera_info_lock = threading.Lock()
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)

        if wait_for_camera_info:
            rospy.wait_for_message(self.camera_info_topic, CameraInfo)

    def camera_info_callback(self, data):
        with self.camera_info_lock:
            self.camera_info = data

    def is_received(self) -> bool:
        return self.camera_info is not None

    def get_camera_info(self):
        if not self.is_received():
            rospy.logwarn(f"Camera info not received on {self.camera_info_topic}")
            return None

        with self.camera_info_lock:
            camera_info = copy.deepcopy(self.camera_info)

        return camera_info
