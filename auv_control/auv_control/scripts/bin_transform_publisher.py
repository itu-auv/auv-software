#!/usr/bin/env python3

import rospy
import math
import tf
import message_filters
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from ultralytics_ros.msg import YoloResult
from nav_msgs.msg import Odometry
import auv_common_lib.vision.camera_calibrations as camera_calibrations
from sensor_msgs.msg import Range
import tf2_ros
import time


def create_calibration(namespaces: list) -> dict:
    calibrations = {}

    for namespace in namespaces:
        camera_calibration = camera_calibrations.CameraCalibrationFetcher(
            namespace, True
        )
        calibrations[namespace] = camera_calibration.get_camera_info()

    return calibrations


def calculate_angles(camera_info, pixel_coordinates: tuple) -> tuple:
    # Extract intrinsic parameters
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    # Normalize pixel coordinates
    norm_x = (pixel_coordinates[0] - cx) / fx
    norm_y = (pixel_coordinates[1] - cy) / fy

    # Calculate angles
    angle_x = math.atan(norm_x)
    angle_y = math.atan(norm_y)

    return angle_x, angle_y


class ObjectPositionEstimator:
    def __init__(self):
        rospy.init_node("object_position_estimator", anonymous=True)

        self.calibrations = create_calibration(
            ["taluy/cameras/cam_front", "taluy/cameras/cam_bottom"]
        )

        self.camera_frames = {
            "taluy/cameras/cam_front": "taluy/base_link/front_camera_optical_link",
            "taluy/cameras/cam_bottom": "taluy/base_link/bottom_camera_optical_link",
        }

        self.id_tf_map = {
            "taluy/cameras/cam_front": {8: "buoy"},
            "taluy/cameras/cam_bottom": {9: "bin/whole", 10: "bin/red", 11: "bin/blue"},
        }
        
        # Initialize TransformBroadcaster
        self.broadcaster = tf2_ros.TransformBroadcaster()

        # Services
        rospy.loginfo("Waiting for set_object_transform service...")
        self.set_object_transform_service = rospy.ServiceProxy(
            "/taluy/map/set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        # Subscriptions
        yolo_result_subscriber = message_filters.Subscriber("/yolo_result", YoloResult)
        altitude_subscriber = message_filters.Subscriber(
            "/taluy/sensors/dvl/altitude", Float32
        )
        front_sonar_range_subscriber = message_filters.Subscriber(
            "/taluy/sensors/sonar_front/range", Range
        )
        ts = message_filters.ApproximateTimeSynchronizer(
            [yolo_result_subscriber, altitude_subscriber, front_sonar_range_subscriber],
            10,
            0.5,
            allow_headerless=True,
        )
        ts.registerCallback(self.callback)
        rospy.loginfo("Object position estimator node initialized")

    def callback(
        self, detection_msg: YoloResult, altitude_msg: Float32, sonar_msg: Range
    ):
        print("Received messages")
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue

            sonar_distance = sonar_msg.range
            altitude = altitude_msg.data

            detection_id = detection.results[0].id

            if detection_id in self.id_tf_map["taluy/cameras/cam_front"]:
                self.process_front_camera(detection, sonar_distance)

            if detection_id in self.id_tf_map["taluy/cameras/cam_bottom"]:
                self.process_bottom_camera(detection, altitude)

    def send_transform(self, transform: TransformStamped):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(f"Failed to set object transform, reason: {resp.message}")

    def process_front_camera(self, detection, distance: float):
        camera_name = "taluy/cameras/cam_front"
        detection_id = detection.results[0].id

        angle_x, angle_y = calculate_angles(
            self.calibrations[camera_name],
            (detection.bbox.center.x, detection.bbox.center.y),
        )

        offset_x = math.tan(angle_x) * 100.0 * 1.0
        offset_y = math.tan(angle_y) * 100.0 * 1.0

        time_ = int(time.time())

        transform_message = TransformStamped()
        transform_message.header.stamp = rospy.Time.now()
        transform_message.header.frame_id = self.camera_frames[camera_name]
        transform_message.child_frame_id = f"{self.id_tf_map[camera_name][detection_id]}_link/start{time_}"
        transform_message.transform.translation.x = 0
        transform_message.transform.translation.y = 0
        transform_message.transform.translation.z = 0
        transform_message.transform.rotation.x = 0.0
        transform_message.transform.rotation.y = 0.0
        transform_message.transform.rotation.z = 0.0
        transform_message.transform.rotation.w = 1.0

        self.send_transform(transform_message)

        transform_message = TransformStamped()
        transform_message.header.stamp = rospy.Time.now()
        transform_message.header.frame_id = f"{self.id_tf_map[camera_name][detection_id]}_link/start{time_}" # self.camera_frames[camera_name]
        transform_message.child_frame_id = f"{self.id_tf_map[camera_name][detection_id]}_link/end{time_}"
        transform_message.transform.translation.x = offset_x
        transform_message.transform.translation.y = offset_y
        transform_message.transform.translation.z = 100
        transform_message.transform.rotation.x = 0.0
        transform_message.transform.rotation.y = 0.0
        transform_message.transform.rotation.z = 0.0
        transform_message.transform.rotation.w = 1.0

        self.broadcaster.sendTransform(transform_message)
        # self.send_transform(transform_message)

    def process_bottom_camera(self, detection, distance: float):
        camera_name = "taluy/cameras/cam_bottom"
        detection_id = detection.results[0].id

        angle_x, angle_y = calculate_angles(
            self.calibrations[camera_name],
            (detection.bbox.center.x, detection.bbox.center.y),
        )

        # Calculate the offset in the bottom_camera_optical_link frame
        offset_x = math.tan(angle_x) * distance * -1.0
        offset_y = math.tan(angle_y) * distance * -1.0

        transform_message = TransformStamped()
        transform_message.header.stamp = rospy.Time.now()
        transform_message.header.frame_id = self.camera_frames[camera_name]
        transform_message.child_frame_id = f"{self.id_tf_map[camera_name][detection_id]}_link"
        transform_message.transform.translation.x = offset_x
        transform_message.transform.translation.y = offset_y
        transform_message.transform.translation.z = distance
        transform_message.transform.rotation.x = 0.0
        transform_message.transform.rotation.y = 0.0
        transform_message.transform.rotation.z = 0.0
        transform_message.transform.rotation.w = 1.0

        self.send_transform(transform_message)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = ObjectPositionEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
