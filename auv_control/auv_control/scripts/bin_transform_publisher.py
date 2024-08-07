#!/usr/bin/env python3

import rospy
import math
import tf
import message_filters
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
from std_msgs.msg import Float32
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from ultralytics_ros.msg import YoloResult
from nav_msgs.msg import Odometry
import auv_common_lib.vision.camera_calibrations as camera_calibrations
from sensor_msgs.msg import Range
import tf2_ros
import tf2_geometry_msgs
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
            "taluy/cameras/cam_front": {8: "red_buoy", 7: "path", 9: "bin_whole", 12: "torpedo_map", 13: "torpedo_hole"},
            "taluy/cameras/cam_bottom": {9: "bin/whole", 10: "bin/red", 11: "bin/blue"},
        }

        # Initialize TransformBroadcaster
        self.broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize PoseArray publisher
        self.detection_line_pubs = {
            x: rospy.Publisher(
                f"/taluy/missions/{x}/detection_lines", PoseArray, queue_size=10)
            for x in self.id_tf_map["taluy/cameras/cam_front"].values()
        }

        self.pose_array_pub = rospy.Publisher(
            '/line_topic', PoseArray, queue_size=10)

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Services
        rospy.loginfo("Waiting for set_object_transform service...")
        self.set_object_transform_service = rospy.ServiceProxy(
            "/taluy/map/set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        # Subscriptions
        yolo_result_subscriber = message_filters.Subscriber(
            "/yolo_result", YoloResult)
        altitude_subscriber = message_filters.Subscriber(
            "/taluy/sensors/dvl/altitude", Float32
        )
        front_sonar_range_subscriber = message_filters.Subscriber(
            "/taluy/sensors/sonar_front/range", Range
        )
        ts = message_filters.ApproximateTimeSynchronizer(
            [yolo_result_subscriber, altitude_subscriber,
                front_sonar_range_subscriber],
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
                self.process_front_camera(detection, 15.0)

            if detection_id in self.id_tf_map["taluy/cameras/cam_bottom"] and altitude is not None:
                self.process_bottom_camera(detection, altitude)

    def transform_pose_to_odom(self, pose: Pose, source_frame: str) -> Pose:
        # Transform pose to odom frame
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = source_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", source_frame, rospy.Time(0), rospy.Duration(1.0))
            transformed_pose_stamped = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, transform)
            return transformed_pose_stamped.pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Error transforming pose: {e}")
            return None

    def process_front_camera(self, detection, distance: float):
        camera_name = "taluy/cameras/cam_front"
        detection_id = detection.results[0].id
        detection_name = self.id_tf_map[camera_name][detection_id]

        angle_x, angle_y = calculate_angles(
            self.calibrations[camera_name],
            (detection.bbox.center.x, detection.bbox.center.y),
        )

        offset_x = math.tan(angle_x) * distance * 1.0
        offset_y = math.tan(angle_y) * distance * 1.0

        time_ = int(time.time())

        print("Time: ", time_)

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "odom"

        # Start pose in the camera frame
        start_pose = Pose()
        start_pose.position.x = 0
        start_pose.position.y = 0
        start_pose.position.z = 0.0
        start_pose.orientation.x = 0.0
        start_pose.orientation.y = 0.0
        start_pose.orientation.z = 0.0
        start_pose.orientation.w = 1.0

        # End pose in the camera frame
        end_pose = Pose()
        end_pose.position.x = offset_x
        end_pose.position.y = offset_y
        end_pose.position.z = distance
        end_pose.orientation.x = 0.0
        end_pose.orientation.y = 0.0
        end_pose.orientation.z = 0.0
        end_pose.orientation.w = 1.0

        # Transform poses to the odom frame
        start_pose_odom = self.transform_pose_to_odom(
            start_pose, self.camera_frames[camera_name])
        end_pose_odom = self.transform_pose_to_odom(
            end_pose, self.camera_frames[camera_name])

        if start_pose_odom and end_pose_odom:
            pose_array.poses.append(start_pose_odom)
            pose_array.poses.append(end_pose_odom)

            # Publish PoseArray
            self.detection_line_pubs[detection_name].publish(pose_array)

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
