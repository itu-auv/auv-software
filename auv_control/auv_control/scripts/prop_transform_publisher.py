#!/usr/bin/env python3

import rospy
import math
import tf
import message_filters
from geometry_msgs.msg import (
    PoseStamped,
    PoseArray,
    Pose,
    TransformStamped,
    PointStamped,
)
from std_msgs.msg import Float32
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from ultralytics_ros.msg import YoloResult
from nav_msgs.msg import Odometry
import auv_common_lib.vision.camera_calibrations as camera_calibrations
from sensor_msgs.msg import Range
import tf2_ros
import tf2_geometry_msgs
import time
import copy
import threading


class Scene:
    def __init__(self):
        self.lock = threading.Lock()
        self.objects = {}

    def add_object_to_location(self, id: int, x: float, y: float, z: float):
        with self.lock:
            if id not in self.objects:
                self.objects[id] = []

            # check if the new location belongs to already existing object or a new object, just by checking the distance
            for obj in self.objects[id]:
                distance = math.sqrt(
                    (obj.x - x) ** 2 + (obj.y - y) ** 2 + (obj.z - z) ** 2
                )
                if distance < 4.0:
                    obj.update_position(x, y, z)
                    return

            # if it is octagon, dont add new octagon if there is already one
            if id == 14 and len(self.objects[id]) > 0:
                pass  # don't add new octagon
            else:
                new_object = SceneObject(id, x, y, z)
                self.objects[id].append(new_object)

    def get_objects(self):
        objects_copy = None
        with self.lock:
            objects_copy = copy.deepcopy(self.objects)

        return objects_copy

    def update_objects(self):
        with self.lock:
            for key, value in self.objects.items():
                for obj in value:
                    obj.update_filtered_position()

    def print_all_objects_and_their_count(self):
        with self.lock:
            for key, value in self.objects.items():
                print(f"Object {id_to_name_map[key]}: {len(value)}")


class SceneObject:
    def __init__(self, id: int, x: float, y: float, z: float):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.filtered_x = x
        self.filtered_y = y
        self.filtered_z = z

    def get_position(self):
        return self.x, self.y, self.z

    def update_filtered_position(self):
        alpha = 0.2
        self.filtered_x = alpha * self.x + (1 - alpha) * self.filtered_x
        self.filtered_y = alpha * self.y + (1 - alpha) * self.filtered_y
        self.filtered_z = alpha * self.z + (1 - alpha) * self.filtered_z

    def update_position(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


name_to_id_map = {
    "red_buoy": 8,
    "gate_red_arrow": 4,
    "gate_blue_arrow": 3,
    "gate_middle_part": 5,
    "torpedo_map": 12,
    "bin_whole": 9,
    "octagon": 14,
}

id_to_name_map = {
    8: "red_buoy",
    4: "gate_red_arrow",
    3: "gate_blue_arrow",
    5: "gate_middle_part",
    12: "torpedo_map",
    9: "bin_whole",
    14: "octagon",
}


class CameraCalibration:
    def __init__(self, namespace: str):
        self.calibration = camera_calibrations.CameraCalibrationFetcher(
            namespace, True
        ).get_camera_info()

    def calculate_angles(self, pixel_coordinates: tuple) -> tuple:
        fx = self.calibration.K[0]
        fy = self.calibration.K[4]
        cx = self.calibration.K[2]
        cy = self.calibration.K[5]
        norm_x = (pixel_coordinates[0] - cx) / fx
        norm_y = (pixel_coordinates[1] - cy) / fy
        angle_x = math.atan(norm_x)
        angle_y = math.atan(norm_y)
        return angle_x, angle_y

    def distance_from_height(self, real_height: float, measured_height: float) -> float:
        focal_length = self.calibration.K[4]
        distance = (real_height * focal_length) / measured_height
        return distance

    def distance_from_width(self, real_width: float, measured_width: float) -> float:
        focal_length = self.calibration.K[0]
        distance = (real_width * focal_length) / measured_width
        return distance


class Prop:
    def __init__(self, id: int, name: str, real_height: float, real_width: float):
        self.id = id
        self.name = name
        self.real_height = real_height
        self.real_width = real_width
        self.scene_objects = {}

    def estimate_distance(
        self,
        measured_height: float,
        measured_width: float,
        calibration: CameraCalibration,
    ):

        distance_from_height = None
        distance_from_width = None
        distance_avarage = None

        if self.real_height is not None:
            distance_from_height = calibration.distance_from_height(
                self.real_height, measured_height
            )

        if self.real_width is not None:
            distance_from_width = calibration.distance_from_width(
                self.real_width, measured_width
            )

        if distance_from_height is not None and distance_from_width is not None:
            distance_avarage = (distance_from_height + distance_from_width) * 0.5
            return distance_avarage
        elif distance_from_height is not None:
            return distance_from_height
        elif distance_from_width is not None:
            return distance_from_width
        else:
            rospy.logerr(f"Could not estimate distance for prop {self.name}")
            return None


class GateRedArrow(Prop):
    def __init__(self):
        super().__init__(4, "gate_red_arrow", 0.3048, 0.3048)


class GateBlueArrow(Prop):
    def __init__(self):
        super().__init__(3, "gate_blue_arrow", 0.3048, 0.3048)


class GateMiddlePart(Prop):
    def __init__(self):
        super().__init__(5, "gate_middle_part", 0.6096, None)


class BuoyRed(Prop):
    def __init__(self):
        super().__init__(8, "red_buoy", 0.292, 0.203)


class TorpedoMap(Prop):
    def __init__(self):
        super().__init__(12, "torpedo_map", 0.6096, 0.6096)


class BinWhole(Prop):
    def __init__(self):
        super().__init__(9, "bin_whole", None, None)


class Octagon(Prop):
    def __init__(self):
        super().__init__(14, "octagon", 0.92, 1.30)


class ObjectPositionEstimator:
    def __init__(self):
        rospy.init_node("object_position_estimator", anonymous=True)

        self.scene = Scene()

        self.camera_calibrations = {
            "taluy/cameras/cam_front": CameraCalibration("taluy/cameras/cam_front"),
            "taluy/cameras/cam_bottom": CameraCalibration("taluy/cameras/cam_bottom"),
        }

        self.camera_frames = {
            "taluy/cameras/cam_front": "taluy/base_link/front_camera_optical_link",
            "taluy/cameras/cam_bottom": "taluy/base_link/bottom_camera_optical_link",
        }

        self.props = {
            "red_buoy": BuoyRed(),
            "gate_red_arrow": GateRedArrow(),
            "gate_blue_arrow": GateBlueArrow(),
            "gate_middle_part": GateMiddlePart(),
            "torpedo_map": TorpedoMap(),
            # "bin_whole": BinWhole(),
            "octagon": Octagon(),
        }

        self.id_tf_map = {
            "taluy/cameras/cam_front": {
                8: "red_buoy",
                7: "path",
                9: "bin_whole",
                12: "torpedo_map",
                13: "torpedo_hole",
                1: "gate_left",
                2: "gate_right",
                3: "gate_blue_arrow",
                4: "gate_red_arrow",
                5: "gate_middle_part",
                14: "octagon",
            },
            "taluy/cameras/cam_bottom": {9: "bin/whole", 10: "bin/red", 11: "bin/blue"},
        }

        # self.real_height_map = {
        #     "red_buoy": 0.292,
        #     "gate_red_arrow": 0.3048,
        #     "gate_blue_arrow": 0.3048,
        #     "gate_middle_part": 0.6096,
        #     "torpedo_map": 0.6096,
        #     "octagon": 0.7,
        # }

        # self.real_width_map = {
        #     "red_buoy": 0.203,
        #     "gate_red_arrow": 0.3048,
        #     "gate_blue_arrow": 0.3048,
        #     "torpedo_map": 0.6096,
        #     "octagon": 1.2,
        # }

        # Initialize TransformBroadcaster
        self.broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize PoseArray publisher
        self.detection_line_pubs = {
            x: rospy.Publisher(
                f"/taluy/missions/{x}/detection_lines", PoseArray, queue_size=10
            )
            for x in self.id_tf_map["taluy/cameras/cam_front"].values()
        }

        # create an area publisher for each detection
        self.detection_distance_pubs = {
            x: rospy.Publisher(
                f"/taluy/missions/{x}/detection_distance", Float32, queue_size=10
            )
            for x in self.id_tf_map["taluy/cameras/cam_front"].values()
        }

        self.pose_array_pub = rospy.Publisher("/line_topic", PoseArray, queue_size=10)

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
        yolo_result_subscriber = message_filters.Subscriber("/yolo_result", YoloResult)
        altitude_subscriber = message_filters.Subscriber(
            "/taluy/sensors/dvl/altitude", Float32
        )
        front_sonar_range_subscriber = message_filters.Subscriber(
            "/taluy/sensors/sonar_front/range", Range
        )

        # altitude callback from dvl
        # self.dvl_callback = rospy.Subscriber("/taluy/sensors/dvl/altitude", Float32, self.dvl_callback)
        # self.latest_altitude = None

        ts = message_filters.ApproximateTimeSynchronizer(
            [yolo_result_subscriber, altitude_subscriber],
            10,
            0.5,
            allow_headerless=True,
        )
        ts.registerCallback(self.callback)
        rospy.loginfo("Object position estimator node initialized")

    def scene_transform_publisher_callback(self, event):
        # first update scene objects
        self.scene.update_objects()

        # traverse over all objects in the scene and publish their transform as tf
        for key, value in self.scene.get_objects().items():
            # find out the closest object to the taluy/base_link
            closest_object = None
            min_distance = 1000.0
            for obj in value:

                point1 = PointStamped()
                point1.header.frame_id = "odom"
                point1.point.x = obj.filtered_x
                point1.point.y = obj.filtered_y
                point1.point.z = obj.filtered_z
                transform = None
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "taluy/base_link", "odom", rospy.Time(0), rospy.Duration(1.0)
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    rospy.logerr("Error looking up transform")
                    return

                point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)

                distance = math.sqrt(
                    point1_odom.point.x**2
                    + point1_odom.point.y**2
                    + point1_odom.point.z**2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj

            for obj in value:
                transform_message = TransformStamped()
                transform_message.header.stamp = rospy.Time.now()
                transform_message.header.frame_id = "odom"
                if obj == closest_object:
                    transform_message.child_frame_id = f"{id_to_name_map[key]}_link"
                else:
                    transform_message.child_frame_id = (
                        f"{id_to_name_map[key]}_{value.index(obj)}_link"
                    )
                transform_message.transform.translation.x = obj.filtered_x
                transform_message.transform.translation.y = obj.filtered_y
                transform_message.transform.translation.z = obj.filtered_z
                transform_message.transform.rotation.x = 0.0
                transform_message.transform.rotation.y = 0.0
                transform_message.transform.rotation.z = 0.0
                transform_message.transform.rotation.w = 1.0

                self.broadcaster.sendTransform(transform_message)

    # def dvl_callback(self, msg: Float32):
    #     self.latest_altitude = msg.data

    # def distance_from_height(self, real_height: float, measured_height: float) -> float:
    #     # get the camera calibration y fov
    #     calibration = self.camera_calibrations["taluy/cameras/cam_front"].calibration
    #     focal_length = calibration.K[4]  # fy from camera calibration
    #     distance = (real_height * focal_length) / measured_height
    #     return distance

    # def distance_from_width(self, real_width: float, measured_width: float) -> float:
    #     # get the camera calibration x fov
    #     calibration = self.camera_calibrations["taluy/cameras/cam_front"].calibration
    #     focal_length = calibration.K[0]
    #     distance = (real_width * focal_length) / measured_width
    #     return distance

    def check_if_detection_is_inside_image(
        self, detection, image_width: int = 640, image_height: int = 480
    ) -> bool:
        center = detection.bbox.center
        half_size_x = detection.bbox.size_x * 0.5
        half_size_y = detection.bbox.size_y * 0.5
        deadzone = 5  # pixels
        if (
            center.x + half_size_x >= image_width - deadzone
            or center.x - half_size_x <= deadzone
        ):
            return False
        if (
            center.y + half_size_y >= image_height - deadzone
            or center.y - half_size_y <= deadzone
        ):
            return False
        return True

    def callback(self, detection_msg: YoloResult, altitude_msg: Float32):
        print("Received messages")
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue

            altitude = altitude_msg.data

            detection_id = detection.results[0].id

            if detection_id in self.id_tf_map["taluy/cameras/cam_front"]:
                detection_name = self.id_tf_map["taluy/cameras/cam_front"][detection_id]

                if detection_id == 9:
                    self.process_altitude_projection(detection, altitude)
                    continue

                if detection_name in self.props:
                    prop = self.props[detection_name]
                    measured_height = detection.bbox.size_y
                    measured_width = detection.bbox.size_x
                    if self.check_if_detection_is_inside_image(detection) == False:
                        continue
                    distance = prop.estimate_distance(
                        measured_height,
                        measured_width,
                        self.camera_calibrations["taluy/cameras/cam_front"],
                    )
                    if distance is not None:
                        detection_distance = Float32()
                        detection_distance.data = distance
                        self.detection_distance_pubs[detection_name].publish(
                            detection_distance
                        )
                        self.process_front_estimated_camera(
                            detection, distance, suffix="average"
                        )

            # if detection_id in self.id_tf_map["taluy/cameras/cam_front"]:

            #     detection_name = self.id_tf_map["taluy/cameras/cam_front"][detection_id]
            #     if detection_name in self.real_height_map and detection_name in self.real_width_map:
            #         real_height = self.real_height_map[detection_name]
            #         real_width = self.real_width_map[detection_name]
            #         measured_height = detection.bbox.size_y
            #         measured_width = detection.bbox.size_x
            #         if self.check_if_detection_is_inside_image(detection) == False:
            #             continue
            #         distance_from_height = self.distance_from_height(
            #             real_height, measured_height)
            #         distance_from_width = self.distance_from_width(
            #             real_width, measured_width)
            #         distance_avarage = (
            #             distance_from_height + distance_from_width) * 0.5

            #         detection_distance = Float32()
            #         detection_distance.data = distance_avarage
            #         self.detection_distance_pubs[detection_name].publish(
            #             detection_distance)

            #         # self.process_front_estimated_camera(detection, distance_from_height, suffix="from_height")
            #         # self.process_front_estimated_camera(detection, distance_from_width, suffix="from_width")
            #         self.process_front_estimated_camera(
            #             detection, distance_avarage, suffix="average")

            # if id is bin whole
            # if detection_id == 9:
            #     self.process_altitude_projection(detection, altitude)

            # self.process_front_camera(detection, 15.0)

            # if detection_id in self.id_tf_map["taluy/cameras/cam_bottom"] and altitude is not None:
            #     self.process_bottom_camera(detection, altitude)

    def transform_pose_to_odom(self, pose: Pose, source_frame: str) -> Pose:
        # Transform pose to odom frame
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = source_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", source_frame, rospy.Time(0), rospy.Duration(1000000.0)
            )
            transformed_pose_stamped = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, transform
            )
            return transformed_pose_stamped.pose
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Error transforming pose: {e}")
            return None

    def process_front_camera(self, detection, distance: float):
        camera_name = "taluy/cameras/cam_front"
        detection_id = detection.results[0].id
        detection_name = self.id_tf_map[camera_name][detection_id]

        angle_x, angle_y = self.camera_calibrations[camera_name].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        # angle_x, angle_y = calculate_angles(
        #     self.camera_calibrations[camera_name].calibration,
        #     (detection.bbox.center.x, detection.bbox.center.y),
        # )

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
            start_pose, self.camera_frames[camera_name]
        )
        end_pose_odom = self.transform_pose_to_odom(
            end_pose, self.camera_frames[camera_name]
        )

        if start_pose_odom and end_pose_odom:
            pose_array.poses.append(start_pose_odom)
            pose_array.poses.append(end_pose_odom)

            # Publish PoseArray
            self.detection_line_pubs[detection_name].publish(pose_array)

    # def send_transform(self, transform: TransformStamped):
    #     req = SetObjectTransformRequest()
    #     req.transform = transform
    #     resp = self.set_object_transform_service.call(req)
    #     if not resp.success:
    #         rospy.logerr(
    #             f"Failed to set object transform, reason: {resp.message}")

    def process_bottom_camera(self, detection, distance: float):
        camera_name = "taluy/cameras/cam_bottom"
        detection_id = detection.results[0].id

        angle_x, angle_y = self.camera_calibrations[camera_name].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        # angle_x, angle_y = calculate_angles(
        #     self.camera_calibrations[camera_name].calibration,
        #     (detection.bbox.center.x, detection.bbox.center.y),
        # )

        # Calculate the offset in the bottom_camera_optical_link frame
        offset_x = math.tan(angle_x) * distance * -1.0
        offset_y = math.tan(angle_y) * distance * -1.0

        transform_message = TransformStamped()
        transform_message.header.stamp = rospy.Time.now()
        transform_message.header.frame_id = self.camera_frames[camera_name]
        transform_message.child_frame_id = (
            f"{self.id_tf_map[camera_name][detection_id]}_link"
        )
        transform_message.transform.translation.x = offset_x
        transform_message.transform.translation.y = offset_y
        transform_message.transform.translation.z = distance
        transform_message.transform.rotation.x = 0.0
        transform_message.transform.rotation.y = 0.0
        transform_message.transform.rotation.z = 0.0
        transform_message.transform.rotation.w = 1.0

        self.send_transform(transform_message)

    def process_front_estimated_camera(
        self, detection, distance: float, suffix: str = ""
    ):
        camera_name = "taluy/cameras/cam_front"
        detection_id = detection.results[0].id

        angle_x, angle_y = self.camera_calibrations[camera_name].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        # angle_x, angle_y = calculate_angles(
        #     self.camera_calibrations[camera_name].calibration,
        #     (detection.bbox.center.x, detection.bbox.center.y),
        # )

        # Calculate the offset in the bottom_camera_optical_link frame
        offset_x = math.tan(angle_x) * distance * 1.0
        offset_y = math.tan(angle_y) * distance * 1.0

        point1 = PointStamped()
        point1.header.frame_id = "taluy/base_link/front_camera_optical_link"
        point1.point.x = offset_x
        point1.point.y = offset_y
        point1.point.z = distance

        try:
            # transform from optical_camera_frame to odom frame
            transform = self.tf_buffer.lookup_transform(
                "odom",
                "taluy/base_link/front_camera_optical_link",
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)
            self.scene.add_object_to_location(
                detection_id,
                point1_odom.point.x,
                point1_odom.point.y,
                point1_odom.point.z,
            )
            self.scene.print_all_objects_and_their_count()

        except tf2_ros.LookupException as e:
            rospy.logerr(f"Error transforming point: {e}")

        transform_message = TransformStamped()
        transform_message.header.stamp = rospy.Time.now()
        transform_message.header.frame_id = self.camera_frames[camera_name]
        transform_message.child_frame_id = (
            f"{self.id_tf_map[camera_name][detection_id]}_link"
        )
        transform_message.transform.translation.x = offset_x
        transform_message.transform.translation.y = offset_y
        transform_message.transform.translation.z = distance
        transform_message.transform.rotation.x = 0.0
        transform_message.transform.rotation.y = 0.0
        transform_message.transform.rotation.z = 0.0
        transform_message.transform.rotation.w = 1.0

        # self.send_transform(transform_message)

    def calculate_intersection_with_ground(self, point1_odom, point2_odom):
        # Calculate t where the z component is zero (ground plane)
        if point2_odom.point.z != point1_odom.point.z:
            t = -point1_odom.point.z / (point2_odom.point.z - point1_odom.point.z)

            # Check if t is within the segment range [0, 1]
            if 0 <= t <= 1:
                # Calculate intersection point
                x = point1_odom.point.x + t * (
                    point2_odom.point.x - point1_odom.point.x
                )
                y = point1_odom.point.y + t * (
                    point2_odom.point.y - point1_odom.point.y
                )
                z = 0  # ground plane
                return x, y, z
            else:
                rospy.logwarn("No intersection with ground plane within the segment.")
                return None
        else:
            rospy.logwarn("The line segment is parallel to the ground plane.")
            return None

    def process_altitude_projection(self, detection, altitude: float):
        camera_name = "taluy/cameras/cam_front"
        detection_id = detection.results[0].id

        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5

        angle_x, angle_y = self.camera_calibrations[camera_name].calculate_angles(
            (bbox_bottom_x, bbox_bottom_y)
        )

        # angle_x, angle_y = calculate_angles(
        #     self.camera_calibrations[camera_name].calibration,
        #     (bbox_bottom_x, bbox_bottom_y),
        # )

        distance = 500.0

        offset_x = math.tan(angle_x) * distance * 1.0
        offset_y = math.tan(angle_y) * distance * 1.0

        # optical_camera_frame'de tanımlı iki nokta
        point1 = PointStamped()
        point1.header.frame_id = "taluy/base_link/front_camera_optical_link"
        point1.point.x = 0
        point1.point.y = 0
        point1.point.z = 0

        point2 = PointStamped()
        point2.header.frame_id = "taluy/base_link/front_camera_optical_link"
        point2.point.x = offset_x
        point2.point.y = offset_y
        point2.point.z = distance

        try:
            # optical_camera_frame'den odom frame'ine transform
            transform = self.tf_buffer.lookup_transform(
                "odom",
                "taluy/base_link/front_camera_optical_link",
                rospy.Time(0),
                rospy.Duration(1000000.0),
            )
            point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)
            point2_odom = tf2_geometry_msgs.do_transform_point(point2, transform)
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Error transforming point: {e}")

        point1_odom.point.z += altitude + 0.18
        point2_odom.point.z += altitude + 0.18

        # find the intersection of the line from point1 to point2 to the plane z=0
        # find the xyz here, where z=0
        intersect_return = self.calculate_intersection_with_ground(
            point1_odom, point2_odom
        )

        if not intersect_return:
            return

        x_, y_, z_ = intersect_return

        self.scene.add_object_to_location(detection_id, x_, y_, z_)

        # transform_message = TransformStamped()
        # transform_message.header.stamp = rospy.Time.now()
        # transform_message.header.frame_id = "odom"
        # transform_message.child_frame_id = f"{self.id_tf_map[camera_name][detection_id]}_link"
        # transform_message.transform.translation.x = x_
        # transform_message.transform.translation.y = y_
        # transform_message.transform.translation.z = z_
        # transform_message.transform.rotation.x = 0.0
        # transform_message.transform.rotation.y = 0.0
        # transform_message.transform.rotation.z = 0.0
        # transform_message.transform.rotation.w = 1.0
        # self.send_transform(transform_message)

        # transform_message = TransformStamped()
        # transform_message.header.stamp = rospy.Time.now()
        # transform_message.header.frame_id = self.camera_frames[camera_name]
        # transform_message.child_frame_id = f"{self.id_tf_map[camera_name][detection_id]}_altitude_link"
        # transform_message.transform.translation.x = offset_x
        # transform_message.transform.translation.y = offset_y
        # transform_message.transform.translation.z = distance
        # transform_message.transform.rotation.x = 0.0
        # transform_message.transform.rotation.y = 0.0
        # transform_message.transform.rotation.z = 0.0
        # transform_message.transform.rotation.w = 1.0

    def run(self):
        self.rate = rospy.Rate(30.0)
        while rospy.is_shutdown() is False:
            self.scene_transform_publisher_callback(None)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = ObjectPositionEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
