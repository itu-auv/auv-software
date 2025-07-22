#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import (
    PointStamped,
    PoseArray,
    PoseStamped,
    Pose,
    TransformStamped,
    Transform,
    Vector3,
    Quaternion,
)
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from sensor_msgs.msg import Range
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetDetectionFocus, SetDetectionFocusResponse
import auv_common_lib.vision.camera_calibrations as camera_calibrations
import tf2_ros
import tf2_geometry_msgs


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

    def estimate_distance(
        self,
        measured_height: float,
        measured_width: float,
        calibration: CameraCalibration,
    ):
        distance_from_height = None
        distance_from_width = None

        if self.real_height is not None:
            distance_from_height = calibration.distance_from_height(
                self.real_height, measured_height
            )

        if self.real_width is not None:
            distance_from_width = calibration.distance_from_width(
                self.real_width, measured_width
            )

        if distance_from_height is not None and distance_from_width is not None:
            return (distance_from_height + distance_from_width) * 0.5
        elif distance_from_height is not None:
            return distance_from_height
        elif distance_from_width is not None:
            return distance_from_width
        else:
            rospy.logerr(f"Could not estimate distance for prop {self.name}")
            return None


class Sawfish(Prop):
    def __init__(self):
        super().__init__(0, "sawfish", 0.3048, 0.3048)


class Shark(Prop):
    def __init__(self):
        super().__init__(1, "shark", 0.3048, 0.3048)


class RedPipe(Prop):
    def __init__(self):
        super().__init__(2, "red_pipe", 0.900, None)


class WhitePipe(Prop):
    def __init__(self):
        super().__init__(3, "white_pipe", 0.900, None)


class TorpedoMap(Prop):
    def __init__(self):
        super().__init__(4, "torpedo_map", 0.6096, 0.6096)


class TorpedoHole(Prop):
    def __init__(self):
        super().__init__(5, "torpedo_hole", 0.125, 0.125)


class BinWhole(Prop):
    def __init__(self):
        super().__init__(6, "bin_whole", None, None)


class Octagon(Prop):
    def __init__(self):
        super().__init__(7, "octagon", 0.92, 1.30)


class BinShark(Prop):
    def __init__(self):
        super().__init__(10, "bin_shark", 0.30480, 0.30480)


class BinSawfish(Prop):
    def __init__(self):
        super().__init__(11, "bin_sawfish", 0.30480, 0.30480)


class CameraDetectionNode:
    def __init__(self):
        rospy.init_node("camera_detection_pose_estimator", anonymous=True)
        rospy.loginfo("Camera detection node started")

        self.front_camera_enabled = True
        self.bottom_camera_enabled = False
        self.active_front_camera_ids = list(range(15))  # Allow all by default

        self.object_id_map = {
            "gate": [0, 1],
            "pipe": [2, 3],
            "torpedo": [4, 5],
            "bin": [6],
            "octagon": [7],
            "all": [0, 1, 2, 3, 4, 5, 6, 7],
        }

        self.object_transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )
        self.props_yaw_pub = rospy.Publisher("props_yaw", PropsYaw, queue_size=10)
        # Initialize tf2 buffer and listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_calibrations = {
            "taluy/cameras/cam_front": CameraCalibration("cameras/cam_front"),
            "taluy/cameras/cam_bottom": CameraCalibration("cameras/cam_bottom"),
        }
        # Use lambda to pass camera source information to the callback
        rospy.Subscriber(
            "/yolo_result",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="front_camera"),
        )
        rospy.Subscriber(
            "/yolo_result_2",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="bottom_camera"),
        )
        self.frame_id_to_camera_ns = {
            "taluy/base_link/bottom_camera_link": "taluy/cameras/cam_bottom",
            "taluy/base_link/front_camera_link": "taluy/cameras/cam_front",
        }
        self.camera_frames = {  # Keep camera_frames for camera frame lookup based on ns
            "taluy/cameras/cam_front": "taluy/base_link/front_camera_optical_link",
            "taluy/cameras/cam_bottom": "taluy/base_link/bottom_camera_optical_link",
        }
        self.props = {
            "gate_sawfish_link": Sawfish(),
            "gate_shark_link": Shark(),
            "red_pipe_link": RedPipe(),
            "white_pipe_link": WhitePipe(),
            "torpedo_map_link": TorpedoMap(),
            "octagon_link": Octagon(),
            "bin_sawfish_link": BinShark(),
            "bin_shark_link": BinSawfish(),
            "torpedo_hole_shark_link": TorpedoHole(),
            "torpedo_hole_sawfish_link": TorpedoHole(),
        }

        self.id_tf_map = {
            "taluy/cameras/cam_front": {
                0: "gate_sawfish_link",
                1: "gate_shark_link",
                2: "red_pipe_link",
                3: "white_pipe_link",
                4: "torpedo_map_link",
                5: "torpedo_hole_link",
                6: "bin_whole_link",
                7: "octagon_link",
            },
            "taluy/cameras/cam_bottom": {
                0: "bin_shark_link",
                1: "bin_sawfish_link",
            },
        }
        # Subscribe to YOLO detections and altitude
        self.altitude = None
        self.pool_depth = rospy.get_param("/env/pool_depth")
        rospy.Subscriber("odom_pressure", Odometry, self.altitude_callback)

        # Services to enable/disable cameras
        rospy.Service(
            "enable_front_camera_detections",
            SetBool,
            self.handle_enable_front_camera,
        )
        rospy.Service(
            "enable_bottom_camera_detections",
            SetBool,
            self.handle_enable_bottom_camera,
        )
        rospy.Service(
            "set_front_camera_focus",
            SetDetectionFocus,
            self.handle_set_front_camera_focus,
        )

    def handle_set_front_camera_focus(self, req):
        focus_objects = [obj.strip() for obj in req.focus_object.split(",")]
        all_target_ids = []
        unfound_objects = []

        for focus_object in focus_objects:
            target_ids = self.object_id_map.get(focus_object)
            if target_ids is not None:
                all_target_ids.extend(target_ids)
            else:
                unfound_objects.append(focus_object)

        if not all_target_ids:
            message = f"Unknown focus object(s): '{req.focus_object}'. Available options: {list(self.object_id_map.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        self.active_front_camera_ids = list(set(all_target_ids))
        message = f"Front camera focus set to IDs: {self.active_front_camera_ids}"

        if unfound_objects:
            message += f". Could not find: {unfound_objects}"

        rospy.loginfo(message)
        return SetDetectionFocusResponse(success=True, message=message)

    def handle_enable_front_camera(self, req):
        self.front_camera_enabled = req.data
        message = "Front camera detections " + ("enabled" if req.data else "disabled")
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_enable_bottom_camera(self, req):
        self.bottom_camera_enabled = req.data
        message = "Bottom camera detections " + ("enabled" if req.data else "disabled")
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def altitude_callback(self, msg: Odometry):
        depth = -msg.pose.pose.position.z
        self.altitude = self.pool_depth - depth
        rospy.loginfo_once(
            f"Calculated altitude from odom_pressure: {self.altitude:.2f} m (pool_depth={self.pool_depth})"
        )

    def calculate_intersection_with_plane(self, point1_odom, point2_odom, z_plane):
        # Calculate t where the z component is z_plane
        if point2_odom.point.z != point1_odom.point.z:
            t = (z_plane - point1_odom.point.z) / (
                point2_odom.point.z - point1_odom.point.z
            )

            # Check if the intersection point is within the segment [point1, point2]
            if 0 <= t <= 1:
                # Calculate intersection point
                x = point1_odom.point.x + t * (
                    point2_odom.point.x - point1_odom.point.x
                )
                y = point1_odom.point.y + t * (
                    point2_odom.point.y - point1_odom.point.y
                )
                return x, y, z_plane
            else:
                # Intersection is outside the defined segment
                return None
        else:
            rospy.logwarn("The line segment is parallel to the ground plane.")
            return None

    def process_altitude_projection(self, detection, camera_ns: str, stamp):
        if self.altitude is None:
            rospy.logwarn("No altitude data available")
            return

        detection_id = detection.results[0].id
        if detection_id != 6:
            return

        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5

        angles = self.camera_calibrations[camera_ns].calculate_angles(
            (bbox_bottom_x, bbox_bottom_y)
        )

        distance = 500.0

        offset_x = math.tan(angles[0]) * distance * 1.0
        offset_y = math.tan(angles[1]) * distance * 1.0

        point1 = PointStamped()
        point1.header.frame_id = self.camera_frames[camera_ns]
        point1.point.x = 0
        point1.point.y = 0
        point1.point.z = 0

        point2 = PointStamped()
        point2.header.frame_id = self.camera_frames[camera_ns]
        point2.point.x = offset_x
        point2.point.y = offset_y
        point2.point.z = distance
        try:
            # Get the transform from the camera frame to the odom frame
            transform = self.tf_buffer.lookup_transform(
                "odom",
                self.camera_frames[camera_ns],
                stamp,
                rospy.Duration(1.0),
            )
            # Transform the ray points from the camera frame to the odom frame
            point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)
            point2_odom = tf2_geometry_msgs.do_transform_point(point2, transform)

            # The ground plane in the 'odom' frame is at a fixed Z-coordinate,
            # which corresponds to the negative pool depth.
            # We calculate the intersection of the camera ray with this plane.
            # We do not add self.altitude here, as both the ray and the plane
            # are now correctly expressed in the same 'odom' frame.
            intersection = self.calculate_intersection_with_plane(
                point1_odom, point2_odom, z_plane=-self.pool_depth
            )
            if intersection:
                x, y, z = intersection
                transform_stamped_msg = TransformStamped()
                transform_stamped_msg.header.stamp = stamp
                transform_stamped_msg.header.frame_id = "odom"
                transform_stamped_msg.child_frame_id = self.id_tf_map[camera_ns][
                    detection_id
                ]

                transform_stamped_msg.transform.translation = Vector3(x, y, z)
                transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
                self.object_transform_pub.publish(transform_stamped_msg)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Transform error: {e}")

    def process_torpedo_holes_on_map(
        self, detection_msg: YoloResult, camera_ns: str, torpedo_map_bbox
    ):
        map_min_x = torpedo_map_bbox.center.x - torpedo_map_bbox.size_x * 0.5
        map_max_x = torpedo_map_bbox.center.x + torpedo_map_bbox.size_x * 0.5
        map_min_y = torpedo_map_bbox.center.y - torpedo_map_bbox.size_y * 0.5
        map_max_y = torpedo_map_bbox.center.y + torpedo_map_bbox.size_y * 0.5

        detected_holes_in_map = []
        camera_frame = self.camera_frames[camera_ns]

        # First, find all hole detections inside the torpedo map
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id

            if detection_id == 5:  # Torpedo hole ID
                hole_center_x = detection.bbox.center.x
                hole_center_y = detection.bbox.center.y

                # Check if the center of the hole is within the map's bounding box
                if (
                    map_min_x <= hole_center_x <= map_max_x
                    and map_min_y <= hole_center_y <= map_max_y
                ):
                    detected_holes_in_map.append(detection)

        # We expect to find exactly two holes. If not, we cannot proceed.
        if len(detected_holes_in_map) != 2:
            rospy.logwarn_throttle(
                5,
                f"Expected 2 torpedo holes, but found {len(detected_holes_in_map)}. Skipping.",
            )
            return

        # Determine which hole is upper and which is bottom based on their Y coordinate.
        # In image coordinates, a smaller Y value means it is higher up in the image.
        hole1 = detected_holes_in_map[0]
        hole2 = detected_holes_in_map[1]

        if hole1.bbox.center.y < hole2.bbox.center.y:
            upper_hole_detection = hole1
            bottom_hole_detection = hole2
        else:
            upper_hole_detection = hole2
            bottom_hole_detection = hole1

        if upper_hole_detection.bbox.center.x > bottom_hole_detection.bbox.center.x:
            upper_hole_child_frame_id = "torpedo_hole_sawfish_link"
            bottom_hole_child_frame_id = "torpedo_hole_shark_link"
        else:
            upper_hole_child_frame_id = "torpedo_hole_shark_link"
            bottom_hole_child_frame_id = "torpedo_hole_sawfish_link"

        rospy.loginfo_once(
            f"Upper hole assigned to {upper_hole_child_frame_id}, bottom to {bottom_hole_child_frame_id}"
        )

        # Publish transform for the upper hole with its new name
        self._publish_torpedo_hole_transform(
            upper_hole_detection,
            camera_ns,
            camera_frame,
            upper_hole_child_frame_id,
            upper_hole_child_frame_id,
            detection_msg.header.stamp,
        )

        # Publish transform for the bottom hole with its new name
        self._publish_torpedo_hole_transform(
            bottom_hole_detection,
            camera_ns,
            camera_frame,
            bottom_hole_child_frame_id,
            bottom_hole_child_frame_id,
            detection_msg.header.stamp,
        )

    def _publish_torpedo_hole_transform(
        self, detection, camera_ns, camera_frame, prop_key, child_frame_id, stamp
    ):
        prop_to_use = self.props.get(prop_key)
        if not prop_to_use:
            rospy.logerr(f"Prop '{prop_key}' not found. Cannot estimate distance.")
            return

        distance = prop_to_use.estimate_distance(
            detection.bbox.size_y,
            detection.bbox.size_x,
            self.camera_calibrations[camera_ns],
        )

        if distance is None:
            return

        angles = self.camera_calibrations[camera_ns].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        try:
            camera_to_odom_transform = self.tf_buffer.lookup_transform(
                camera_frame,
                "odom",
                stamp,
                rospy.Duration(1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Transform error for {child_frame_id}: {e}")
            return

        offset_x = math.tan(angles[0]) * distance * 1.0
        offset_y = math.tan(angles[1]) * distance * 1.0

        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = stamp
        transform_stamped_msg.header.frame_id = camera_frame
        transform_stamped_msg.child_frame_id = child_frame_id

        transform_stamped_msg.transform.translation = Vector3(
            offset_x, offset_y, distance
        )
        transform_stamped_msg.transform.rotation = (
            camera_to_odom_transform.transform.rotation
        )
        self.object_transform_pub.publish(transform_stamped_msg)

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

    def detection_callback(self, detection_msg: YoloResult, camera_source: str):
        # Determine camera_ns based on the source passed by the subscriber
        if camera_source == "front_camera":
            if not self.front_camera_enabled:
                return
            camera_ns = "taluy/cameras/cam_front"
        elif camera_source == "bottom_camera":
            if not self.bottom_camera_enabled:
                return
            camera_ns = "taluy/cameras/cam_bottom"
        else:
            rospy.logerr(f"Unknown camera_source: {camera_source}")
            return  # Stop processing if the source is unknown

        camera_frame = self.camera_frames[camera_ns]

        # Search for torpedo map
        torpedo_map_bbox = None
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            detection_id = detection.results[0].id
            if detection_id == 4:  # Torpedo map ID
                torpedo_map_bbox = detection.bbox
                break

        torpedo_ids = set(self.object_id_map.get("torpedo", []))
        process_torpedo = True
        if camera_source == "front_camera":
            if not torpedo_ids.intersection(self.active_front_camera_ids):
                process_torpedo = False

        if torpedo_map_bbox and process_torpedo:
            self.process_torpedo_holes_on_map(
                detection_msg, camera_ns, torpedo_map_bbox
            )

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            skip_inside_image = False
            detection_id = detection.results[0].id

            if camera_source == "front_camera":
                if detection_id not in self.active_front_camera_ids:
                    continue

            if detection_id == 5:
                continue

            if detection_id not in self.id_tf_map[camera_ns]:
                continue
            if camera_ns == "taluy/cameras/cam_bottom" and detection_id in [0, 1]:
                skip_inside_image = True
                # use altidude for bin
                distance = self.altitude

            if detection_id == 6:
                self.process_altitude_projection(
                    detection, camera_ns, detection_msg.header.stamp
                )
                continue
            if not skip_inside_image:
                if self.check_if_detection_is_inside_image(detection) is False:
                    continue
            prop_name = self.id_tf_map[camera_ns][detection_id]
            if prop_name not in self.props:
                continue

            prop = self.props[prop_name]

            if not skip_inside_image:  # Calculate distance using object dimensions
                distance = prop.estimate_distance(
                    detection.bbox.size_y,
                    detection.bbox.size_x,
                    self.camera_calibrations[camera_ns],
                )

            if distance is None:
                continue

            angles = self.camera_calibrations[camera_ns].calculate_angles(
                (detection.bbox.center.x, detection.bbox.center.y)
            )

            props_yaw_msg = PropsYaw()
            props_yaw_msg.header.stamp = detection_msg.header.stamp
            props_yaw_msg.object = prop.name
            props_yaw_msg.angle = -angles[0]
            self.props_yaw_pub.publish(props_yaw_msg)
            try:
                camera_to_odom_transform = self.tf_buffer.lookup_transform(
                    camera_frame,
                    "odom",
                    detection_msg.header.stamp,
                    rospy.Duration(1.0),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logerr(f"Transform error: {e}")
                return

            offset_x = math.tan(angles[0]) * distance * 1.0
            offset_y = math.tan(angles[1]) * distance * 1.0
            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = detection_msg.header.stamp
            transform_stamped_msg.header.frame_id = camera_frame
            transform_stamped_msg.child_frame_id = prop_name

            transform_stamped_msg.transform.translation = Vector3(
                offset_x, offset_y, distance
            )
            transform_stamped_msg.transform.rotation = (
                camera_to_odom_transform.transform.rotation
            )
            # Calculate the rotation based on odom
            self.object_transform_pub.publish(transform_stamped_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
