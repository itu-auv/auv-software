#!/usr/bin/env python3

import glob
import math

import rospy
import rospkg
import yaml
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
from auv_msgs.srv import SetModelConfig, SetModelConfigResponse
import auv_common_lib.vision.camera_calibrations as camera_calibrations
import tf2_ros
import tf2_geometry_msgs
from dynamic_reconfigure.client import Client


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
    """Generic prop for distance estimation based on known real dimensions."""

    def __init__(self, name: str, real_height: float, real_width: float):
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


def _load_all_model_configs(config_dir: str) -> dict:
    """Load all YAML model configs from directory."""
    configs = {}
    for yaml_path in glob.glob(f"{config_dir}/*.yaml"):
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            model_name = config.get("model_name")
            if model_name:
                configs[model_name] = config
                rospy.loginfo(f"Loaded model config: {model_name} from {yaml_path}")
        except Exception as e:
            rospy.logwarn(f"Failed to load config {yaml_path}: {e}")
    return configs


class CameraDetectionNode:
    def __init__(self):
        rospy.init_node("camera_detection_pose_estimator", anonymous=True)
        rospy.loginfo("Camera detection node started")
        self.selected_side = "left"  # Default value
        self.dynamic_reconfigure_client = Client(
            "smach_parameters_server",
            timeout=10,
            config_callback=self.dynamic_reconfigure_callback,
        )
        self.front_camera_enabled = True
        self.bottom_camera_enabled = False

        self.red_pipe_x = None

        # Load all model configurations from YAML files
        try:
            rospack = rospkg.RosPack()
            config_dir = rospy.get_param(
                "~model_config_dir",
                rospack.get_path("auv_detection") + "/config/models",
            )
        except rospkg.common.ResourceNotFound:
            rospy.logfatal("auv_detection package not found")
            raise

        self.model_configs = _load_all_model_configs(config_dir)
        if not self.model_configs:
            rospy.logfatal(f"No model configs found in {config_dir}")
            raise RuntimeError("No model configurations loaded")

        # Initialize model state
        self.active_model = None
        self.id_to_class = {}
        self.focus_groups = {}
        self.class_name_to_id = {}
        self.active_focus_classes = set()

        # Activate initial model (defaults to robosub for backward compatibility)
        initial_model = rospy.get_param("~initial_model", "robosub")
        if not self._activate_model(initial_model):
            # Fallback to first available model
            fallback = list(self.model_configs.keys())[0]
            rospy.logwarn(
                f"Initial model '{initial_model}' not found, using '{fallback}'"
            )
            self._activate_model(fallback)

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
            "/yolo_result_front",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="front_camera"),
            queue_size=1,
        )
        rospy.Subscriber(
            "/yolo_result_bottom",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="bottom_camera"),
            queue_size=1,
        )
        self.frame_id_to_camera_ns = {
            "taluy/base_link/bottom_camera_link": "taluy/cameras/cam_bottom",
            "taluy/base_link/front_camera_link": "taluy/cameras/cam_front",
        }
        self.camera_frames = {
            "taluy/cameras/cam_front": "taluy/base_link/front_camera_optical_link_stabilized",
            "taluy/cameras/cam_bottom": "taluy/base_link/bottom_camera_optical_link",
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
        rospy.Service(
            "set_model_config",
            SetModelConfig,
            self.handle_set_model_config,
        )

    def _activate_model(self, model_name: str) -> bool:
        """Switch to a different model's class configuration."""
        if model_name not in self.model_configs:
            rospy.logerr(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.model_configs.keys())}"
            )
            return False

        config = self.model_configs[model_name]
        self.active_model = model_name

        # Build ID -> class_info lookup per camera
        self.id_to_class = {"front": {}, "bottom": {}}
        for camera_key in ["front", "bottom"]:
            camera_config = config.get("cameras", {}).get(camera_key, {})
            for class_name, class_props in camera_config.get("classes", {}).items():
                class_id = class_props["id"]
                self.id_to_class[camera_key][class_id] = {
                    "name": class_name,
                    "tf_frame": class_props["tf_frame"],
                    "real_height": class_props.get("real_height"),
                    "real_width": class_props.get("real_width"),
                }

        # Build focus groups: group_name -> set of class_names
        self.focus_groups = {}
        for group_name, class_list in config.get("focus_groups", {}).items():
            self.focus_groups[group_name] = set(class_list) if class_list else set()

        # Build reverse lookup: class_name -> class_id (for focus filtering)
        self.class_name_to_id = {}
        for camera_key, classes in self.id_to_class.items():
            for class_id, class_info in classes.items():
                self.class_name_to_id[class_info["name"]] = class_id

        # Reset active focus to "all" for new model
        self.active_focus_classes = self.focus_groups.get("all", set())

        rospy.loginfo(f"Activated model config: {model_name}")
        return True

    def handle_set_model_config(self, req):
        """Handle service call to switch model configuration."""
        success = self._activate_model(req.model_name)

        return SetModelConfigResponse(
            success=success,
            message=(
                f"Model set to '{req.model_name}'"
                if success
                else f"Unknown model: {req.model_name}"
            ),
            available_models=list(self.model_configs.keys()),
            active_model=self.active_model or "",
        )

    def dynamic_reconfigure_callback(self, config):
        if config is None:
            rospy.logwarn("Could not get parameters from server for camera detection")
            return
        self.selected_side = config.slalom_direction
        rospy.loginfo(f"Slalom direction updated to: {self.selected_side}")

    def handle_set_front_camera_focus(self, req):
        focus_objects = [
            obj.strip() for obj in req.focus_object.split(",") if obj.strip()
        ]

        if not focus_objects:
            message = (
                f"Empty focus object provided. No changes made. "
                f"Available options: {list(self.focus_groups.keys())}"
            )
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        unfound_objects = [obj for obj in focus_objects if obj not in self.focus_groups]

        if unfound_objects:
            message = (
                f"Unknown focus group(s): '{', '.join(unfound_objects)}'. "
                f"Available: {list(self.focus_groups.keys())}"
            )
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        if "none" in focus_objects and len(focus_objects) > 1:
            message = "Cannot specify 'none' with other focus objects."
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        all_classes = set()
        for focus_object in focus_objects:
            all_classes.update(self.focus_groups.get(focus_object, set()))

        self.active_focus_classes = all_classes

        if "none" in focus_objects:
            message = "Front camera focus set to none. Detections will be ignored."
        else:
            message = f"Front camera focus set to classes: {all_classes}"

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

    def process_altitude_projection(self, detection, camera_ns: str, stamp, tf_frame):
        if self.altitude is None:
            rospy.logwarn("No altitude data available")
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
                transform_stamped_msg.child_frame_id = tf_frame

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
        self, detection_msg: YoloResult, camera_ns: str, torpedo_map_bbox, camera_key
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
            raw_id = detection.results[0].id
            class_info = self.id_to_class.get(camera_key, {}).get(raw_id)

            if class_info and class_info["name"] == "torpedo_hole":
                hole_center_x = detection.bbox.center.x
                hole_center_y = detection.bbox.center.y

                # Check if the center of the hole is within the map's bounding box
                if (
                    map_min_x <= hole_center_x <= map_max_x
                    and map_min_y <= hole_center_y <= map_max_y
                ):
                    detected_holes_in_map.append((detection, class_info))

        # We expect to find exactly two holes. If not, we cannot proceed.
        if len(detected_holes_in_map) != 2:
            rospy.logwarn_throttle(
                5,
                f"Expected 2 torpedo holes, but found {len(detected_holes_in_map)}. Skipping.",
            )
            return

        # Determine which hole is upper and which is bottom based on their Y coordinate.
        # In image coordinates, a smaller Y value means it is higher up in the image.
        hole1, class_info1 = detected_holes_in_map[0]
        hole2, class_info2 = detected_holes_in_map[1]

        if hole1.bbox.center.y < hole2.bbox.center.y:
            upper_hole_detection = hole1
            bottom_hole_detection = hole2
            upper_class_info = class_info1
            bottom_class_info = class_info2
        else:
            upper_hole_detection = hole2
            bottom_hole_detection = hole1
            upper_class_info = class_info2
            bottom_class_info = class_info1

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
            upper_class_info,
            upper_hole_child_frame_id,
            detection_msg.header.stamp,
        )

        # Publish transform for the bottom hole with its new name
        self._publish_torpedo_hole_transform(
            bottom_hole_detection,
            camera_ns,
            camera_frame,
            bottom_class_info,
            bottom_hole_child_frame_id,
            detection_msg.header.stamp,
        )

    def _publish_torpedo_hole_transform(
        self, detection, camera_ns, camera_frame, class_info, child_frame_id, stamp
    ):
        # Create Prop instance from class_info for distance estimation
        prop = Prop(
            name=class_info["name"],
            real_height=class_info.get("real_height"),
            real_width=class_info.get("real_width"),
        )

        distance = prop.estimate_distance(
            detection.bbox.size_y,
            detection.bbox.size_x,
            self.camera_calibrations[camera_ns],
        )

        if distance is None:
            return

        angles = self.camera_calibrations[camera_ns].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        offset_x = math.tan(angles[0]) * distance * 1.0
        offset_y = math.tan(angles[1]) * distance * 1.0

        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = stamp
        transform_stamped_msg.header.frame_id = camera_frame
        transform_stamped_msg.child_frame_id = child_frame_id

        transform_stamped_msg.transform.translation = Vector3(
            offset_x, offset_y, distance
        )
        transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)

        try:
            # Create a PoseStamped message from the TransformStamped
            pose_stamped = PoseStamped()
            pose_stamped.header = transform_stamped_msg.header
            pose_stamped.pose.position = transform_stamped_msg.transform.translation
            pose_stamped.pose.orientation = transform_stamped_msg.transform.rotation

            # Transform the PoseStamped message
            transformed_pose_stamped = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(4.0)
            )

            # Create a new TransformStamped message from the transformed PoseStamped
            final_transform_stamped = TransformStamped()
            final_transform_stamped.header = transformed_pose_stamped.header
            final_transform_stamped.child_frame_id = child_frame_id
            final_transform_stamped.transform.translation = (
                transformed_pose_stamped.pose.position
            )
            final_transform_stamped.transform.rotation = (
                transform_stamped_msg.transform.rotation
            )

            self.object_transform_pub.publish(final_transform_stamped)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Transform error for {child_frame_id}: {e}")
            return

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
        # Determine camera_ns and camera_key based on the source passed by the subscriber
        if camera_source == "front_camera":
            if not self.front_camera_enabled:
                return
            camera_ns = "taluy/cameras/cam_front"
            camera_key = "front"
        elif camera_source == "bottom_camera":
            if not self.bottom_camera_enabled:
                return
            camera_ns = "taluy/cameras/cam_bottom"
            camera_key = "bottom"
        else:
            rospy.logerr(f"Unknown camera_source: {camera_source}")
            return  # Stop processing if the source is unknown

        camera_frame = self.camera_frames[camera_ns]
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
            rospy.logwarn_throttle(15, f"Transform error: {e}")
            return

        # Search for torpedo map (semantic lookup)
        torpedo_map_bbox = None
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            raw_id = detection.results[0].id
            class_info = self.id_to_class.get(camera_key, {}).get(raw_id)
            if class_info and class_info["name"] == "torpedo_map":
                torpedo_map_bbox = detection.bbox
                break

        # Check if torpedo processing is needed
        process_torpedo = "torpedo_map" in self.active_focus_classes
        process_torpedo_holes = "torpedo_hole" in self.active_focus_classes

        if torpedo_map_bbox and process_torpedo and process_torpedo_holes:
            self.process_torpedo_holes_on_map(
                detection_msg, camera_ns, torpedo_map_bbox, camera_key
            )

        # Track red pipe for slalom filtering (semantic lookup)
        red_pipe_x = self.red_pipe_x
        largest_red_pipe_size = 0
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            raw_id = detection.results[0].id
            class_info = self.id_to_class.get(camera_key, {}).get(raw_id)
            if class_info and class_info["name"] == "red_pipe":
                if detection.bbox.size_y > largest_red_pipe_size:
                    red_pipe_x = detection.bbox.center.x
                    largest_red_pipe_size = detection.bbox.size_y
                    self.red_pipe_x = red_pipe_x

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            skip_inside_image = False
            raw_id = detection.results[0].id

            # Look up class info from active model config
            class_info = self.id_to_class.get(camera_key, {}).get(raw_id)
            if class_info is None:
                continue 

            class_name = class_info["name"]
            tf_frame = class_info["tf_frame"]

            if camera_source == "front_camera":
                if class_name not in self.active_focus_classes:
                    continue

            if class_name == "torpedo_hole":
                continue

            if class_name == "white_pipe" and red_pipe_x is not None:
                white_x = detection.bbox.center.x
                if self.selected_side == "left" and white_x > red_pipe_x:
                    continue
                if self.selected_side == "right" and white_x < red_pipe_x:
                    continue

            if camera_key == "bottom" and class_name in ["bin_shark", "bin_sawfish"]:
                skip_inside_image = True
                distance = self.altitude
            if class_name == "bin_whole":
                self.process_altitude_projection(
                    detection, camera_ns, detection_msg.header.stamp, tf_frame
                )
                continue

            if not skip_inside_image:
                if self.check_if_detection_is_inside_image(detection) is False:
                    continue

            # Create Prop instance from class_info for distance estimation
            prop = Prop(
                name=class_name,
                real_height=class_info.get("real_height"),
                real_width=class_info.get("real_width"),
            )

            if not skip_inside_image:
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
            props_yaw_msg.object = class_name
            props_yaw_msg.angle = -angles[0]
            self.props_yaw_pub.publish(props_yaw_msg)

            offset_x = math.tan(angles[0]) * distance * 1.0
            offset_y = math.tan(angles[1]) * distance * 1.0

            if (offset_x**2 + offset_y**2 + distance**2) > 30**2:  # 30 meters squared
                rospy.logdebug(f"Detection for {tf_frame} is too far away. Skipping.")
                continue

            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = detection_msg.header.stamp
            transform_stamped_msg.header.frame_id = camera_frame
            transform_stamped_msg.child_frame_id = tf_frame

            transform_stamped_msg.transform.translation = Vector3(
                offset_x, offset_y, distance
            )
            transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)

            try:
                # Create a PoseStamped message from the TransformStamped
                pose_stamped = PoseStamped()
                pose_stamped.header = transform_stamped_msg.header
                pose_stamped.pose.position = transform_stamped_msg.transform.translation
                pose_stamped.pose.orientation = transform_stamped_msg.transform.rotation

                # Transform the PoseStamped message
                transformed_pose_stamped = self.tf_buffer.transform(
                    pose_stamped, "odom", rospy.Duration(4.0)
                )

                # Create a new TransformStamped message from the transformed PoseStamped
                final_transform_stamped = TransformStamped()
                final_transform_stamped.header = transformed_pose_stamped.header
                final_transform_stamped.child_frame_id = tf_frame
                final_transform_stamped.transform.translation = (
                    transformed_pose_stamped.pose.position
                )
                final_transform_stamped.transform.rotation = (
                    transform_stamped_msg.transform.rotation
                )

                self.object_transform_pub.publish(final_transform_stamped)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logerr(f"Transform error for {tf_frame}: {e}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
