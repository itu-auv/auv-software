#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    TransformStamped,
    Vector3,
    Quaternion,
)
from ultralytics_ros.msg import YoloResult
from auv_msgs.msg import PropsYaw
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetDetectionFocus, SetDetectionFocusResponse
import auv_common_lib.vision.camera_calibrations as camera_calibrations
import tf2_ros
import tf2_geometry_msgs
from dynamic_reconfigure.client import Client

# Import local modules
from constants import (
    OBJECT_ID_MAP, CAMERA_NAMES, CAMERA_FRAMES, ID_TF_MAP,
    SAWFISH_ID, SHARK_ID, RED_PIPE_ID, WHITE_PIPE_ID,
    TORPEDO_MAP_ID, TORPEDO_HOLE_ID, BIN_WHOLE_ID,
    OCTAGON_ID, BIN_SHARK_ID, BIN_SAWFISH_ID
)
from props import PROPS_CONFIG, LINK_TO_PROP_MAP
from pose_utils import (
    calculate_angles, estimate_distance,
    calculate_intersection_with_plane, check_if_detection_is_inside_image
)

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

        self.object_id_map = OBJECT_ID_MAP
        self.active_front_camera_ids = self.object_id_map["gate"]  # Allow gate by default

        self.object_transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )
        self.props_yaw_pub = rospy.Publisher("props_yaw", PropsYaw, queue_size=10)
        # Initialize tf2 buffer and listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Camera calibrations
        self.camera_calibrations = {
            CAMERA_NAMES["front"]: camera_calibrations.CameraCalibrationFetcher(
                "cameras/cam_front", True
            ).get_camera_info(),
            CAMERA_NAMES["bottom"]: camera_calibrations.CameraCalibrationFetcher(
                "cameras/cam_bottom", True
            ).get_camera_info(),
        }

        # Use lambda to pass camera source information to the callback
        rospy.Subscriber(
            "/yolo_result",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="front_camera"),
            queue_size=1,
        )
        rospy.Subscriber(
            "/yolo_result_2",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="bottom_camera"),
            queue_size=1,
        )

        self.camera_frames = CAMERA_FRAMES
        self.id_tf_map = ID_TF_MAP

        # Subscribe to YOLO detections and altitude
        self.altitude = None
        self.pool_depth = rospy.get_param("/env/pool_depth", 2.0) # Default to 2.0 if param not found
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
            message = f"Empty focus object provided. No changes made. Available options: {list(self.object_id_map.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        unfound_objects = [
            obj for obj in focus_objects if obj not in self.object_id_map
        ]

        if unfound_objects:
            message = f"Unknown focus object(s): '{', '.join(unfound_objects)}'. Available options: {list(self.object_id_map.keys())}"
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        if "none" in focus_objects and len(focus_objects) > 1:
            message = "Cannot specify 'none' with other focus objects."
            rospy.logwarn(message)
            return SetDetectionFocusResponse(success=False, message=message)

        all_target_ids = []
        for focus_object in focus_objects:
            all_target_ids.extend(self.object_id_map[focus_object])

        self.active_front_camera_ids = list(set(all_target_ids))

        if "none" in focus_objects:
            message = "Front camera focus set to none. Detections will be ignored."
        else:
            message = f"Front camera focus set to IDs: {self.active_front_camera_ids}"

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

    def process_altitude_projection(self, detection, camera_ns: str, stamp):
        if self.altitude is None:
            rospy.logwarn("No altitude data available")
            return

        detection_id = detection.results[0].id
        if detection_id != BIN_WHOLE_ID:
            return

        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5

        calibration = self.camera_calibrations[camera_ns]
        angles = calculate_angles((bbox_bottom_x, bbox_bottom_y), calibration.K)

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

            intersection = calculate_intersection_with_plane(
                point1_odom, point2_odom, z_plane=-self.pool_depth
            )
            if intersection:
                x, y, z = intersection
                child_frame_id = self.id_tf_map[camera_ns][detection_id]
                self.publish_transform(
                    child_frame_id,
                    x, y, z,
                    Quaternion(0, 0, 0, 1),
                    stamp,
                    "odom"
                )

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

            if detection_id == TORPEDO_HOLE_ID:  # Torpedo hole ID
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

        self._process_and_publish_prop(
            upper_hole_detection, camera_ns, camera_frame, upper_hole_child_frame_id, detection_msg.header.stamp, override_prop_key="torpedo_hole_shark_link"
        ) # Using shark/sawfish links which map to PROPS_CONFIG["torpedo_hole"] in props.py

        self._process_and_publish_prop(
            bottom_hole_detection, camera_ns, camera_frame, bottom_hole_child_frame_id, detection_msg.header.stamp, override_prop_key="torpedo_hole_sawfish_link"
        )

    def _process_and_publish_prop(self, detection, camera_ns, camera_frame, child_frame_id, stamp, override_prop_key=None):
        prop_key = override_prop_key if override_prop_key else child_frame_id
        prop_config = LINK_TO_PROP_MAP.get(prop_key)

        if not prop_config:
            # Fallback or error if prop config not found
            # Sometimes child_frame_id is the key itself
            if child_frame_id in LINK_TO_PROP_MAP:
                prop_config = LINK_TO_PROP_MAP[child_frame_id]
            else:
                 rospy.logerr(f"Prop configuration for '{prop_key}' or '{child_frame_id}' not found.")
                 return

        calibration = self.camera_calibrations[camera_ns]
        distance = estimate_distance(
            prop_config.real_height,
            prop_config.real_width,
            detection.bbox.size_y,
            detection.bbox.size_x,
            calibration.K
        )

        if distance is None:
            return

        angles = calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y),
            calibration.K
        )

        offset_x = math.tan(angles[0]) * distance * 1.0
        offset_y = math.tan(angles[1]) * distance * 1.0

        # Publish Yaw
        # Note: PropsYaw uses "object" field, which usually expects prop name, not link name?
        # Original code used prop.name from Prop object.
        props_yaw_msg = PropsYaw()
        props_yaw_msg.header.stamp = stamp
        props_yaw_msg.object = prop_config.name
        props_yaw_msg.angle = -angles[0]
        self.props_yaw_pub.publish(props_yaw_msg)

        if (offset_x**2 + offset_y**2 + distance**2) > 30**2:  # 30 meters squared
             rospy.logdebug(f"Detection for {prop_config.name} is too far away. Skipping.")
             return

        self.publish_transform_from_camera(
             child_frame_id,
             Vector3(offset_x, offset_y, distance),
             Quaternion(0, 0, 0, 1),
             stamp,
             camera_frame
        )

    def publish_transform(self, child_frame_id, x, y, z, rotation, stamp, frame_id):
        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = stamp
        transform_stamped_msg.header.frame_id = frame_id
        transform_stamped_msg.child_frame_id = child_frame_id

        transform_stamped_msg.transform.translation = Vector3(x, y, z)
        transform_stamped_msg.transform.rotation = rotation
        self.object_transform_pub.publish(transform_stamped_msg)

    def publish_transform_from_camera(self, child_frame_id, translation, rotation, stamp, camera_frame):
        # Create a PoseStamped message from the relative transform
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = camera_frame
        pose_stamped.pose.position = translation
        pose_stamped.pose.orientation = rotation

        try:
            # Transform the PoseStamped message to odom
            transformed_pose_stamped = self.tf_buffer.transform(
                pose_stamped, "odom", rospy.Duration(4.0)
            )

            # Publish the transform in odom frame
            self.publish_transform(
                child_frame_id,
                transformed_pose_stamped.pose.position.x,
                transformed_pose_stamped.pose.position.y,
                transformed_pose_stamped.pose.position.z,
                rotation, # We keep original rotation as per original code, or should we transform it?
                          # Original code: final...rotation = transform_stamped_msg.transform.rotation
                          # which was (0,0,0,1). So it keeps orientation relative to something?
                          # Actually original code sets rotation to (0,0,0,1) in camera frame,
                          # converts position to odom, and publishes transform in odom
                          # BUT sets rotation to the ORIGINAL (0,0,0,1).
                          # This effectively means the object has 0 orientation in Odom frame?
                          # Or is it 0 relative to child?
                          # TransformStamped child_frame_id is the object.
                          # If rotation is (0,0,0,1), it means child frame is aligned with parent frame (odom).
                          # So yes, it sets orientation to identity in odom.
                transformed_pose_stamped.header.stamp,
                "odom"
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Transform error for {child_frame_id}: {e}")

    def detection_callback(self, detection_msg: YoloResult, camera_source: str):
        # Determine camera_ns based on the source passed by the subscriber
        if camera_source == "front_camera":
            if not self.front_camera_enabled:
                return
            camera_ns = CAMERA_NAMES["front"]
        elif camera_source == "bottom_camera":
            if not self.bottom_camera_enabled:
                return
            camera_ns = CAMERA_NAMES["bottom"]
        else:
            rospy.logerr(f"Unknown camera_source: {camera_source}")
            return

        camera_frame = self.camera_frames[camera_ns]
        # Check if camera to odom transform is available
        if not self.tf_buffer.can_transform("odom", camera_frame, detection_msg.header.stamp, rospy.Duration(1.0)):
             return

        # Search for torpedo map
        torpedo_map_bbox = None
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            if detection.results[0].id == TORPEDO_MAP_ID:
                torpedo_map_bbox = detection.bbox
                break

        torpedo_ids = set(self.object_id_map.get("torpedo", []))
        torpedo_holes_ids = set(self.object_id_map.get("torpedo_holes", []))
        process_torpedo = True
        process_torpedo_holes = True
        if camera_source == "front_camera":
            if not torpedo_ids.intersection(self.active_front_camera_ids):
                process_torpedo = False
            if not torpedo_holes_ids.intersection(self.active_front_camera_ids):
                process_torpedo_holes = False

        if torpedo_map_bbox and process_torpedo and process_torpedo_holes:
            self.process_torpedo_holes_on_map(
                detection_msg, camera_ns, torpedo_map_bbox
            )

        # Red Pipe logic
        red_pipe_x = self.red_pipe_x
        red_pipes = [
            d for d in detection_msg.detections.detections if d.results[0].id == RED_PIPE_ID
        ]
        if red_pipes:
            largest_red_pipe = max(red_pipes, key=lambda d: d.bbox.size_y)
            red_pipe_x = largest_red_pipe.bbox.center.x
            self.red_pipe_x = red_pipe_x

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue

            detection_id = detection.results[0].id

            # Filter out based on active ids
            if camera_source == "front_camera":
                if detection_id not in self.active_front_camera_ids:
                    continue

            # Special cases
            if detection_id == TORPEDO_HOLE_ID:
                continue

            if detection_id == WHITE_PIPE_ID and red_pipe_x is not None:
                white_x = detection.bbox.center.x
                if self.selected_side == "left" and white_x > red_pipe_x:
                    continue
                if self.selected_side == "right" and white_x < red_pipe_x:
                    continue

            if detection_id not in self.id_tf_map[camera_ns]:
                continue

            skip_inside_image = False
            # Special logic for bin shark/sawfish from bottom camera
            if camera_ns == CAMERA_NAMES["bottom"] and detection_id in [0, 1]:
                 # Note: in constants.py or props.py we don't have direct mapping for 0->bin_shark for bottom camera
                 # Wait, ID_TF_MAP in constants.py:
                 # CAMERA_NAMES["bottom"]: { 0: "bin_shark_link", 1: "bin_sawfish_link" }
                 # So yes, detection_id 0 and 1 from bottom camera are bins.
                 skip_inside_image = True
                 distance = self.altitude
            else:
                 distance = None

            if detection_id == BIN_WHOLE_ID:
                self.process_altitude_projection(
                    detection, camera_ns, detection_msg.header.stamp
                )
                continue

            if not skip_inside_image:
                if not check_if_detection_is_inside_image(
                    detection.bbox.center.x, detection.bbox.center.y,
                    detection.bbox.size_x, detection.bbox.size_y
                ):
                    continue

            child_frame_id = self.id_tf_map[camera_ns][detection_id]

            if skip_inside_image and distance is not None:
                # If we used altitude, we already have distance.
                # We need to manually calculate offsets and publish.
                # Logic is similar to _process_and_publish_prop but we force distance.

                calibration = self.camera_calibrations[camera_ns]
                angles = calculate_angles(
                    (detection.bbox.center.x, detection.bbox.center.y),
                    calibration.K
                )

                # Publish Yaw
                prop_config = LINK_TO_PROP_MAP.get(child_frame_id)
                if prop_config:
                    props_yaw_msg = PropsYaw()
                    props_yaw_msg.header.stamp = detection_msg.header.stamp
                    props_yaw_msg.object = prop_config.name
                    props_yaw_msg.angle = -angles[0]
                    self.props_yaw_pub.publish(props_yaw_msg)

                offset_x = math.tan(angles[0]) * distance * 1.0
                offset_y = math.tan(angles[1]) * distance * 1.0

                self.publish_transform_from_camera(
                    child_frame_id,
                    Vector3(offset_x, offset_y, distance),
                    Quaternion(0, 0, 0, 1),
                    detection_msg.header.stamp,
                    camera_frame
                )

            else:
                self._process_and_publish_prop(
                    detection, camera_ns, camera_frame, child_frame_id, detection_msg.header.stamp
                )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
