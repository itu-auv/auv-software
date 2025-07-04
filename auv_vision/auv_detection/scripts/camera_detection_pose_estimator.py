#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
    Transform,
    Vector3,
    Quaternion,
)
from ultralytics_ros.msg import YoloResult
from nav_msgs.msg import Odometry
import auv_common_lib.vision.camera_calibrations as camera_calibrations
import tf2_ros
import tf2_geometry_msgs


class CameraCalibration:
    """Handles camera calibration parameters and calculations"""

    def __init__(self, namespace: str):
        self.calibration = camera_calibrations.CameraCalibrationFetcher(
            namespace, True
        ).get_camera_info()

    def calculate_angles(self, pixel_coordinates: tuple) -> tuple:
        """Calculate angles from pixel coordinates to optical center"""
        fx = self.calibration.K[0]
        fy = self.calibration.K[4]
        cx = self.calibration.K[2]
        cy = self.calibration.K[5]
        norm_x = (pixel_coordinates[0] - cx) / fx
        norm_y = (pixel_coordinates[1] - cy) / fy
        return math.atan(norm_x), math.atan(norm_y)

    def distance_from_height(self, real_height: float, measured_height: float) -> float:
        """Calculate distance using known height"""
        focal_length = self.calibration.K[4]
        return (real_height * focal_length) / measured_height

    def distance_from_width(self, real_width: float, measured_width: float) -> float:
        """Calculate distance using known width"""
        focal_length = self.calibration.K[0]
        return (real_width * focal_length) / measured_width


class Prop:
    """Base class for all detectable props"""

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
    ) -> float:
        """Estimate distance using both height and width if available"""
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

        rospy.logerr(f"Could not estimate distance for prop {self.name}")
        return None


# Prop subclasses with specific dimensions
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


class TorpedoHole(Prop):
    def __init__(self):
        super().__init__(13, "torpedo_hole", None, None)


class BinWhole(Prop):
    def __init__(self):
        super().__init__(9, "bin_whole", None, None)


class Octagon(Prop):
    def __init__(self):
        super().__init__(14, "octagon", 0.92, 1.30)


class BinRed(Prop):
    def __init__(self):
        super().__init__(10, "bin_red", 0.30480, 0.30480)


class BinBlue(Prop):
    def __init__(self):
        super().__init__(11, "bin_blue", 0.30480, 0.30480)


class CameraDetectionNode:
    """Main node for processing camera detections and estimating object poses"""

    def __init__(self):
        rospy.init_node("camera_detection_pose_estimator", anonymous=True)
        rospy.loginfo("Camera detection node started")

        # Pool parameters
        self.pool_depth = rospy.get_param("~pool_depth", 2.2)
        self.altitude = None

        # Initialize publishers, subscribers and TF
        self._setup_communication()
        self._load_configuration()
        self._initialize_props()

    def _setup_communication(self):
        """Setup ROS publishers, subscribers and TF"""
        self.object_transform_pub = rospy.Publisher(
            "update_object_transforms", TransformStamped, queue_size=10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers for different cameras
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
        rospy.Subscriber("odom_pressure", Odometry, self.altitude_callback)

    def _load_configuration(self):
        """Load camera configurations and mappings"""
        self.camera_calibrations = {
            "taluy/cameras/cam_front": CameraCalibration("cameras/cam_front"),
            "taluy/cameras/cam_bottom": CameraCalibration("cameras/cam_bottom"),
        }

        self.frame_id_to_camera_ns = {
            "taluy/base_link/bottom_camera_link": "taluy/cameras/cam_bottom",
            "taluy/base_link/front_camera_link": "taluy/cameras/cam_front",
        }

        self.camera_frames = {
            "taluy/cameras/cam_front": "taluy/base_link/front_camera_optical_link",
            "taluy/cameras/cam_bottom": "taluy/base_link/bottom_camera_optical_link",
        }

        self.id_tf_map = {
            "taluy/cameras/cam_front": {
                8: "red_buoy_link",
                7: "path_link",
                9: "bin_whole_link",
                12: "torpedo_map_link",
                13: "torpedo_hole_link",
                1: "gate_left_link",
                2: "gate_right_link",
                3: "gate_blue_arrow_link",
                4: "gate_red_arrow_link",
                5: "gate_middle_part_link",
                14: "octagon_link",
            },
            "taluy/cameras/cam_bottom": {
                9: "bin/whole",
                10: "bin/red_link",
                11: "bin/blue_link",
            },
        }

    def _initialize_props(self):
        """Initialize all known props with their properties"""
        self.props = {
            "red_buoy_link": BuoyRed(),
            "gate_red_arrow_link": GateRedArrow(),
            "gate_blue_arrow_link": GateBlueArrow(),
            "gate_middle_part_link": GateMiddlePart(),
            "torpedo_map_link": TorpedoMap(),
            "torpedo_hole_link": TorpedoHole(),
            "octagon_link": Octagon(),
            "bin/red_link": BinRed(),
            "bin/blue_link": BinBlue(),
        }

    def altitude_callback(self, msg: Odometry):
        """Callback for altitude/pressure sensor data"""
        depth = -msg.pose.pose.position.z
        self.altitude = self.pool_depth - depth
        rospy.loginfo_once(
            f"Calculated altitude from odom_pressure: {self.altitude:.2f} m (pool_depth={self.pool_depth})"
        )

    def calculate_intersection_with_ground(self, point1_odom, point2_odom):
        """Calculate where a line intersects with the ground plane (z=0)"""
        if point2_odom.point.z != point1_odom.point.z:
            t = -point1_odom.point.z / (point2_odom.point.z - point1_odom.point.z)

            if 0 <= t <= 1:
                # Calculate intersection point
                x = point1_odom.point.x + t * (
                    point2_odom.point.x - point1_odom.point.x
                )
                y = point1_odom.point.y + t * (
                    point2_odom.point.y - point1_odom.point.y
                )
                return x, y, 0  # ground plane
            else:
                rospy.logwarn("No intersection with ground plane within the segment.")
                return None
        else:
            rospy.logwarn("The line segment is parallel to the ground plane.")
            return None

    def process_altitude_projection(self, detection, camera_ns: str):
        """Special processing for altitude-based projections (like bins)"""
        if self.altitude is None:
            rospy.logwarn("No altitude data available")
            return

        detection_id = detection.results[0].id
        if detection_id != 9:  # Only process bin_whole
            return

        # Calculate bottom center of bounding box
        bbox_bottom_x = detection.bbox.center.x
        bbox_bottom_y = detection.bbox.center.y + detection.bbox.size_y * 0.5

        angles = self.camera_calibrations[camera_ns].calculate_angles(
            (bbox_bottom_x, bbox_bottom_y)
        )

        # Create two points to define a line
        point1 = PointStamped()
        point1.header.frame_id = self.camera_frames[camera_ns]
        point1.point.x = 0
        point1.point.y = 0
        point1.point.z = 0

        # Second point is along the projection line
        distance = 500.0  # Arbitrary large distance
        offset_x = math.tan(angles[0]) * distance
        offset_y = math.tan(angles[1]) * distance

        point2 = PointStamped()
        point2.header.frame_id = self.camera_frames[camera_ns]
        point2.point.x = offset_x
        point2.point.y = offset_y
        point2.point.z = distance

        try:
            # Transform points to odom frame
            transform = self.tf_buffer.lookup_transform(
                "odom",
                self.camera_frames[camera_ns],
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            point1_odom = tf2_geometry_msgs.do_transform_point(point1, transform)
            point2_odom = tf2_geometry_msgs.do_transform_point(point2, transform)

            # Adjust for current altitude
            point1_odom.point.z += self.altitude + 0.18
            point2_odom.point.z += self.altitude + 0.18

            # Find where this line intersects the ground
            intersection = self.calculate_intersection_with_ground(
                point1_odom, point2_odom
            )
            if intersection:
                x, y, z = intersection
                self._publish_transform(
                    detection_msg=detection,
                    camera_frame="odom",
                    child_frame=self.id_tf_map[camera_ns][detection_id],
                    position=(x, y, z),
                    orientation=(0, 0, 0, 1),
                )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Transform error: {e}")

    def check_if_detection_is_inside_image(
        self, detection, image_width: int = 640, image_height: int = 480
    ) -> bool:
        """Check if detection is too close to image edges"""
        center = detection.bbox.center
        half_size_x = detection.bbox.size_x * 0.5
        half_size_y = detection.bbox.size_y * 0.5
        deadzone = 5  # pixels

        return not (
            (center.x + half_size_x >= image_width - deadzone)
            or (center.x - half_size_x <= deadzone)
            or (center.y + half_size_y >= image_height - deadzone)
            or (center.y - half_size_y <= deadzone)
        )

    def _publish_transform(
        self, detection_msg, camera_frame, child_frame, position, orientation
    ):
        """Helper to publish transform messages"""
        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = detection_msg.header.stamp
        transform_stamped_msg.header.frame_id = camera_frame
        transform_stamped_msg.child_frame_id = child_frame
        transform_stamped_msg.transform.translation = Vector3(*position)
        transform_stamped_msg.transform.rotation = Quaternion(*orientation)
        self.object_transform_pub.publish(transform_stamped_msg)

    def detection_callback(self, detection_msg: YoloResult, camera_source: str):
        """Main callback for processing detections from either camera"""
        # Determine camera namespace based on source
        if camera_source == "front_camera":
            camera_ns = "taluy/cameras/cam_front"
        elif camera_source == "bottom_camera":
            camera_ns = "taluy/cameras/cam_bottom"
        else:
            rospy.logerr(f"Unknown camera_source: {camera_source}")
            return

        camera_frame = self.camera_frames[camera_ns]

        # Pre-calculate distances for all detections
        detection_distances = self._precalculate_distances(detection_msg, camera_ns)

        # Process each detection
        for detection in detection_msg.detections.detections:
            if not detection.results:
                continue

            detection_id = detection.results[0].id
            if detection_id not in self.id_tf_map[camera_ns]:
                continue

            # Handle special cases
            if detection_id in [9, 10, 11, 13]:
                self._handle_special_detections(
                    detection, camera_ns, detection_distances
                )
                continue

            # Skip detections at image edges
            if not self.check_if_detection_is_inside_image(detection):
                continue

            # Get prop information
            prop_name = self.id_tf_map[camera_ns][detection_id]
            if prop_name not in self.props:
                continue

            # Get or calculate distance
            distance = detection_distances.get(detection_id)
            if distance is None:
                continue

            # Calculate angles and position
            angles = self.camera_calibrations[camera_ns].calculate_angles(
                (detection.bbox.center.x, detection.bbox.center.y)
            )

            # Calculate position offset
            offset_x = math.tan(angles[0]) * distance
            offset_y = math.tan(angles[1]) * distance

            # Get camera orientation in odom frame
            try:
                camera_to_odom_transform = self.tf_buffer.lookup_transform(
                    camera_frame,
                    "odom",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                )
                orientation = (
                    camera_to_odom_transform.transform.rotation.x,
                    camera_to_odom_transform.transform.rotation.y,
                    camera_to_odom_transform.transform.rotation.z,
                    camera_to_odom_transform.transform.rotation.w,
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logerr(f"Transform error: {e}")
                continue

            # Publish transform
            self._publish_transform(
                detection_msg=detection,
                camera_frame=camera_frame,
                child_frame=prop_name,
                position=(offset_x, offset_y, distance),
                orientation=orientation,
            )

    def _precalculate_distances(self, detection_msg, camera_ns):
        """Pre-calculate distances for all detections to optimize processing"""
        distances = {}
        for detection in detection_msg.detections.detections:
            if not detection.results:
                continue

            detection_id = detection.results[0].id
            if detection_id not in self.id_tf_map[camera_ns]:
                continue

            # Skip special cases that don't use normal distance calculation
            if detection_id in [9, 10, 11, 13]:
                continue

            prop_name = self.id_tf_map[camera_ns][detection_id]
            if prop_name in self.props:
                prop = self.props[prop_name]
                distance = prop.estimate_distance(
                    detection.bbox.size_y,
                    detection.bbox.size_x,
                    self.camera_calibrations[camera_ns],
                )
                distances[detection_id] = distance

        return distances

    def _handle_special_detections(self, detection, camera_ns, detection_distances):
        """Handle special cases like bins and torpedo holes"""
        detection_id = detection.results[0].id

        if detection_id == 9:  # bin_whole
            self.process_altitude_projection(detection, camera_ns)
        elif detection_id in [10, 11]:  # bin_red or bin_blue
            if self.altitude is not None:
                self._process_bin_detection(detection, camera_ns)
        elif detection_id == 13:  # torpedo_hole
            self._process_torpedo_hole(detection, camera_ns, detection_distances)

    def _process_bin_detection(self, detection, camera_ns):
        """Process bin detections using altitude"""
        detection_id = detection.results[0].id
        prop_name = self.id_tf_map[camera_ns][detection_id]

        angles = self.camera_calibrations[camera_ns].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        offset_x = math.tan(angles[0]) * self.altitude
        offset_y = math.tan(angles[1]) * self.altitude

        self._publish_transform(
            detection_msg=detection,
            camera_frame=self.camera_frames[camera_ns],
            child_frame=prop_name,
            position=(offset_x, offset_y, self.altitude),
            orientation=(0, 0, 0, 1),
        )

    def _process_torpedo_hole(
        self, detection, camera_ns, detection_distances, detection_msg=None
    ):
        """Process torpedo hole detections using torpedo map's distance but own angular position"""
        if 12 not in detection_distances:  # torpedo_map ID
            rospy.logwarn(
                "No torpedo_map detection found for torpedo_hole distance reference"
            )
            return

        # Get the distance from the torpedo map (ID 12)
        distance = detection_distances[12]

        # Calculate angles based on this torpedo hole's pixel position
        angles = self.camera_calibrations[camera_ns].calculate_angles(
            (detection.bbox.center.x, detection.bbox.center.y)
        )

        # Calculate offsets using torpedo map's distance but this hole's angles
        offset_x = math.tan(angles[0]) * distance
        offset_y = math.tan(angles[1]) * distance

        # Create unique frame name
        torpedo_hole_index = 0
        if detection_msg is not None:
            torpedo_hole_index = sum(
                1
                for d in detection_msg.detections.detections
                if d.results and d.results[0].id == 13 and d != detection
            )
        prop_name = f"torpedo_hole_{torpedo_hole_index}_link"

        # Publish transform
        self._publish_transform(
            detection_msg=detection,
            camera_frame=self.camera_frames[camera_ns],
            child_frame=prop_name,
            position=(offset_x, offset_y, distance),
            orientation=(0, 0, 0, 1),
        )

    def run(self):
        """Main run loop"""
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
