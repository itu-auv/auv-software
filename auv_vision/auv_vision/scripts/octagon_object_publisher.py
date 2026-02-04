#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import (
    PointStamped,
    TransformStamped,
    Vector3,
    Quaternion,
    PoseStamped,
)
from std_srvs.srv import SetBool, SetBoolResponse
from ultralytics_ros.msg import YoloResult
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import auv_common_lib.vision.camera_calibrations as camera_calibrations
import tf2_ros
import tf2_geometry_msgs
from tf import transformations as tf_transformations

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

        if self.real_height is not None and measured_height is not None:
            distance_from_height = calibration.distance_from_height(
                self.real_height, measured_height
            )

        if self.real_width is not None and measured_width is not None:
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
            # rospy.logerr(f"Could not estimate distance for prop {self.name}")
            return None


class Bottle(Prop):
    def __init__(self):
        super().__init__(0, "bottle", None, 0.09)


class OctagonObjectPublisher:
    def __init__(self):
        rospy.init_node("octagon_object_publisher", anonymous=True)
        rospy.loginfo("Octagon Object Publisher node started")
        
        self.bottom_camera_enabled = False 


        self.bottle_angle = None  # Angle of bottle relative to base_link
        self.current_yaw = 0.0  # Current yaw of base_link in odom frame
        self.bottle_thickness_px = None  # Pixel width from bottle_angle_node

        self.object_transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )

        # Initialize tf2 buffer and listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.camera_calibrations = {
            "taluy/cameras/cam_bottom": CameraCalibration("cameras/cam_bottom"),
        }
        # Segmentation source uses the same camera calibration as bottom camera
        self.camera_calibrations["taluy/cameras/cam_bottom_seg"] = (
            self.camera_calibrations["taluy/cameras/cam_bottom"]
        )
        print("CAM_CALİB_LOADED")
        rospy.Subscriber(
            "/yolo_result_bottom",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="bottom_camera"),
            queue_size=1,
        )
        rospy.Subscriber(
            "/yolo_result_seg",
            YoloResult,
            lambda msg: self.detection_callback(msg, camera_source="bottom_camera_seg"),
             queue_size=1,
        )

        # Services to enable/disable detections
        rospy.Service(
            "enable_bottom_camera_detections",
            SetBool,
            self.handle_enable_bottom_camera,
        )

        self.frame_id_to_camera_ns = {
            "taluy/base_link/bottom_camera_link": "taluy/cameras/cam_bottom",
        }
        self.camera_frames = {
            "taluy/cameras/cam_bottom": "taluy/base_link/bottom_camera_optical_link",
            "taluy/cameras/cam_bottom_seg": "taluy/base_link/bottom_camera_optical_link",
        }

        self.props = {
            "bottle_link": Bottle(),
        }

        self.id_tf_map = {
            "taluy/cameras/cam_bottom": {
                0: "bottle_link",
            },
            "taluy/cameras/cam_bottom_seg": {
                0: "bottle_link",
            },
        }

        self.altitude = None
        self.pool_depth = rospy.get_param("/env/pool_depth", 2.2) # Default to 2.2 if not set
        rospy.Subscriber("odom_pressure", Odometry, self.altitude_callback)

        # Subscribe to bottle angle and thickness
        rospy.Subscriber(
            "bottle_angle", Float32, self.bottle_angle_callback, queue_size=1
        )
        rospy.Subscriber(
            "bottle_thickness", Float32, self.bottle_thickness_callback, queue_size=1
        )

    def handle_enable_bottom_camera(self, req):
        self.bottom_camera_enabled = req.data
        message = "Bottom camera DETECTIONS " + ("ENABLED" if req.data else "DISABLED")
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def altitude_callback(self, msg: Odometry):
        depth = -msg.pose.pose.position.z
        self.altitude = self.pool_depth - depth

    def bottle_thickness_callback(self, msg: Float32):
        if not math.isnan(msg.data) and msg.data > 0:
            self.bottle_thickness_px = msg.data
            rospy.logdebug(
                f"Updated bottle thickness: {self.bottle_thickness_px:.1f}px"
            )

    def bottle_angle_callback(self, msg: Float32):
        print("BOTTLE ANGLE RECEIVED", msg.data)
        if not math.isnan(msg.data):
            self.bottle_angle = -msg.data  # Store in base_link frame

            # Get current yaw from odom to base_link transform
            try:
                # Get transform from odom to base_link
                transform = self.tf_buffer.lookup_transform(
                    "odom", "taluy/base_link", rospy.Time(0), rospy.Duration(1.0)
                )
                # Extract yaw from quaternion
                _, _, self.current_yaw = tf_transformations.euler_from_quaternion(
                    [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ]
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn(f"Could not get transform from odom to base_link: {e}")
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
        print("detection_recived")

        if camera_source == "bottom_camera":
            camera_ns = "taluy/cameras/cam_bottom"
        elif camera_source == "bottom_camera_seg":
            camera_ns = "taluy/cameras/cam_bottom_seg"
        else:
            print("Unknown camera source:", camera_source)
            return

        camera_frame = self.camera_frames[camera_ns]
        try:
             self.tf_buffer.lookup_transform(
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
            print(f"Transform error: {e}")
            return

        print(f"Processing {len(detection_msg.detections.detections)} detections")
        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue
            
            detection_id = detection.results[0].id
            print(f"Detection ID: {detection_id}")
            
            # We only care about Bottle (ID 0)
            if detection_id != 0:
                continue

            prop_name = self.id_tf_map[camera_ns].get(detection_id)
            if not prop_name or prop_name not in self.props:
                print(f"Prop name not found for ID {detection_id}")
                continue
            
            prop = self.props[prop_name]

            # Logic for Bottle Distance using thickness
            distance = None
            skip_inside_image = False    

            if (
                self.bottle_thickness_px is not None
                and self.bottle_thickness_px > 0
            ):
                 # Use the bottle_thickness_px as the width in pixels
                distance = prop.estimate_distance(
                    None,
                    self.bottle_thickness_px,
                    self.camera_calibrations[camera_ns],
                )
                print(f"Calculated distance from thickness: {distance}")
                rospy.logdebug(
                     f"Using bottle_thickness_px: {self.bottle_thickness_px}px for distance calculation"
                )
            else:
                 # Fallback to using bbox width
                distance = prop.estimate_distance(
                    detection.bbox.size_y,  # height
                    detection.bbox.size_x,  # width
                    self.camera_calibrations[camera_ns],
                )
                print(f"Calculated distance from bbox: {distance}")
            
            if distance is None:
                rospy.logwarn_throttle(5, "Could not calculate bottle distance from pixel width, using altitude")
                distance = self.altitude
                print(f"Using altitude fallback: {distance}")

            if distance is None:
                print("Distance is None, skipping")
                continue

            # Calculate offsets
            angles = self.camera_calibrations[camera_ns].calculate_angles(
                (detection.bbox.center.x, detection.bbox.center.y)
            )

            offset_x = math.tan(angles[0]) * distance * 1.0
            offset_y = math.tan(angles[1]) * distance * 1.0

            # Create TransformStamped
            transform_stamped_msg = TransformStamped()
            transform_stamped_msg.header.stamp = detection_msg.header.stamp
            transform_stamped_msg.header.frame_id = camera_frame
            transform_stamped_msg.child_frame_id = prop_name

            transform_stamped_msg.transform.translation = Vector3(
                offset_x, offset_y, distance
            )

            # Apply Bottle Rotation
            if self.bottle_angle is not None:
                 # Calculate angle in odom frame: bottle_angle_odom = current_yaw + bottle_angle_base_link
                bottle_angle_odom = self.current_yaw + self.bottle_angle
                quat = tf_transformations.quaternion_from_euler(0, 0, bottle_angle_odom)
                transform_stamped_msg.transform.rotation = Quaternion(*quat)
                print("Applied bottle angle rotation")
            else:
                 transform_stamped_msg.transform.rotation = Quaternion(0, 0, 0, 1)
                 print("Applied default rotation (no bottle angle)")

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
                final_transform_stamped.child_frame_id = prop_name
                final_transform_stamped.transform.translation = (
                    transformed_pose_stamped.pose.position
                )
                final_transform_stamped.transform.rotation = (
                    transform_stamped_msg.transform.rotation
                )

                self.object_transform_pub.publish(final_transform_stamped)
                print("PUBLISHED TRANSFORM")
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logerr(f"Transform error for {prop_name}: {e}")
                print(f"Final transform error: {e}")


if __name__ == "__main__":
    try:
        node = OctagonObjectPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
