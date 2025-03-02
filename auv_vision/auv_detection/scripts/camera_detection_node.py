#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, Pose
from ultralytics_ros.msg import YoloResult
from sensor_msgs.msg import Range
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


class CameraDetectionNode:
    def __init__(self):
        rospy.init_node("camera_detection_node", anonymous=True)
        rospy.loginfo("Camera detection node started")
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
            "octagon": Octagon(),
        }

        self.id_tf_map = {
            "taluy/cameras/cam_front": {
                8: "red_buoy",
                4: "gate_red_arrow",
                3: "gate_blue_arrow",
                5: "gate_middle_part",
                12: "torpedo_map",
                14: "octagon",
            },
            "taluy/cameras/cam_bottom": {
                9: "bin/whole",
                10: "bin/red",
                11: "bin/blue"
            },
        }

        # Initialize tf2 buffer and listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers for detected object coordinates
        self.detection_pubs = {}
        for camera in self.id_tf_map:
            for obj_id in self.id_tf_map[camera]:
                topic = f"/detection/{obj_id}/point"
                self.detection_pubs[obj_id] = rospy.Publisher(topic, PointStamped, queue_size=10)

        # Publishers for detected object lines
        self.detection_line_pubs = {}
        for camera in self.id_tf_map:
            for obj_id in self.id_tf_map[camera]:
                topic = f"/detection/{obj_id}/line"
                self.detection_line_pubs[obj_id] = rospy.Publisher(topic, PoseArray, queue_size=10)

        # Subscribe to YOLO detections
        rospy.Subscriber("/yolo_result", YoloResult, self.detection_callback)
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

    def detection_callback(self, detection_msg: YoloResult):
        camera_ns = "taluy/cameras/cam_front"  
        if camera_ns not in self.camera_calibrations:
            rospy.logwarn(f"Unknown camera namespace: {camera_ns}")
            return

        calibration = self.camera_calibrations[camera_ns]
        camera_frame = self.camera_frames[camera_ns]

        for detection in detection_msg.detections.detections:
            if len(detection.results) == 0:
                continue

            detection_id = detection.results[0].id
            if detection_id not in self.id_tf_map[camera_ns]:
                continue

            prop_name = self.id_tf_map[camera_ns][detection_id]
            if prop_name not in self.props:
                continue

            prop = self.props[prop_name]
            if self.check_if_detection_is_inside_image(detection) == False:
                continue
            # Calculate distance using object dimensions
            distance = prop.estimate_distance(
                detection.bbox.size_y,
                detection.bbox.size_x,
                self.camera_calibrations[camera_ns]
            )

            if distance is None:
                continue

            # Calculate angles from pixel coordinates
            angles = self.camera_calibrations[camera_ns].calculate_angles(
                (detection.bbox.center.x, detection.bbox.center.y)
            )

            # Calculate the offset using the same logic as prop_transform
            offset_x = math.tan(angles[0]) * distance * 1.0
            offset_y = math.tan(angles[1]) * distance * 1.0

            # Create point message
            point_msg = PointStamped()
            point_msg.header.frame_id = camera_frame
            point_msg.header.stamp = rospy.Time.now()
            point_msg.point.x = offset_x
            point_msg.point.y = offset_y
            point_msg.point.z = distance

            # Publish point message
            if detection_id in self.detection_pubs:
                self.detection_pubs[detection_id].publish(point_msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
