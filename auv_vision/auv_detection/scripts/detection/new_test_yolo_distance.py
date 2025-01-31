#!/usr/bin/env python3


import rospy
import sys
import os
import traceback
import time
import numpy as np
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
# Add the path to the auv_control package
sys.path.append('/home/frk/catkin_ws/src/auv-software/auv_control/auv_control/scripts')
sys.path.append('/home/frk/catkin_ws/src/auv-software/auv_common_lib/python')

from ultralytics_ros.msg import YoloResult
from std_msgs.msg import Float32
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from prop_transform_publisher import TorpedoMap, name_to_id_map, GateBlueArrow
from sensor_msgs.msg import Image


class CustomCameraCalibration:
    def __init__(self, camera_namespace):
        # Actual camera calibration values from cameras.yaml
        self.camera_calibrations = {
            'cam_front': {
                'K': [
                    602.090661,  # fx
                    0.0,         # skew
                    300.333850,  # cx
                    0.0,         # 0
                    594.250620,  # fy
                    222.590416,  # cy
                    0.0, 0.0, 1.0  # Last row of camera matrix
                ]
            }
        }
        
        # Determine camera type
        self.camera_type = 'cam_front'
        
        # Fallback to front camera if not recognized
        if self.camera_type not in self.camera_calibrations:
            print(f"WARNING: Unknown camera type {self.camera_type}. Using front camera calibration.")
            self.camera_type = 'cam_front'
        
        # Try to fetch actual calibration
        try:
            # First, try ROS camera info
            calibration_fetcher = CameraCalibrationFetcher(camera_namespace)
            camera_info = calibration_fetcher.get_camera_info()
            
            if camera_info and any(camera_info.K):
                print(f"Using ROS camera calibration for {self.camera_type}")
                self.calibration = camera_info
            else:
                print(f"ROS camera calibration empty. Using {self.camera_type} fallback.")
                self.calibration = type('CalibrationData', (), {
                    'K': self.camera_calibrations[self.camera_type]['K']
                })()
        except Exception as e:
            print(f"Error fetching camera calibration: {e}")
            print(f"Using {self.camera_type} fallback calibration")
            self.calibration = type('CalibrationData', (), {
                'K': self.camera_calibrations[self.camera_type]['K']
            })()
        
        # Validate calibration
        self.validate_calibration()
    
    def validate_calibration(self):
        # Print and validate calibration details
        print("Camera Calibration Details:")
        print(f"Camera Type: {self.camera_type}")
        print(f"Camera Matrix (K): {self.calibration.K}")
        
        # Validate focal lengths
        fx = self.calibration.K[0]  # Focal length x
        fy = self.calibration.K[4]  # Focal length y
        
        if fx == 0 or fy == 0:
            print("WARNING: Focal lengths are zero. Using default values.")
            default_calib = self.camera_calibrations[self.camera_type]
            self.calibration.K[0] = default_calib['K'][0]  # fx
            self.calibration.K[4] = default_calib['K'][4]  # fy
        
        print(f"Focal Length (x): {self.calibration.K[0]}")
        print(f"Focal Length (y): {self.calibration.K[4]}")
    
    def distance_from_height(self, real_height: float, measured_height: float) -> float:
        if measured_height == 0:
            raise ValueError("Measured height cannot be zero")
        
        focal_length = self.calibration.K[4]
        distance = (real_height * focal_length) / measured_height
        return distance
    
    def distance_from_width(self, real_width: float, measured_width: float) -> float:
        if measured_width == 0:
            raise ValueError("Measured width cannot be zero")
        
        focal_length = self.calibration.K[0]
        distance = (real_width * focal_length) / measured_width
        return distance

    def calculate_angles(self, pixel_coordinates: tuple) -> tuple:
        """
        Calculate angular coordinates from pixel coordinates.
        
        Args:
            pixel_coordinates (tuple): (x, y) pixel coordinates in the image
        
        Returns:
            tuple: (angle_x, angle_y) angular coordinates in radians
        """
        # Extract camera matrix parameters
        fx = self.calibration.K[0]  # Focal length in x direction
        fy = self.calibration.K[4]  # Focal length in y direction
        cx = self.calibration.K[2]  # Principal point x-coordinate
        cy = self.calibration.K[5]  # Principal point y-coordinate
        
        # Normalize pixel coordinates relative to the camera's optical center
        norm_x = (pixel_coordinates[0] - cx) / fx
        norm_y = (pixel_coordinates[1] - cy) / fy
        
        # Convert to angular coordinates
        angle_x = math.atan(norm_x)
        angle_y = math.atan(norm_y)
        
        return angle_x, angle_y
    
    def calculate_offset_and_distance(self, pixel_coordinates: tuple, distance: float) -> tuple:
        """
        Calculate offset and position based on pixel coordinates and distance.
        
        Args:
            pixel_coordinates (tuple): (x, y) pixel coordinates in the image
            distance (float): Estimated distance to the object
        
        Returns:
            tuple: (offset_x, offset_y, x, y, z)
        """
        # Calculate angles
        angle_x, angle_y = self.calculate_angles(pixel_coordinates)
        
        # Calculate offsets
        offset_x = math.tan(angle_x) * distance
        offset_y = math.tan(angle_y) * distance
        
        # Calculate 3D coordinates
        x = offset_x
        y = offset_y
        z = distance
        
        return offset_x, offset_y, x, y, z


class TorpedoMapDistanceNode:
    def __init__(self):
        try:
            print("Starting Torpedo Map Distance Node initialization...")
            rospy.init_node('torpedo_map_distance_node', disable_signals=True)
            
            # Camera namespace for usb_cam
            camera_namespace = 'taluy/cameras/cam_front'
            
            print(f"Initializing with camera namespace: {camera_namespace}")
            
            # Use custom camera calibration with fallback mechanism
            self.camera_calibration = CustomCameraCalibration(camera_namespace)
            
            # Initialize torpedo map prop
            self.torpedo_map_prop = TorpedoMap()
            
            # Publisher for torpedo map distance
            self.distance_pub = rospy.Publisher('/torpedo_map/distance', Float32, queue_size=10)
            
            # Subscriber to YOLO results
            self.yolo_sub = rospy.Subscriber('/yolo_result', YoloResult, self.yolo_callback)
            
            # Subscriber to usb_cam image
            self.image_sub = rospy.Subscriber('/taluy/cameras/cam_front/image_raw', Image, self.image_callback)
            
            # CV Bridge
            self.bridge = CvBridge()
            self.cv_image = None  # Initialize cv_image
            self.image_received = False  # Flag to check if image is received
            
            print("Subscriber created. Waiting for messages...")
            
            # Add a timer to check if node is alive
            self.alive_timer = rospy.Timer(rospy.Duration(5), self.check_alive)
            
            print("Node initialization complete.")
            rospy.loginfo("Torpedo Map Distance Node initialized successfully")
        
        except Exception as e:
            print(f"Initialization error: {e}")
            print(traceback.format_exc())
            rospy.logerr(f"Node initialization failed: {e}")
    
    def check_alive(self, event):
        print("Node is still running. Waiting for YOLO results...")
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received = True  # Set flag to true when image is received
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
    
    def yolo_callback(self, msg):
        try:
            print(f"Received YOLO message with {len(msg.detections.detections)} detections.")
            # Check if there are any detections
            if not msg.detections.detections:
                print("No detections found.")
                return

            # Find torpedo map detection
            torpedo_map_id = name_to_id_map.get('torpedo_map', 12)  # Fallback to 12 if not found
            torpedo_map_detection = None

            for detection in msg.detections.detections:
                print(f"Detection ID: {detection.results[0].id}")
                if detection.results[0].id == torpedo_map_id:
                    print("Torpedo map detection found.")
                    torpedo_map_detection = detection
                    break

            if not torpedo_map_detection:
                return

            # Get bounding box details
            bbox_center_x = torpedo_map_detection.bbox.center.x
            bbox_center_y = torpedo_map_detection.bbox.center.y
            measured_height = torpedo_map_detection.bbox.size_y
            measured_width = torpedo_map_detection.bbox.size_x

            # Check for valid measurements
            if measured_height == 0 or measured_width == 0:
                rospy.logerr("Invalid measurement: height or width is zero")
                return
            
            try:
                # Calculate distance using height and width
                distance_from_height = self.camera_calibration.distance_from_height(
                    self.torpedo_map_prop.real_height, measured_height
                )
                distance_from_width = self.camera_calibration.distance_from_width(
                    self.torpedo_map_prop.real_width, measured_width
                )

                # Average the distances
                distance = (distance_from_height + distance_from_width) / 2.0

                # Calculate angles and offsets
                offset_x, offset_y, x, y, z = self.camera_calibration.calculate_offset_and_distance(
                    (bbox_center_x, bbox_center_y), distance
                )

                # Print detailed information
                print("\n--- Torpedo Map Distance Calculation ---")
                print(f"Bounding Box Center: ({bbox_center_x}, {bbox_center_y})")
                print(f"Real Height: {self.torpedo_map_prop.real_height} m")
                print(f"Real Width: {self.torpedo_map_prop.real_width} m")
                print(f"Measured Height: {measured_height} pixels")
                print(f"Measured Width: {measured_width} pixels")
                print(f"Distance from Height: {distance_from_height:.2f} m")
                print(f"Distance from Width: {distance_from_width:.2f} m")
                print(f"Estimated Distance: {distance:.2f} m")
                
                # Angle calculations
                angle_x, angle_y = self.camera_calibration.calculate_angles((bbox_center_x, bbox_center_y))
                print(f"Angle X: {math.degrees(angle_x):.2f}°")
                print(f"Angle Y: {math.degrees(angle_y):.2f}°")
                
                print(f"Offset X: {offset_x:.2f} m")
                print(f"Offset Y: {offset_y:.2f} m")
                print(f"3D Coordinates (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) m")

                # Publish the distance
                distance_msg = Float32()
                distance_msg.data = distance
                self.distance_pub.publish(distance_msg)

            except ValueError as ve:
                rospy.logerr(f"Distance calculation error: {ve}")

        except Exception as e:
            rospy.logerr(f"Error in yolo_callback: {e}")
            print(traceback.format_exc())
### END OF FIRST NODE 

class GateDistanceNode:
    def __init__(self):
        try:
            print("Starting GATEARROW Distance Node initialization...")
            #rospy.init_node('gate_blue_arrow_distance_node', disable_signals=True)
            
            # Camera namespace for usb_cam
            camera_namespace = 'taluy/cameras/cam_front'
            
            print(f"Initializing with camera namespace: {camera_namespace}")
            
            # Use custom camera calibration with fallback mechanism
            self.camera_calibration = CustomCameraCalibration(camera_namespace)
            
            # Initialize torpedo map prop
            self.gate_blue_arrow_prop = GateBlueArrow()
            
            # Publisher for torpedo map distance
            self.distance_pub = rospy.Publisher('/gate_bluearrow/distance', Float32, queue_size=10)
            
            # Subscriber to YOLO results
            self.yolo_sub = rospy.Subscriber('/yolo_result', YoloResult, self.yolo_callback)
            
            print("Subscriber created. Waiting for messages...")
            
            # Add a timer to check if node is alive
            self.alive_timer = rospy.Timer(rospy.Duration(5), self.check_alive)
            
            print("Node initialization complete.")
            rospy.loginfo("Gate Distance initialized successfully")
        
        except Exception as e:
            print(f"Initialization error: {e}")
            print(traceback.format_exc())
            rospy.logerr(f"Node initialization failed: {e}")
    
    def check_alive(self, event):
        print("GATENode is still running. Waiting for YOLO results...")
    
    def yolo_callback(self, msg):
        try:
            print(f"Received YOLO message with {len(msg.detections.detections)} detections.")
            # Check if there are any detections
            if not msg.detections.detections:
                print("No detections found.")
                return

            # Find torpedo map detection
            gate_blue_arrow_id = name_to_id_map.get('gate_blue_arrow', 3)  # Fallback to 12 if not found
            print(f"Searching for gate blue arrow with ID: {gate_blue_arrow_id}")
            gate_blue_arrow_detection = None

            for detection in msg.detections.detections:
                print(f"Detection ID: {detection.results[0].id}")
                if detection.results[0].id == gate_blue_arrow_id:
                    print("Gate blue arrow detection found.")
                    gate_blue_arrow_detection = detection
                    break

            if not gate_blue_arrow_detection:
                return

            # Get bounding box details
            bbox_center_x = gate_blue_arrow_detection.bbox.center.x
            bbox_center_y = gate_blue_arrow_detection.bbox.center.y
            measured_height = gate_blue_arrow_detection.bbox.size_y
            measured_width = gate_blue_arrow_detection.bbox.size_x

            # Check for valid measurements
            if measured_height == 0 or measured_width == 0:
                rospy.logerr("Invalid measurement: height or width is zero")
                return

            try:
                # Calculate distance using height and width
                distance_from_height = self.camera_calibration.distance_from_height(
                    self.gate_blue_arrow_prop.real_height, measured_height
                )
                distance_from_width = self.camera_calibration.distance_from_width(
                    self.gate_blue_arrow_prop.real_width, measured_width
                )

                # Average the distances
                distance = (distance_from_height + distance_from_width) / 2.0

                # Calculate angles and offsets
                offset_x, offset_y, x, y, z = self.camera_calibration.calculate_offset_and_distance(
                    (bbox_center_x, bbox_center_y), distance
                )

                # Print detailed information
                print("\n--- Gate_Blue_Arrow Distance Calculation ---")
                print(f"Bounding Box Center: ({bbox_center_x}, {bbox_center_y})")
                print(f"Real Height: {self.gate_blue_arrow_prop.real_height} m")
                print(f"Real Width: {self.gate_blue_arrow_prop.real_width} m")
                print(f"Measured Height: {measured_height} pixels")
                print(f"Measured Width: {measured_width} pixels")
                print(f"Distance from Height: {distance_from_height:.2f} m")
                print(f"Distance from Width: {distance_from_width:.2f} m")
                print(f"Estimated Distance: {distance:.2f} m")
                
                # Angle calculations
                angle_x, angle_y = self.camera_calibration.calculate_angles((bbox_center_x, bbox_center_y))
                print(f"Angle X: {math.degrees(angle_x):.2f}°")
                print(f"Angle Y: {math.degrees(angle_y):.2f}°")
                
                print(f"Offset X: {offset_x:.2f} m")
                print(f"Offset Y: {offset_y:.2f} m")
                print(f"3D Coordinates (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) m")

                # Publish the distance
                distance_msg = Float32()
                distance_msg.data = distance
                self.distance_pub.publish(distance_msg)

            except ValueError as ve:
                rospy.logerr(f"Distance calculation error: {ve}")
            
        except Exception as e:
            rospy.logerr(f"Error in yolo_callback: {e}")
            print(traceback.format_exc())



if __name__ == '__main__':
    try:
        node = TorpedoMapDistanceNode()
        node2 = GateDistanceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass