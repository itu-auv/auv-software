#!/usr/bin/env python3

import rospy
import sys
import os
import traceback
import time
import numpy as np

# Add the path to the auv_control package
sys.path.append('/home/frk/catkin_ws/src/auv-software/auv_control/auv_control/scripts')
sys.path.append('/home/frk/catkin_ws/src/auv-software/auv_common_lib/python')

from ultralytics_ros.msg import YoloResult
from std_msgs.msg import Float32
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from prop_transform_publisher import TorpedoMap, name_to_id_map


class CustomCameraCalibration:
    def __init__(self, camera_namespace):
        # Actual camera calibration values from cameras.yaml
        self.camera_calibrations = {
            'cam_front': {
                'K': [
                    432.803545,  # fx
                    0.0,         # skew
                    306.698087,  # cx
                    0.0,         # 0
                    567.001237,  # fy
                    221.133655,  # cy
                    0.0, 0.0, 1.0  # Last row of camera matrix
                ],
                'D': [-0.002190, -0.089089, -0.013968, -0.006001, 0.0]  # Distortion coefficients
            },
            'cam_bottom': {
                'K': [
                    662.777153,  # fx
                    0.0,         # skew
                    316.454652,  # cx
                    0.0,         # 0
                    871.683826,  # fy
                    249.195884,  # cy
                    0.0, 0.0, 1.0  # Last row of camera matrix
                ],
                'D': [0.666467, -0.503379, 0.0089, -0.008196, 0.0]  # Distortion coefficients
            }
        }
        
        # Determine camera type
        self.camera_type = camera_namespace.split('/')[-1] if '/' in camera_namespace else camera_namespace
        
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
                    'K': self.camera_calibrations[self.camera_type]['K'],
                    'D': self.camera_calibrations[self.camera_type]['D']
                })()
        except Exception as e:
            print(f"Error fetching camera calibration: {e}")
            print(f"Using {self.camera_type} fallback calibration")
            self.calibration = type('CalibrationData', (), {
                'K': self.camera_calibrations[self.camera_type]['K'],
                'D': self.camera_calibrations[self.camera_type]['D']
            })()
        
        # Validate calibration
        self.validate_calibration()
    
    def validate_calibration(self):
        # Print and validate calibration details
        print("Camera Calibration Details:")
        print(f"Camera Type: {self.camera_type}")
        print(f"Camera Matrix (K): {self.calibration.K}")
        print(f"Distortion Coefficients: {self.calibration.D}")
        
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


class TorpedoMapDistanceNode:
    def __init__(self):
        try:
            print("Starting Torpedo Map Distance Node initialization...")
            rospy.init_node('torpedo_map_distance_node', disable_signals=True)
            
            # Camera namespace for usb_cam
            camera_namespace = 'usb_cam'
            
            print(f"Initializing with camera namespace: {camera_namespace}")
            
            # Use custom camera calibration with fallback mechanism
            self.camera_calibration = CustomCameraCalibration(camera_namespace)
            
            # Initialize torpedo map prop
            self.torpedo_map_prop = TorpedoMap()
            
            # Publisher for torpedo map distance
            self.distance_pub = rospy.Publisher('/torpedo_map/distance', Float32, queue_size=10)
            
            # Subscriber to YOLO results
            self.yolo_sub = rospy.Subscriber('/yolo_result', YoloResult, self.yolo_callback)
            
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
    
    def yolo_callback(self, msg):
        try:
            # Add a small delay to prevent overwhelming processing
            time.sleep(0.1)
            
            print(f"Received YOLO result with {len(msg.detections.detections)} detections")
            
            # Find torpedo map detection
            torpedo_map_detection = None
            torpedo_map_id = name_to_id_map.get('torpedo_map', 12)  # Fallback to 12 if not found
            
            for detection in msg.detections.detections:
                # Print out all detection details for debugging
                print(f"Detection ID: {detection.results[0].id}")
                print(f"Detection Confidence: {detection.results[0].score}")
                print(f"Bounding Box: x={detection.bbox.center.x}, y={detection.bbox.center.y}")
                print(f"Bounding Box Size: width={detection.bbox.size_x}, height={detection.bbox.size_y}")
                
                if detection.results[0].id == torpedo_map_id:
                    torpedo_map_detection = detection
                    break
            
            if torpedo_map_detection:
                # Calculate distance using height and width
                measured_height = torpedo_map_detection.bbox.size_y
                measured_width = torpedo_map_detection.bbox.size_x
                
                print(f"Torpedo map detection - Height: {measured_height}, Width: {measured_width}")
                
                # Add more detailed debugging for distance estimation
                try:
                    # Detailed logging of distance calculation parameters
                    print(f"Real Height: {self.torpedo_map_prop.real_height}")
                    print(f"Real Width: {self.torpedo_map_prop.real_width}")
                    print(f"Focal Length (x): {self.camera_calibration.calibration.K[0]}")
                    print(f"Focal Length (y): {self.camera_calibration.calibration.K[4]}")
                    
                    # Sanity checks
                    if measured_height == 0 or measured_width == 0:
                        rospy.logerr("Invalid measurement: height or width is zero")
                        return
                    
                    # Manually calculate distance using both height and width
                    try:
                        distance_from_height = self.camera_calibration.distance_from_height(
                            self.torpedo_map_prop.real_height, measured_height
                        )
                        distance_from_width = self.camera_calibration.distance_from_width(
                            self.torpedo_map_prop.real_width, measured_width
                        )
                    except ValueError as ve:
                        rospy.logerr(f"Distance calculation error: {ve}")
                        return
                    
                    print(f"Manually calculated distance from height: {distance_from_height}")
                    print(f"Manually calculated distance from width: {distance_from_width}")
                    
                    # Use the average of height and width distances
                    distance = (distance_from_height + distance_from_width) * 0.5
                    
                    # Additional sanity check
                    if not (0 < distance < 10):  # Reasonable distance range
                        rospy.logwarn(f"Calculated distance {distance} seems unrealistic")
                        # Optional: use a more conservative estimate
                        distance = min(max(distance, 0.1), 10)
                    
                    # Publish distance
                    distance_msg = Float32()
                    distance_msg.data = distance
                    self.distance_pub.publish(distance_msg)
                    
                    rospy.loginfo(f"Torpedo Map Distance: {distance} meters")
                
                except Exception as dist_error:
                    print(f"Unexpected distance estimation error: {dist_error}")
                    print(traceback.format_exc())
        
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")
            print(f"Callback error: {e}")
            print(traceback.format_exc())


def main():
    try:
        print("Starting main function...")
        node = TorpedoMapDistanceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROS Interrupt Exception")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())


if __name__ == '__main__':
    main()