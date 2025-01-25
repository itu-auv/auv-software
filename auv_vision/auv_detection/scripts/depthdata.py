#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics_ros.msg import YoloResult

class DepthYOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        # Subscribe to depth image
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        # Subscribe to YOLO detections
        self.yolo_sub = rospy.Subscriber('/yolo_result', YoloResult, self.yolo_callback)
        
        self.latest_depth_image = None
        self.latest_bbox = None
        
    def depth_callback(self, msg):
        try:
            # Convert depth image to OpenCV format
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # If we have both depth image and bbox, process them
            if self.latest_bbox is not None:
                self.process_depth_in_bbox()
                
        except Exception as e:
            rospy.logerr(f"Error in depth callback: {e}")
    
    def yolo_callback(self, msg):
        try:
            if msg.detections.detections:
                # Get the first detection's bbox
                detection = msg.detections.detections[0]
                bbox = detection.bbox
                self.latest_bbox = [
                    int(bbox.center.x - bbox.size_x/2),  # x1
                    int(bbox.center.y - bbox.size_y/2),  # y1
                    int(bbox.center.x + bbox.size_x/2),  # x2
                    int(bbox.center.y + bbox.size_y/2)   # y2
                ]
                
                # If we have both depth image and bbox, process them
                if self.latest_depth_image is not None:
                    self.process_depth_in_bbox()
            else:
                self.latest_bbox = None
                
        except Exception as e:
            rospy.logerr(f"Error in YOLO callback: {e}")
    
    def process_depth_in_bbox(self):
        if self.latest_depth_image is None or self.latest_bbox is None:
            return
            
        try:
            # Extract the region of interest from depth image
            x1, y1, x2, y2 = self.latest_bbox
            depth_roi = self.latest_depth_image[y1:y2, x1:x2]
            
            # Calculate average depth in the ROI (excluding 0 values which might be invalid)
            valid_depths = depth_roi[depth_roi > 0]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                rospy.loginfo(f"Average depth in bbox: {avg_depth:.2f} mm")
                
                # Visualize the ROI
                viz_image = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
                
        except Exception as e:
            rospy.logerr(f"Error processing depth in bbox: {e}")

if __name__ == '__main__':
    rospy.init_node('depth_yolo_processor')
    processor = DepthYOLOProcessor()
    rospy.spin()