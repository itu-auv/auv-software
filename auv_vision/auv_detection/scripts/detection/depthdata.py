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
        
        # Parameters for depth filtering
        self.kernel_size = 5  # Size of the neighborhood to consider
        self.depth_threshold = 100  # Maximum allowed difference in mm
        self.min_valid_neighbors = 4  # Minimum number of valid neighbors required
        
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
    
    def filter_outliers(self, depth_roi):
        """
        Filter depth values that are significantly different from their surroundings
        using a sliding window approach.
        """
        height, width = depth_roi.shape
        filtered_roi = np.copy(depth_roi)
        pad = self.kernel_size // 2
        
        # Create padded array to handle boundaries
        padded_roi = np.pad(depth_roi, pad, mode='edge')
        
        for i in range(height):
            for j in range(width):
                # Get the neighborhood window
                window = padded_roi[i:i+self.kernel_size, j:j+self.kernel_size]
                center_value = depth_roi[i, j]
                
                # Skip if center value is 0 (invalid depth)
                if center_value == 0:
                    continue
                
                # Get valid depth values in the window (non-zero values)
                valid_neighbors = window[window > 0]
                valid_neighbors = valid_neighbors[valid_neighbors != center_value]  # Exclude center value
                
                # Skip if not enough valid neighbors
                if len(valid_neighbors) < self.min_valid_neighbors:
                    continue
                
                # Calculate median of valid neighbors
                median_depth = np.median(valid_neighbors)
                
                # If the center value is too different from the median, mark it as invalid
                if abs(center_value - median_depth) > self.depth_threshold:
                    filtered_roi[i, j] = 0
        
        return filtered_roi

    def process_depth_in_bbox(self):
        if self.latest_depth_image is None or self.latest_bbox is None:
            return
            
        try:
            # Extract the region of interest from depth image
            x1, y1, x2, y2 = self.latest_bbox
            
            # Ensure bbox coordinates are within image bounds
            height, width = self.latest_depth_image.shape
            x1 = max(0, min(x1, width-1))
            x2 = max(0, min(x2, width-1))
            y1 = max(0, min(y1, height-1))
            y2 = max(0, min(y2, height-1))
            
            depth_roi = self.latest_depth_image[y1:y2, x1:x2]
            
            # Apply outlier filtering
            filtered_roi = self.filter_outliers(depth_roi)
            
            # Calculate average depth in the ROI (excluding 0 values which might be invalid)
            valid_depths = filtered_roi[filtered_roi > 0]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                median_depth = np.median(valid_depths)
                rospy.loginfo(f"Average depth in bbox: {avg_depth:.2f} mm, Median depth: {median_depth:.2f} mm")
                
                # Visualize the ROI
                viz_image = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
                
                # Draw bbox and depth info
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz_image, f"Depth: {median_depth:.0f}mm", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display original and filtered depth ROIs side by side
                roi_viz = np.hstack([
                    cv2.normalize(depth_roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                    cv2.normalize(filtered_roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ])
                cv2.imshow("Depth ROI (Original | Filtered)", roi_viz)
                cv2.imshow("Depth Visualization", viz_image)
                cv2.waitKey(1)
                
        except Exception as e:
            rospy.logerr(f"Error processing depth in bbox: {e}")

if __name__ == '__main__':
    rospy.init_node('depth_yolo_processor')
    processor = DepthYOLOProcessor()
    rospy.spin()