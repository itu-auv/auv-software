#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DepthTo3D:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.image_callback)
        self.sample_points = []  # Store grid points
    
    def visualize_depth(self, depth_image):
        # Normalize the depth image for visualization
        depth_colormap = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply a color map
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
        
        # Add text showing depth values at grid points
        height, width = depth_image.shape
        
        # Create a grid of points (5x5 grid) only if not already created
        if not self.sample_points:
            rows, cols = 5, 5
            row_step = height // (rows + 1)
            col_step = width // (cols + 1)
            
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    self.sample_points.append((j * col_step, i * row_step))
        
        # Create a copy for drawing
        vis_image = depth_colormap.copy()
        
        # Add depth values at sample points
        for pt in self.sample_points:
            depth_val = depth_image[pt[1], pt[0]]
            # Always show the point
            cv2.circle(vis_image, pt, 2, (255, 255, 255), -1)
            
            # Show depth value
            if depth_val > 0:
                text = f"{depth_val:.0f}"
            else:
                text = "N/A"  # Show N/A for invalid depth values
            
            # Always show the text
            cv2.putText(vis_image, text, (pt[0]-20, pt[1]-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add colorbar
        colorbar_width = 30
        colorbar = np.zeros((height, colorbar_width, 3), dtype=np.uint8)
        for i in range(height):
            color = cv2.applyColorMap(np.array([[int(255 * (1 - i/height))]], dtype=np.uint8), cv2.COLORMAP_JET)
            colorbar[i, :] = color[0, 0]
        
        # Add depth scale values
        depth_min = np.min(depth_image[depth_image > 0])
        depth_max = np.max(depth_image)
        cv2.putText(colorbar, f"{depth_max:.0f}mm", (2, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(colorbar, f"{depth_min:.0f}mm", (2, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine depth map and colorbar
        combined_image = np.hstack((vis_image, colorbar))
        
        # Show the visualization
        cv2.imshow('Depth Map', combined_image)
        cv2.waitKey(1)
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Visualize the depth map
        self.visualize_depth(cv_image)
        
        # Görüntü boyutları
        height, width = cv_image.shape
        
        # Derinlik bilgilerini kullanarak 3D koordinatları hesaplayabiliriz
        for y in range(height):
            for x in range(width):
                depth = cv_image[y, x]
                if depth > 0:  # Geçerli bir derinlik varsa
                    # Kamera iç parametrelerini kullanarak (fx, fy, cx, cy) bu derinliği 3D koordinatlara dönüştürebilirsiniz
                    x3d = (x - 306) * depth / 432 #fx
                    y3d = (y - 221) * depth / 567 #fy
                    z3d = depth

if __name__ == '__main__':
    rospy.init_node('depth_to_3d')
    depth_processor = DepthTo3D()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()