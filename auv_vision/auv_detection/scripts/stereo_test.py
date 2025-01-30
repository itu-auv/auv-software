#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from stereo_msgs.msg import DisparityImage
import message_filters
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class StereoProcessor:
    def __init__(self):
        rospy.init_node('stereo_processor', anonymous=True)
        self.bridge = CvBridge()
        
        # Camera calibration parameters for both cameras (they are the same)
        K = np.array([
            [602.0906606373956, 0.0, 300.33385025264073],
            [0.0, 594.2506199363708, 222.5904158005273],
            [0.0, 0.0, 1.0]
        ])
        D = np.array([0.010762757721444076, -0.08670132807475911, -0.0005974624862354021, -0.0028558476549465682, 0.0])
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P = np.array([
            [598.1726684570312, 0.0, 297.917814312219, 0.0],
            [0.0, 592.7899169921875, 221.8392037641206, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        # Set camera parameters for both left and right cameras
        self.camera_matrix_left = K
        self.camera_matrix_right = K
        self.dist_left = D
        self.dist_right = D
        self.R = R
        
        # Set a nominal baseline (distance between cameras)
        # This should be adjusted based on your actual camera setup
        self.T = np.array([3.0, 0.0, 0.0])  # 3m baseline along X-axis

        # Initialize image size from the calibration data
        self.image_size = (640, 480)  # width, height
        
        # Initialize stereo rectification maps
        self.init_stereo_maps()
        
        # Subscribe to the stereo camera topics
        self.left_sub = message_filters.Subscriber('/taluy/cameras/cam_front2/image_raw', Image)
        self.right_sub = message_filters.Subscriber('/taluy/cameras/cam_front/image_raw', Image)
        
        # Time synchronizer for the camera topics
        ts = message_filters.ApproximateTimeSynchronizer([self.left_sub, self.right_sub], 10, 0.1)
        ts.registerCallback(self.stereo_callback)
        
        # Publisher for disparity image
        self.disparity_pub = rospy.Publisher('/stereo/disparity', DisparityImage, queue_size=1)
        self.pcd_pub = rospy.Publisher('/stereo/pcd', PointCloud2, queue_size=1)

    def init_stereo_maps(self):
        if self.image_size is not None:
            # stereoRectify returns R1, R2, P1, P2, Q, roi1, roi2
            (self.R1, self.R2, 
             self.P1, self.P2, 
             self.Q, self.roi1, 
             self.roi2) = cv.stereoRectify(
                self.camera_matrix_left, self.dist_left,
                self.camera_matrix_right, self.dist_right,
                self.image_size, self.R, self.T,
                flags=cv.CALIB_ZERO_DISPARITY, alpha=0
            )
            
            # Generate rectification maps
            self.map1x, self.map1y = cv.initUndistortRectifyMap(
                self.camera_matrix_left, self.dist_left, self.R1, self.P1,
                self.image_size, cv.CV_32F
            )
            self.map2x, self.map2y = cv.initUndistortRectifyMap(
                self.camera_matrix_right, self.dist_right, self.R2, self.P2,
                self.image_size, cv.CV_32F
            )

    def compute_disparity(self, img_left, img_right):
        # Increase numDisparities and adjust other parameters
        window_size = 5
        stereo = cv.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*6,  # 96 disparities (must be multiple of 16)
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,  # Adjust based on blockSize
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
        
        # Filter out noise and invalid disparities
        disparity = np.abs(disparity)
        disparity[disparity == 0] = 0.1  # Avoid division by zero in reprojection
        
        return disparity

    def reproject_to_3D(self, disparity):
        # Convert disparity map to 3D points
        points_3D = cv.reprojectImageTo3D(disparity, self.Q)
        return points_3D

    def create_depth_visualization(self, points_3D):
        # Extract Z-coordinates (depth)
        depth = points_3D[:, :, 2]
        
        # Filter out invalid points (too close or too far)
        min_depth = 0.1  # 10cm
        max_depth = 5.0  # 5m
        mask = (depth > min_depth) & (depth < max_depth)
        valid_depth = np.where(mask, depth, min_depth)
        
        # Normalize depth for visualization (0-255)
        depth_normalized = np.clip(((valid_depth - min_depth) / (max_depth - min_depth) * 255), 0, 255)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colormap = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
        
        # Add legend
        height, width = depth_colormap.shape[:2]
        legend_width = 30
        legend = np.zeros((height, legend_width, 3), dtype=np.uint8)
        for i in range(height):
            value = np.array([[int(255 * (1 - i/height))]], dtype=np.uint8)
            color = cv.applyColorMap(value, cv.COLORMAP_JET)[0, 0]
            legend[i, :] = color
            
        # Add text for depth range
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv.putText(legend, f'{max_depth}m', (0, 20), font, font_scale, (255, 255, 255), 1)
        cv.putText(legend, f'{min_depth}m', (0, height-10), font, font_scale, (255, 255, 255), 1)
        
        # Combine depth map with legend
        visualization = np.hstack((depth_colormap, legend))
        
        return visualization

    def stereo_callback(self, left_msg, right_msg):
        try:
            # Convert ROS images to OpenCV format
            left_image = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            right_image = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
            
            # Rectify images
            left_rect = cv.remap(left_image, self.map1x, self.map1y, cv.INTER_LINEAR)
            right_rect = cv.remap(right_image, self.map2x, self.map2y, cv.INTER_LINEAR)
            
            # Compute disparity
            disparity = self.compute_disparity(left_rect, right_rect)
            
            # Create and publish disparity message
            disp_msg = DisparityImage()
            disp_msg.header = left_msg.header
            disp_msg.image = self.bridge.cv2_to_imgmsg(disparity, "32FC1")
            disp_msg.f = self.P1[0, 0]  # Focal length
            disp_msg.T = float(abs(self.T[0]))  # Baseline
            self.disparity_pub.publish(disp_msg)
            
            # Compute 3D points
            points_3D = self.reproject_to_3D(disparity)
            
            # Flatten the 3D points and remove invalid points
            valid_points = []
            for v in range(points_3D.shape[0]):
                for u in range(points_3D.shape[1]):
                    x, y, z = points_3D[v, u]
                    if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                        valid_points.append((x, y, z))
            
            # Create a PointCloud2 message without is_dense
            point_cloud = pc2.create_cloud_xyz32(left_msg.header, valid_points)
            # Ensure is_dense is not set
            point_cloud.is_dense = False
            
            # Publish the PointCloud2 message
            self.pcd_pub.publish(point_cloud)
            
            # Create depth visualization
            depth_vis = self.create_depth_visualization(points_3D)
            cv.imshow('Depth Map', depth_vis)
            cv.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing stereo images: {str(e)}")


def main():
    try:
        processor = StereoProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()