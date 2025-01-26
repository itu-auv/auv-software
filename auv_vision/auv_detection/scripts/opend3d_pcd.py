#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import open3d as o3d
from threading import Lock
import os

class PointCloudVisualizer:
    def __init__(self):
        rospy.init_node('pointcloud_visualizer')
        
        # Initialize subscriber with queue_size=1 to drop old messages
        self.pc_sub = rospy.Subscriber('/taluy/camera/depth/points', PointCloud2, 
                                     self.pointcloud_callback, queue_size=1)
        
        # Filtering parameters
        self.max_distance = 15.0
        self.min_distance = 0.2
        self.height_threshold = -0.3
        self.forward_threshold = 0.2
        
        # DBSCAN parameters - increased eps for faster clustering
        self.eps = 0.10
        self.min_samples = 10
        self.max_samples = 250000
        self.downsample_factor = 4  # Increased downsampling for better performance
        
        # Initialize Open3D visualization
        self.vis = o3d.visualization.Visualizer()
        try:
            # Try headless rendering first
            self.vis.create_window(window_name="Point Cloud Viewer", 
                                 width=1280, 
                                 height=720,
                                 visible=False)  # Set to headless mode
        except Exception as e:
            rospy.logwarn(f"Failed to create window in headless mode: {e}")
            try:
                # Try software rendering as fallback
                os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
                self.vis.create_window(window_name="Point Cloud Viewer", 
                                     width=1280, 
                                     height=720)
            except Exception as e:
                rospy.logerr(f"Failed to create visualization window: {e}")
                raise
        
        # Add coordinate frame
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.world_frame)
        
        # Initialize empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        self.lock = Lock()
        
        # Set default view
        self.reset_view()
        
    def reset_view(self):
        try:
            ctr = self.vis.get_view_control()
            if ctr is not None:
                ctr.set_zoom(0.5)
                ctr.set_front([0.4257, -0.2125, -0.8795])
                ctr.set_lookat([0.0, 0.0, 0.0])
                ctr.set_up([-0.0694, -0.9768, 0.2024])
        except Exception as e:
            rospy.logwarn(f"Failed to set view: {e}")
            
    def filter_points(self, points):
        """Filter points based on distance and position from camera"""
        # Calculate distances from origin (camera)
        distances = np.linalg.norm(points, axis=1)
        
        # Combine all filters in one operation for efficiency
        mask = ((distances > self.min_distance) & 
                (distances < self.max_distance) & 
                (points[:, 1] > self.height_threshold) &  # Y axis (up/down)
                (points[:, 2] > self.forward_threshold))  # Z axis (forward)
        
        return points[mask]

    def pointcloud_callback(self, msg):
        with self.lock:
            # Convert ROS PointCloud2 to numpy array
            pc_data = []
            for point in pc2.read_points(msg, skip_nans=True):
                pc_data.append([point[0], point[1], point[2]])
            
            if not pc_data:
                return
                
            points = np.array(pc_data)
            
            # Filter points
            filtered_points = self.filter_points(points)
            
            if len(filtered_points) < self.min_samples:
                return
            
            # Downsample points
            downsampled_points = filtered_points[::self.downsample_factor]
            
            # Perform DBSCAN clustering with reduced max_samples
            if len(downsampled_points) > self.max_samples:
                indices = np.random.choice(len(downsampled_points), 
                                        self.max_samples, replace=False)
                downsampled_points = downsampled_points[indices]
            
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(downsampled_points)
            
            # Update Open3D point cloud
            self.pcd.points = o3d.utility.Vector3dVector(downsampled_points)
            
            # Color the clusters
            labels = db.labels_
            max_label = labels.max()
            colors = np.zeros((len(downsampled_points), 3))
            
            # Set colors for clusters
            for i in range(max_label + 1):
                cluster_mask = labels == i
                colors[cluster_mask] = np.random.uniform(0.3, 1, 3)
            
            # Set black color for noise points
            noise_mask = labels == -1
            colors[noise_mask] = [0.3, 0.3, 0.3]  # Dark gray for noise
            
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Update visualization
            self.vis.update_geometry(self.pcd)
            
    def run(self):
        while not rospy.is_shutdown():
            with self.lock:
                self.vis.poll_events()
                self.vis.update_renderer()
                
        # Clean up
        self.vis.destroy_window()

if __name__ == '__main__':
    try:
        visualizer = PointCloudVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass