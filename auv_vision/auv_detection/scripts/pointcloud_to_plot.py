#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threading import Lock
import numpy as np

class PointCloudVisualizer:
    def __init__(self):
        rospy.init_node('pointcloud_visualizer')
        print("Initializing pointcloud visualizer")
        # Initialize subscriber with queue_size=1 to drop old messages
        self.pc_sub = rospy.Subscriber('/stereo/pcd', PointCloud2, 
                                     self.pointcloud_callback, queue_size=1)
        print("Subscribed to /stereo/pcd")
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
        
        # Create figure for 3D plotting
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set initial view angle for better visualization
        self.ax.view_init(elev=45, azim=45)
        
        # Thread safety
        self.lock = Lock()
        self.new_data = False
        self.points = None
        self.labels = None
        
        rospy.loginfo("PointCloud visualizer initialized")
        
        # Setup timer for plot updates (10 seconds)
        self.timer = rospy.Timer(rospy.Duration(10.0), self.update_plot)
        
        # Counter for processing every Nth message
        self.msg_counter = 0
        self.process_nth_msg = 5  # Process every 5th message

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
        # Only process every Nth message
        self.msg_counter += 1
        if self.msg_counter % self.process_nth_msg != 0:
            return
            
        try:
            # Convert PointCloud2 message to a list of points
            points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
            if not points:
                rospy.logwarn("Received empty point cloud or all points are invalid.")
                return

            # Further processing of valid points
            points = np.array(points)
            
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
            
            # Store the new data
            with self.lock:
                # Swap axes for better visualization (Y up, Z forward)
                self.points = downsampled_points[:, [0, 2, 1]]  # X, Z, Y
                self.labels = db.labels_
                self.new_data = True
            
        except Exception as e:
            rospy.logerr(f"Error processing pointcloud: {str(e)}")

    def update_plot(self, event):
        if not self.new_data:
            return
            
        with self.lock:
            points = self.points
            labels = self.labels
            self.new_data = False
        
        try:
            # Clear previous plot
            self.ax.cla()
            
            # Set plot limits and labels
            self.ax.set_xlabel('X (Left/Right)')
            self.ax.set_ylabel('Z (Forward)')
            self.ax.set_zlabel('Y (Up/Down)')
            self.ax.set_title('Point Cloud Clusters (Updated every 10s)')
            
            # Plot clusters with different colors (plot noise last)
            unique_labels = np.unique(labels)
            non_noise_labels = unique_labels[unique_labels != -1]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(non_noise_labels)))
            
            # Plot non-noise clusters
            for label, color in zip(non_noise_labels, colors):
                cluster_mask = labels == label
                cluster_points = points[cluster_mask]
                
                self.ax.scatter(cluster_points[:, 0],  # X
                              cluster_points[:, 1],    # Z (forward)
                              cluster_points[:, 2],    # Y (up)
                              c=[color], marker='.', s=2, 
                              label=f'Cluster {label}')
            
            # Plot noise points in black (if any)
            noise_mask = labels == -1
            if np.any(noise_mask):
                noise_points = points[noise_mask]
                self.ax.scatter(noise_points[:, 0],    # X
                              noise_points[:, 1],      # Z (forward)
                              noise_points[:, 2],      # Y (up)
                              c='black', marker='.', s=1, 
                              label='Noise', alpha=0.5)
            
            # Add legend and grid
            self.ax.legend(loc='upper right')
            self.ax.grid(True)
            
            # Set equal aspect ratio and update plot
            self.ax.set_box_aspect([1,1,1])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            rospy.logerr(f"Error updating plot: {str(e)}")

if __name__ == '__main__':
    try:
        visualizer = PointCloudVisualizer()
        plt.show()  # This blocks and runs the GUI event loop
        rospy.spin()
    except rospy.ROSInterruptException:
        plt.close('all')
        pass