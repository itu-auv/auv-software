#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf_trans

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('pointcloud_processor')
        
        # Initialize subscribers and publishers
        self.pc_sub = rospy.Subscriber('/taluy/camera/depth/points', PointCloud2, self.pointcloud_callback)
        self.marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
        self.plane_marker_pub = rospy.Publisher('/plane_markers', MarkerArray, queue_size=10)
        
        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster()
        
        # Filtering parameters
        self.max_distance = 15.0  # Maximum distance in meters to consider points
        self.min_distance = 0.5  # Minimum distance to filter out very close points
        self.height_threshold = 0.0  # Points below camera height
        self.forward_threshold = 0.3  # Minimum forward distance (z-axis)
        
        # DBSCAN clustering parameters
        self.eps = 0.1  # Reduced for more precise clustering
        self.min_samples = 10  # Reduced to detect smaller objects
        self.max_samples = 2500  # Maximum points per cluster
        self.downsample_factor = 4  # Keep more points for better detection
        
        rospy.loginfo("PointCloud processor initialized")

    def filter_points(self, points):
        """Filter points based on distance and position from camera"""
        # Calculate distances from origin (camera)
        distances = np.linalg.norm(points, axis=1)
        
        # Distance filter (not too close, not too far)
        distance_mask = (distances > self.min_distance) & (distances < self.max_distance)
        
        # Height filter (ignore points too low - pool floor)
        height_mask = points[:, 1] > self.height_threshold
        
        # Forward filter (only points in front of camera)
        forward_mask = points[:, 2] > self.forward_threshold
        
        # Combine all filters
        combined_mask = distance_mask & height_mask & forward_mask
        
        return points[combined_mask]

    def calculate_surface_frame(self, cluster_points):
        """Calculate surface frame aligned with the fitted rectangle's edges"""
        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_points)
        
        # Use RANSAC to fit a plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                               ransac_n=3,
                                               num_iterations=1000)
        
        if len(inliers) < 3:
            rospy.logwarn("Not enough inliers for plane fitting")
            return None, None
            
        # Get points that belong to the plane
        inlier_cloud = pcd.select_by_index(inliers)
        points_array = np.asarray(inlier_cloud.points)
        
        # Calculate plane center
        center = np.mean(points_array, axis=0)
        
        # Get normal from plane model (this will be our Z axis)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Ensure normal points "up"
        if normal[2] < 0:
            normal = -normal
            
        # Find X axis: Use a vector that's not parallel to normal
        # If normal is more vertical, use [1,0,0], otherwise use [0,1,0]
        if abs(normal[2]) > 0.707:  # cos(45°) ≈ 0.707
            temp_vector = np.array([1.0, 0.0, 0.0])
        else:
            temp_vector = np.array([0.0, 1.0, 0.0])
            
        # X axis is perpendicular to both normal and temp_vector
        x_axis = np.cross(normal, temp_vector)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y axis is cross product of Z (normal) and X
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Create rotation matrix from these axes
        rotation_matrix = np.column_stack((x_axis, y_axis, normal))
        
        #rospy.loginfo(f"Frame calculated. Normal: {normal}, X: {x_axis}, Y: {y_axis}")
        return center, rotation_matrix

    def calculate_euler_angles(self, rotation_matrix):
        """
        Calculate Euler angles from rotation matrix
        Returns: roll, pitch, yaw in degrees
        """
        # Convert rotation matrix to euler angles
        euler_angles = tf_trans.euler_from_matrix(rotation_matrix)
        
        # Convert to degrees
        roll_deg = np.degrees(euler_angles[0])
        pitch_deg = np.degrees(euler_angles[1])
        yaw_deg = np.degrees(euler_angles[2])
        
        return roll_deg, pitch_deg, yaw_deg

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # Convert degrees to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        return tf_trans.quaternion_from_euler(roll, pitch, yaw)

    def calculate_plane_corners(self, cluster_points):
        """Calculate plane corners and normal using RANSAC"""
        try:
            # Convert numpy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Use RANSAC to fit a plane
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                   ransac_n=3,
                                                   num_iterations=1000)
            
            if len(inliers) < 3:
                rospy.logwarn("Not enough inliers for plane fitting")
                return None, None, None
                
            # Get points that belong to the plane
            inlier_cloud = pcd.select_by_index(inliers)
            points_array = np.asarray(inlier_cloud.points)
            
            # Calculate plane center and size
            center = np.mean(points_array, axis=0)
            min_bound = np.min(points_array, axis=0)
            max_bound = np.max(points_array, axis=0)
            size = max_bound - min_bound
            
            # Create corners (1 meter x 1 meter plane centered at center point)
            plane_size = 0.5  # half size in meters
            corners = []
            corners.append(center + np.array([-plane_size, -plane_size, 0]))
            corners.append(center + np.array([plane_size, -plane_size, 0]))
            corners.append(center + np.array([plane_size, plane_size, 0]))
            corners.append(center + np.array([-plane_size, plane_size, 0]))
            
            # Calculate normal from plane model
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            
            # Ensure normal points "up"
            if normal[2] < 0:
                normal = -normal
            
            #rospy.loginfo(f"Plane fitted successfully. Center: {center}, Normal: {normal}")
            return corners, normal, center
            
        except Exception as e:
            rospy.logerr(f"Error in plane fitting: {str(e)}")
            return None, None, None

    def pointcloud_callback(self, msg):
        """Process incoming pointcloud messages"""
        try:
            # Convert PointCloud2 to numpy array
            points = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
            
            # Log initial point count
            rospy.loginfo(f"Processing {len(points)} points")
            
            # Filter points
            filtered_points = self.filter_points(points)
            rospy.loginfo(f"After filtering: {len(filtered_points)} points remaining")
            
            if len(filtered_points) < self.min_samples:
                rospy.logwarn("Not enough points after filtering")
                return
                
            # Downsample points for faster processing
            downsampled_points = filtered_points[::self.downsample_factor]
            
            # Perform DBSCAN clustering
            rospy.loginfo("Starting DBSCAN clustering...")
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(downsampled_points)
            
            # Get cluster labels
            labels = db.labels_
            unique_labels = np.unique(labels[labels != -1])
            rospy.loginfo(f"Found {len(unique_labels)} clusters")
            
            # Log number of points in each cluster using downsampled points
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = downsampled_points[cluster_mask]
                rospy.loginfo(f"Cluster {label} has {len(cluster_points)} points")
            
            # Process each cluster
            marker_array = MarkerArray()
            plane_marker_array = MarkerArray()
            
            for cluster_id, label in enumerate(unique_labels):
                # Get points in this cluster
                cluster_mask = labels == label
                cluster_points = downsampled_points[cluster_mask]
                
                # Skip if cluster is too large
                if len(cluster_points) > self.max_samples:
                    rospy.loginfo(f"Skipping cluster {cluster_id} with {len(cluster_points)} points (exceeds max_samples)")
                    continue
                
                # Calculate surface frame
                center, rotation_matrix = self.calculate_surface_frame(cluster_points)
                
                if center is not None and rotation_matrix is not None:
                    # Calculate Euler angles
                    roll, pitch, yaw = self.calculate_euler_angles(rotation_matrix)
                    
                    # Convert to quaternion
                    q = self.euler_to_quaternion(roll, pitch, yaw)
                    
                    # Broadcast transform
                    transform = TransformStamped()
                    transform.header.stamp = rospy.Time.now()
                    transform.header.frame_id = msg.header.frame_id
                    transform.child_frame_id = f"surface_frame_{cluster_id}"
                    
                    transform.transform.translation.x = center[0]
                    transform.transform.translation.y = center[1]
                    transform.transform.translation.z = center[2]
                    
                    transform.transform.rotation.x = q[0]
                    transform.transform.rotation.y = q[1]
                    transform.transform.rotation.z = q[2]
                    transform.transform.rotation.w = q[3]
                    
                    self.tf_broadcaster.sendTransform(transform)
                    
                    # Calculate and publish plane corners
                    corners, normal, center = self.calculate_plane_corners(cluster_points)
                    if corners is not None:
                        plane_marker = self.create_plane_marker(corners, cluster_id, msg.header.frame_id)
                        plane_marker_array.markers.append(plane_marker)
            
            # Publish markers
            if len(plane_marker_array.markers) > 0:
                self.plane_marker_pub.publish(plane_marker_array)
            
        except Exception as e:
            rospy.logerr(f"Error processing pointcloud: {str(e)}")

    def create_plane_marker(self, corners, cluster_id, frame_id):
        plane_marker = Marker()
        plane_marker.header.frame_id = frame_id
        plane_marker.header.stamp = rospy.Time.now()
        plane_marker.ns = "planes"
        plane_marker.id = cluster_id
        plane_marker.type = Marker.TRIANGLE_LIST
        plane_marker.action = Marker.ADD
        
        # Create two triangles from the corners to form a rectangle
        triangle_indices = [0, 1, 2, 0, 2, 3]  # Triangle vertices order
        for i in triangle_indices:
            p = Point()
            p.x = float(corners[i][0])
            p.y = float(corners[i][1])
            p.z = float(corners[i][2])
            plane_marker.points.append(p)
        
        # Set marker color (semi-transparent blue)
        plane_marker.color.r = 0.0
        plane_marker.color.g = 0.0
        plane_marker.color.b = 1.0
        plane_marker.color.a = 0.5
        
        # Set marker scale (actual size)
        plane_marker.scale.x = 10.0
        plane_marker.scale.y = 10.0
        plane_marker.scale.z = 10.0
        
        return plane_marker

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass