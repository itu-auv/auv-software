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
        self.pc_sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        self.marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
        self.plane_marker_pub = rospy.Publisher('/plane_markers', MarkerArray, queue_size=10)
        
        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster()
        
        # Filtering parameters
        self.max_distance = 2.0  # Maximum distance in meters to consider points
        
        # DBSCAN clustering parameters
        self.eps = 0.1  # Increased from 0.05 to 0.1
        self.min_samples = 50  # Increased from 30 to 50
        self.downsample_factor = 4  # Only process every Nth point
        
        rospy.loginfo("PointCloud processor initialized")

    def filter_points(self, points):
        """Filter points based on distance from camera"""
        # Calculate distances from origin (camera)
        distances = np.linalg.norm(points, axis=1)
        # Create mask for points within max_distance
        mask = distances < self.max_distance
        return points[mask]

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
        
        rospy.loginfo(f"Frame calculated. Normal: {normal}, X: {x_axis}, Y: {y_axis}")
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
            
            rospy.loginfo(f"Plane fitted successfully. Center: {center}, Normal: {normal}")
            return corners, normal, center
            
        except Exception as e:
            rospy.logerr(f"Error in plane fitting: {str(e)}")
            return None, None, None

    def pointcloud_callback(self, msg):
        rospy.loginfo("Received pointcloud message")
        # Convert PointCloud2 to numpy array
        points = []
        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            # Downsample by only taking every Nth point
            if i % self.downsample_factor == 0:
                points.append([p[0], p[1], p[2]])
        
        if not points:
            rospy.logwarn("No points received in message")
            return
            
        points = np.array(points)
        rospy.loginfo(f"Processing {len(points)} points")
        
        # Filter distant points
        points = self.filter_points(points)
        rospy.loginfo(f"After filtering: {len(points)} points remaining")
        
        if len(points) < self.min_samples:
            rospy.logwarn(f"Not enough points after filtering: {len(points)} < {self.min_samples}")
            return
        
        # Apply DBSCAN clustering
        rospy.loginfo("Starting DBSCAN clustering...")
        # eps: İki noktanın aynı kümede olması için aralarındaki max mesafe
        # min_samples: Bir küme oluşturmak için gereken min nokta sayısı
        clustering = DBSCAN(
            eps=self.eps,  
            min_samples=self.min_samples,
            algorithm='ball_tree'  # Daha hızlı clustering için ball_tree algoritması
        ).fit(points)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        rospy.loginfo(f"Found {n_clusters} clusters")
        
        # Create marker array message
        marker_array = MarkerArray()
        plane_markers = MarkerArray()
        
        # Process each cluster
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get points in current cluster
            cluster_points = points[labels == label]
            
            # Calculate cluster centroid
            centroid = np.mean(cluster_points, axis=0)
            
            # Calculate cluster size
            cluster_min = np.min(cluster_points, axis=0)
            cluster_max = np.max(cluster_points, axis=0)
            cluster_size = cluster_max - cluster_min
            
            # Skip very small clusters (likely noise)
            if np.all(cluster_size < 0.02):  # 2cm minimum size
                continue

            # Calculate surface frame
            surface_result = self.calculate_surface_frame(cluster_points)
            if surface_result[0] is not None:
                center, rotation_matrix = surface_result
                
                # Calculate Euler angles from rotation matrix
                roll, pitch, yaw = self.calculate_euler_angles(rotation_matrix)
                quaternion = self.euler_to_quaternion(roll, pitch, yaw)

                # Calculate plane corners and normal
                plane_result = self.calculate_plane_corners(cluster_points)
                if plane_result[0] is not None:
                    corners, plane_normal, center = plane_result
                    
                    # Create plane marker
                    plane_marker = Marker()
                    plane_marker.header.frame_id = msg.header.frame_id
                    plane_marker.header.stamp = rospy.Time.now()
                    plane_marker.ns = "planes"
                    plane_marker.id = label
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
                    plane_marker.scale.x = 1.0
                    plane_marker.scale.y = 1.0
                    plane_marker.scale.z = 1.0
                    
                    # Add plane marker to array
                    plane_markers.markers.append(plane_marker)
                    rospy.loginfo(f"Added plane marker for cluster {label} with {len(plane_marker.points)} points")

                # Create transform message
                transform = TransformStamped()
                transform.header.stamp = rospy.Time.now()
                transform.header.frame_id = msg.header.frame_id
                transform.child_frame_id = f"object_{label}"
                
                # Set translation (centroid position)
                transform.transform.translation.x = centroid[0]
                transform.transform.translation.y = centroid[1]
                transform.transform.translation.z = centroid[2]
                
                # Set rotation (quaternion)
                transform.transform.rotation.x = quaternion[0]
                transform.transform.rotation.y = quaternion[1]
                transform.transform.rotation.z = quaternion[2]
                transform.transform.rotation.w = quaternion[3]
                
                # Broadcast transform
                self.tf_broadcaster.sendTransform(transform)

                # Create marker for visualization
                marker = Marker()
                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "objects"
                marker.id = label
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # Set marker position (centroid)
                marker.pose.position.x = centroid[0]
                marker.pose.position.y = centroid[1]
                marker.pose.position.z = centroid[2]
                
                # Set marker orientation (quaternion)
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                
                # Set marker scale (cluster size)
                marker.scale.x = max(cluster_size[0], 0.05)  # Minimum size of 5cm
                marker.scale.y = max(cluster_size[1], 0.05)
                marker.scale.z = max(cluster_size[2], 0.05)
                
                # Set marker color
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.5
                
                marker_array.markers.append(marker)
                
        # Publish marker array
        self.marker_pub.publish(marker_array)
        self.plane_marker_pub.publish(plane_markers)

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass