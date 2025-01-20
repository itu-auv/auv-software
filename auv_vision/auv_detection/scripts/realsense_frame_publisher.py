#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf_trans

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('pointcloud_processor')
        
        # Initialize subscribers and publishers
        self.pc_sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        self.marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
        
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

    def calculate_cluster_normals(self, cluster_points):
        """Calculate normals for cluster points using Open3D"""
        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_points)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Get the average normal vector
        normals = np.asarray(pcd.normals)
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        
        # Ensure normal points "up" (positive z)
        if mean_normal[2] < 0:
            mean_normal = -mean_normal
            
        return mean_normal

    def calculate_euler_angles(self, normal):
        """
        Calculate Euler angles from surface normal relative to z-axis basis vector
        Returns: roll, pitch, yaw in degrees
        """
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate angles
        # Pitch (rotation around y-axis)
        pitch = np.arctan2(normal[0], normal[2])
        
        # Roll (rotation around x-axis)
        roll = -np.arctan2(normal[1], normal[2])
        
        # Yaw (rotation around z-axis)
        yaw = np.arctan2(normal[1], normal[0])
        
        # Convert to degrees
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        return roll_deg, pitch_deg, yaw_deg

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # Convert degrees to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        return tf_trans.quaternion_from_euler(roll, pitch, yaw)

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

            # Calculate cluster normal and orientation
            normal = self.calculate_cluster_normals(cluster_points)
            roll, pitch, yaw = self.calculate_euler_angles(normal)
            quaternion = self.euler_to_quaternion(roll, pitch, yaw)

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

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass