#!/usr/bin/env python3

import rospy
import numpy as np
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
        self.eps = 0.05  # 5cm - Maximum distance between points in a cluster
        self.min_samples = 30  # Minimum points to form a cluster
        
        rospy.loginfo("PointCloud processor initialized")

    def filter_points(self, points):
        """Filter points based on distance from camera"""
        # Calculate distances from origin (camera)
        distances = np.linalg.norm(points, axis=1)
        # Create mask for points within max_distance
        mask = distances < self.max_distance
        return points[mask]

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        if not points:
            return
            
        points = np.array(points)
        
        # Filter distant points
        points = self.filter_points(points)
        
        if len(points) < self.min_samples:
            rospy.logdebug("Not enough points after filtering")
            return
        
        # Apply DBSCAN clustering
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
            
            # Create and publish transform
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = msg.header.frame_id
            transform.child_frame_id = f"object_{label}"
            
            transform.transform.translation.x = centroid[0]
            transform.transform.translation.y = centroid[1]
            transform.transform.translation.z = centroid[2]
            #TODO MAKE SURFACE ESTIMATION AND USE THAT FOR ORİENATION.
            # Use identity rotation
            q = tf_trans.quaternion_from_euler(0, 0, 0)
            transform.transform.rotation.x = q[0]
            transform.transform.rotation.y = q[1]
            transform.transform.rotation.z = q[2]
            transform.transform.rotation.w = q[3]
            
            self.tf_broadcaster.sendTransform(transform)
            
            # Create visualization marker
            marker = Marker()
            marker.header = transform.header
            marker.ns = "objects"
            marker.id = label
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = centroid[0]
            marker.pose.position.y = centroid[1]
            marker.pose.position.z = centroid[2]
            marker.pose.orientation = transform.transform.rotation
            
            # Set marker size to cluster bounds
            marker.scale.x = max(cluster_size[0], 0.02)  # Minimum 2cm size
            marker.scale.y = max(cluster_size[1], 0.02)
            marker.scale.z = max(cluster_size[2], 0.02)
            
            # Color based on distance from camera
            distance = np.linalg.norm(centroid)
            marker.color.r = min(1.0, distance / self.max_distance)
            marker.color.g = max(0.0, 1.0 - distance / self.max_distance)
            marker.color.b = 0.2
            marker.color.a = 0.6
            
            marker_array.markers.append(marker)
        
        # Publish markers
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass