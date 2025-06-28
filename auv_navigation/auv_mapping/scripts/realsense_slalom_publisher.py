#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class PointCloudClusterer:
    def __init__(self):
        rospy.init_node('pointcloud_clusterer')

        self.pc_sub = rospy.Subscriber('taluy/camera/depth/points', PointCloud2, self.pointcloud_callback)

        self.tf_broadcaster = TransformBroadcaster()

        self.max_distance = 30.0
        self.eps = 0.1
        self.min_samples = 30
        self.downsample_factor = 2

        rospy.loginfo("PointCloud clusterer with TF broadcasting initialized")

    def filter_points(self, points):
        distances = np.linalg.norm(points, axis=1)
        mask = distances < self.max_distance
        return points[mask]

    def pointcloud_callback(self, msg):
        rospy.loginfo("\n" + "=" * 50 + "\nProcessing new frame")

        points = []
        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if i % self.downsample_factor == 0:
                points.append([p[0], p[1], p[2]])

        if not points:
            rospy.logwarn("No points in frame")
            return

        points = np.array(points)
        points = self.filter_points(points)

        if len(points) < self.min_samples:
            rospy.logwarn(f"Too few points: {len(points)} < {self.min_samples}")
            return

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm='ball_tree'
        ).fit(points)

        labels = clustering.labels_
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        rospy.loginfo(f"Found {n_clusters} clusters")

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(centroid)

            rospy.loginfo(
                f"Cluster {label}: {len(cluster_points)} points, "
                f"Centroid = ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                f"Distance = {distance:.2f}m"
            )

            # Broadcast TF frame
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = msg.header.frame_id
            transform.child_frame_id = f"cluster_{label}"
            transform.transform.translation.x = centroid[0]
            transform.transform.translation.y = centroid[1]
            transform.transform.translation.z = centroid[2]

            # Identity rotation (no orientation yet)
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(transform)

if __name__ == '__main__':
    try:
        clusterer = PointCloudClusterer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass