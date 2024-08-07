#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from collections import deque
import numpy as np
import tf
from std_msgs.msg import Bool
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


class Line3DIntersector:
    def __init__(self):
        pass

    def line_to_points(self, a0, a1, num_points=100):
        """Convert a line segment into a list of points."""
        a0, a1 = np.array(a0), np.array(a1)
        return np.linspace(a0, a1, num_points)

    def generate_point_cloud(self, lines, num_points_per_line=100):
        """Generate a point cloud from the given lines."""
        if len(lines) == 0:
            return np.array([])
        point_cloud = []

        # Generate points along each line
        for a0, a1 in lines:
            line_points = self.line_to_points(a0, a1, num_points_per_line)
            point_cloud.append(line_points)

        # Flatten the list of points
        point_cloud = np.vstack(point_cloud)
        return point_cloud

    def publish_point_cloud(self, point_cloud, pub):
        """Publish the point cloud to RViz."""

        header = Header()
        header.frame_id = "odom"
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]

        header.stamp = rospy.Time.now()
        pc2_msg = pc2.create_cloud(header, fields, point_cloud)
        pub.publish(pc2_msg)

    def find_intersections(self, lines, d_min):
        if len(lines) < 2:
            return []

        points = []
        line_indices = []

        # Generate points along each line
        for i, (a0, a1) in enumerate(lines):
            line_points = self.line_to_points(a0, a1)
            points.append(line_points)
            line_indices.extend([i] * len(line_points))

        # Flatten the points array
        points = np.vstack(points)

        # Use DBSCAN to find clusters of close points
        clustering = DBSCAN(eps=d_min, min_samples=2).fit(points)
        labels = clustering.labels_

        intersections = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise points

            cluster_indices = np.where(labels == label)[0]
            involved_lines = set(line_indices[i] for i in cluster_indices)

            if len(involved_lines) > 1:
                cluster_points = points[cluster_indices]
                for i in involved_lines:
                    for j in involved_lines:
                        if i < j:
                            point_i = cluster_points[line_indices[cluster_indices] == i]
                            point_j = cluster_points[line_indices[cluster_indices] == j]
                            if point_i.size > 0 and point_j.size > 0:
                                midpoint_i = np.mean(point_i, axis=0)
                                midpoint_j = np.mean(point_j, axis=0)
                                intersections.append(
                                    (
                                        (i, j),
                                        (midpoint_i.tolist(), midpoint_j.tolist()),
                                        np.linalg.norm(midpoint_i - midpoint_j),
                                    )
                                )

        return intersections


def line_points_from_pose_array(pose_array: PoseArray):
    position1 = pose_array.poses[0].position
    position2 = pose_array.poses[1].position

    return (
        [position1.x, position1.y, position1.z],
        [position2.x, position2.y, position2.z],
    )


class LineBufferNode:
    def __init__(self):

        self.eps = rospy.get_param("~eps", 0.5)
        self.min_samples = rospy.get_param("~min_samples", 2)
        self.max_lines = rospy.get_param("~max_lines", 500)  # Default max lines of 10
        self.detection_namespace = rospy.get_param("~detection_namespace", "detection")

        self.color_map = {
            "red_buoy": ColorRGBA(0.5, 0.0, 0.0, 1.0),
            "path": ColorRGBA(1.0, 0.5, 0.0, 1.0),
            "bin_whole": ColorRGBA(0.0, 0.0, 0.5, 1.0),
            "torpedo_map": ColorRGBA(0.0, 1.0, 0.0, 1.0),
            "torpedo_hole": ColorRGBA(0.0, 0.5, 0.0, 1.0),
            "bin_red": ColorRGBA(1.0, 0.0, 0.0, 1.0),
            "bin_blue": ColorRGBA(0.0, 0.0, 1.0, 1.0),
            "gate_blue_arrow": ColorRGBA(0.3, 0.3, 1.0, 1.0),
            "gate_red_arrow": ColorRGBA(1.0, 0.3, 0.3, 1.0),
            "gate_middle_part": ColorRGBA(0.5, 0.5, 0.5, 1.0),
        }

        self.line_buffer = deque(maxlen=self.max_lines)
        self.point_cloud_pub = rospy.Publisher(
            "detection_point_cloud", PointCloud2, queue_size=10
        )

        self.is_odometry_valid_sub = rospy.Subscriber(
            "/taluy/sensors/dvl/is_valid", Bool, self.is_odometry_valid_callback
        )
        self.subscriber = rospy.Subscriber(
            "detection_lines", PoseArray, self.line_callback
        )
        self.pose_pub = rospy.Publisher("detection_poses", PoseArray, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()

    def is_odometry_valid_callback(self, msg):
        if not msg.data:
            self.clear_buffer()

    def clear_buffer(self):
        self.line_buffer.clear()

    def line_callback(self, msg):
        # Extract the start position for the new line
        start_position = self.extract_start_position(msg)

        # Check if the new line's start position is part of a cluster
        if self.is_part_of_cluster(start_position):
            rospy.loginfo(
                "Line start position is part of a cluster, not adding to buffer"
            )
            return

        # Append the new line and its start position to the buffer
        self.line_buffer.append((msg, start_position))

    def is_part_of_cluster(self, new_position):
        # Extract existing start positions
        existing_positions = np.array([pos for _, pos in self.line_buffer])

        if len(existing_positions) == 0:
            return False

        # Include the new position in the list
        positions = np.vstack([existing_positions, new_position])

        # Run DBSCAN on the positions
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(
            positions
        )

        # Check the label of the new position
        return labels[-1] != -1  # If the new position is part of a cluster, return True

    def extract_start_position(self, line):
        p1 = line.poses[0].position
        return np.array([p1.x, p1.y, p1.z])

    def calculate_intersection(self):
        if len(self.line_buffer) < 2:
            return None  # Not enough lines to calculate intersection

        lines = [line[0] for line in self.line_buffer]

        A = []
        B = []
        for line in lines:
            p1 = line.poses[0]
            p2 = line.poses[1]
            direction = np.array(
                [
                    p2.position.x - p1.position.x,
                    p2.position.y - p1.position.y,
                    p2.position.z - p1.position.z,
                ]
            )
            mid_point = (
                np.array(
                    [
                        p1.position.x + p2.position.x,
                        p1.position.y + p2.position.y,
                        p1.position.z + p2.position.z,
                    ]
                )
                / 2
            )
            length = np.linalg.norm(direction)
            if length == 0:
                continue  # Skip zero-length lines
            unit_direction = direction / length

            A.append(np.eye(3) - np.outer(unit_direction, unit_direction))
            B.append(
                np.dot(np.eye(3) - np.outer(unit_direction, unit_direction), mid_point)
            )

        A = np.array(A)
        B = np.array(B)

        A = np.sum(A, axis=0)
        B = np.sum(B, axis=0)

        try:
            # Solve for the point that minimizes the distance to all line segments using least squares
            intersection_point = np.linalg.lstsq(A, B, rcond=None)[0]
            return intersection_point
        except np.linalg.LinAlgError:
            return None  # In case of a cod_minmputational error

    def broadcast_tf(self, intersection):
        if intersection is None:
            return
        x, y, z = intersection
        # Broadcast the transform with no rotation
        self.tf_broadcaster.sendTransform(
            (x, y, z),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            self.detection_namespace + "_link",
            "odom",
        )

    def publish_pose(self, intersection):
        if intersection is None:
            return

        x, y, z = intersection
        pose_array = PoseArray()
        pose_array.header.frame_id = "odom"
        pose_array.header.stamp = rospy.Time.now()
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose_array.poses.append(pose)
        self.pose_pub.publish(pose_array)

    def publish_markers(self):
        lines = [line[0] for line in self.line_buffer]

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = self.detection_namespace
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02  # Line width
        marker.color = self.color_map[self.detection_namespace]

        for line in lines:
            p1 = line.poses[0].position
            p2 = line.poses[1].position
            marker.points.append(Point(p1.x, p1.y, p1.z))
            marker.points.append(Point(p2.x, p2.y, p2.z))

        self.marker_pub.publish(marker)

    def spin(self):
        rate = rospy.Rate(10)  # 10Hz
        intersector = Line3DIntersector()

        while not rospy.is_shutdown():

            lines = [line_points_from_pose_array(line[0]) for line in self.line_buffer]
            pc = intersector.generate_point_cloud(lines)
            intersector.publish_point_cloud(pc, self.point_cloud_pub)

            # print(lines)
            # intersections = intersector.find_intersections(lines, 0.5)  # meters

            # for intersection in intersections:
            #     ((i, j), (pA, pB), dist) = intersection
            #     print(
            #         f"Lines {i} and {j} intersect within {dist:.3f} units at points {pA} and {pB}"
            #     )

            # self.publish_pose(intersection)
            intersection = self.calculate_intersection()
            self.broadcast_tf(intersection)
            self.publish_markers()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("line_buffer_node")

    line_buffer = LineBufferNode()
    line_buffer.spin()
