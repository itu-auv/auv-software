#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker
from collections import deque
import numpy as np
import tf
from std_msgs.msg import Bool
from sklearn.cluster import DBSCAN


class LineBufferNode:
    def __init__(self):

        self.eps = rospy.get_param("~eps", 0.5)
        self.min_samples = rospy.get_param("~min_samples", 2)
        self.max_lines = rospy.get_param("~max_lines", 500)  # Default max lines of 10
        self.line_buffer = deque(maxlen=self.max_lines)

        self.is_odometry_valid_sub = rospy.Subscriber(
            "/taluy/sensors/dvl/is_valid", Bool, self.is_odometry_valid_callback
        )
        self.subscriber = rospy.Subscriber(
            "detection_lines", PoseArray, self.line_callback
        )
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
        intersection_points = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p1_start = lines[i].poses[0].position
                p1_end = lines[i].poses[1].position
                p2_start = lines[j].poses[0].position
                p2_end = lines[j].poses[1].position

                # Line segment 1: p1_start -> p1_end
                # Line segment 2: p2_start -> p2_end

                # Calculate the intersection of the two line segments
                intersect_point = self.segment_intersection(
                    p1_start, p1_end, p2_start, p2_end
                )
                if intersect_point:
                    intersection_points.append(intersect_point)

        if not intersection_points:
            return None  # No intersections found

        # Return the average intersection point
        intersection_array = np.array(intersection_points)
        average_intersection = np.mean(intersection_array, axis=0)
        return average_intersection

    def segment_intersection(self, p1_start, p1_end, p2_start, p2_end):
        """Calculate the intersection point of two line segments if it exists."""
        x1, y1 = p1_start.x, p1_start.y
        x2, y2 = p1_end.x, p1_end.y
        x3, y3 = p2_start.x, p2_start.y
        x4, y4 = p2_end.x, p2_end.y

        # Calculate the denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Lines are parallel

        # Calculate the intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if the intersection point is within the bounds of both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return np.array([intersection_x, intersection_y, 0])
        else:
            return None

    def broadcast_tf(self, intersection):
        if intersection is not None:
            x, y, z = intersection
            # Broadcast the transform with no rotation
            self.tf_broadcaster.sendTransform(
                (x, y, z),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "detection_link",
                "odom",
            )

    def publish_markers(self):
        lines = [line[0] for line in self.line_buffer]

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lines"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue

        for line in lines:
            p1 = line.poses[0].position
            p2 = line.poses[1].position
            marker.points.append(Point(p1.x, p1.y, p1.z))
            marker.points.append(Point(p2.x, p2.y, p2.z))

        self.marker_pub.publish(marker)

    def spin(self):
        rate = rospy.Rate(10)  # 10Hz

        while not rospy.is_shutdown():
            intersection = self.calculate_intersection()
            self.broadcast_tf(intersection)
            self.publish_markers()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("line_buffer_node")

    line_buffer = LineBufferNode()
    line_buffer.spin()
