#!/usr/bin/env python3
"""
Responsible for calculating and publishing spatial lane boundaries.
The lanes are defined as two parallel lines, in odom frame.
"""
import rospy
import math
from typing import Optional
from geometry_msgs.msg import Point, Vector3
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray
from auv_msgs.msg import LaneBoundaries


class LaneCalculatorNode:

    def __init__(self) -> None:
        rospy.init_node("lane_calculator_node")
        rospy.loginfo("Lane Calculator Node Started")

        self.forward_lane_angle: float = rospy.get_param("~forward_lane_angle", 0.0)
        self.lane_distance_to_left: float = rospy.get_param(
            "~lane_distance_to_left", 2.5
        )
        self.lane_distance_to_right: float = rospy.get_param(
            "~lane_distance_to_right", 2.5
        )

        self.lane_boundaries_pub: rospy.Publisher = rospy.Publisher(
            "lane_boundaries", LaneBoundaries, queue_size=10, latch=True
        )
        self.marker_pub: rospy.Publisher = rospy.Publisher(
            "visualization/lane_markers", MarkerArray, queue_size=10, latch=True
        )

        self.lane_boundaries: Optional[LaneBoundaries] = None
        rospy.Service("calculate_lanes", Trigger, self.calculate_lanes_service)

        # Timer to continuously publish markers for RViz
        rospy.Timer(rospy.Duration(4.0), self.publish_lane_markers_callback)

    def publish_lane_markers_callback(self, event=None):
        if self.lane_boundaries is not None:
            self.publish_lane_markers()

    def calculate_lanes_service(self, req: Trigger) -> TriggerResponse:
        """
        This service callback calculates the lane boundaries. The calculation is
        static and relative to the odom frame's origin (0,0,0).
        """
        rospy.loginfo(
            "Calculating static lane boundaries..."
        )  #! Remove this log in production

        # 1. Define the direction vector of the lane's center line.
        lane_direction = Vector3(
            x=math.cos(self.forward_lane_angle),
            y=math.sin(self.forward_lane_angle),
            z=0,
        )

        # 2. Calculate the perpendicular vector to shift the boundaries.
        perp_direction = Vector3(x=-lane_direction.y, y=lane_direction.x, z=0)

        # 3. Calculate a point on each lane boundary
        # New point = (origin) + distance × (direction)
        left_lane_point = Point(
            x=self.lane_distance_to_left * perp_direction.x,
            y=self.lane_distance_to_left * perp_direction.y,
            z=0,
        )

        right_lane_point = Point(
            x=-self.lane_distance_to_right * perp_direction.x,
            y=-self.lane_distance_to_right * perp_direction.y,
            z=0,
        )

        self.lane_boundaries = LaneBoundaries()
        self.lane_boundaries.header.stamp = rospy.Time.now()
        self.lane_boundaries.header.frame_id = "odom"
        self.lane_boundaries.left_lane_point = left_lane_point
        self.lane_boundaries.left_lane_direction = lane_direction
        self.lane_boundaries.right_lane_point = right_lane_point
        self.lane_boundaries.right_lane_direction = lane_direction

        self.lane_boundaries_pub.publish(self.lane_boundaries)
        self.publish_lane_markers()

        rospy.loginfo("Lane boundaries published.")  #! Remove this log in production
        return TriggerResponse(
            success=True, message="Lane boundaries calculated and published."
        )

    def publish_lane_markers(self) -> None:
        if self.lane_boundaries is None:
            return

        marker_array = MarkerArray()

        boundaries = [
            (
                self.lane_boundaries.left_lane_point,
                self.lane_boundaries.left_lane_direction,
            ),
            (
                self.lane_boundaries.right_lane_point,
                self.lane_boundaries.right_lane_direction,
            ),
        ]

        for i, (point, direction) in enumerate(boundaries):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "lanes"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            start_point = Point()
            start_point.x = point.x - direction.x * 100
            start_point.y = point.y - direction.y * 100
            start_point.z = point.z

            end_point = Point()
            end_point.x = point.x + direction.x * 100
            end_point.y = point.y + direction.y * 100
            end_point.z = point.z

            marker.points.append(start_point)
            marker.points.append(end_point)

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def run(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    try:
        node = LaneCalculatorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
