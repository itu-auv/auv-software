#!/usr/bin/env python
import rospy
import numpy as np
import random
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import (
    PoseArray,
    Pose,
    PointStamped,
    Vector3Stamped,
)
from auv_msgs.msg import Pipes, Pipe
from visualization_msgs.msg import Marker, MarkerArray
from auv_navigation.slalom import cluster_pipes_with_ransac

# TODO - Left right is wrong


class Gate:
    """
    Represents a complete or partial slalom gate with left white, red, right white pipes.
    Each attribute is a Pipe or a placeholder if guessed.
    """

    def __init__(self, white_left, red, white_right, direction):
        self.white_left = white_left
        self.red = red
        self.white_right = white_right
        self.direction = direction  # unit 2D vector along the gate


class SlalomProcessorNode:
    def __init__(self):
        self.base_link_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.max_view_distance = rospy.get_param("~max_view_distance", 5.0)
        self.ransac_iterations = rospy.get_param("~ransac_iters", 500)
        self.line_distance_threshold = rospy.get_param("~line_tolerance", 0.1)
        self.min_pipe_cluster_size = rospy.get_param("~min_inliers", 2)
        self.navigation_mode = rospy.get_param("~navigation_mode", "left")
        self.red_white_distance = rospy.get_param("~red_white_distance", 1.5)
        self.gate_angle_tolerance_degrees = rospy.get_param(
            "~gate_angle_tolerance_degrees", 15.0
        )
        self.gate_angle_cos_threshold = np.cos(
            np.deg2rad(self.gate_angle_tolerance_degrees)
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pipe_sub = rospy.Subscriber(
            "/slalom_pipes", Pipes, self.callback_pipes, queue_size=1
        )
        self.centers_pub = rospy.Publisher("/slalom/centers", PoseArray, queue_size=10)
        self.centers_marker_pub = rospy.Publisher(
            "/slalom/centers_markers", MarkerArray, queue_size=10
        )

    def callback_pipes(self, msg):
        rospy.loginfo(
            "[SlalomProcessorNode] Received Pipes message with %d pipes",
            len(msg.pipes),
        )
        # System 1
        valid_pipes = self.filter_pipes_within_distance(msg.pipes)
        rospy.loginfo(
            f"Received {len(msg.pipes)} pipes. Processing for slalom waypoints."
        )
        raw_clusters = cluster_pipes_with_ransac(
            pipes=valid_pipes,
            min_pipe_cluster_size=self.min_pipe_cluster_size,
            base_link_frame=self.base_link_frame,
            odom_frame=self.odom_frame,
            tf_buffer=self.tf_buffer,
            ransac_iterations=self.ransac_iterations,
            gate_angle_cos_threshold=self.gate_angle_cos_threshold,
            line_distance_threshold=self.line_distance_threshold,
        )
        rospy.loginfo(
            "[SlalomProcessorNode] Found %d raw clusters (gate candidates)",
            len(raw_clusters),
        )

        # System 2
        sorted_clusters = [self.sort_pipes_along_line(c) for c in raw_clusters]
        rospy.loginfo("[SlalomProcessorNode] Sorted pipes along line for all clusters")
        validated = [self.validate_cluster(c) for c in sorted_clusters]
        rospy.loginfo("[SlalomProcessorNode] Validated clusters: %d", len(validated))
        gates = [self.create_gate_object(c) for c in validated]
        num_gates = sum(1 for g in gates if g is not None)
        rospy.loginfo(
            "[SlalomProcessorNode] Created %d complete Gate objects", num_gates
        )

        # System 3 (publish navigation targets)
        targets = [self.compute_navigation_targets(g) for g in gates if g is not None]
        rospy.loginfo(
            "[SlalomProcessorNode] Computed navigation targets for %d gates",
            len(targets),
        )
        self.publish_waypoints(targets)
        rospy.loginfo(
            "[SlalomProcessorNode] Published centers and markers for navigation targets"
        )

    def get_pipe_distance_from_base(self, position):
        """
        Transforms a pipe position (odom) to the robot's base frame and computes its distance to the robot base.
        """
        try:
            pipe_point = PointStamped()
            pipe_point.header.frame_id = self.odom_frame
            pipe_point.header.stamp = rospy.Time(0)
            pipe_point.point.x = position.x
            pipe_point.point.y = position.y
            pipe_point.point.z = position.z

            transform = self.tf_buffer.lookup_transform(
                self.base_link_frame,
                self.odom_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            transformed_point = tf2_geometry_msgs.do_transform_point(
                pipe_point, transform
            )

            # Euclidean distance from robot origin in its base_link frame
            return np.linalg.norm(
                [
                    transformed_point.point.x,
                    transformed_point.point.y,
                    transformed_point.point.z,
                ]
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed for distance computation: {e}")
            return None

    # --- System 1 functions ---
    def filter_pipes_within_distance(self, pipes):
        """
        Returns only those pipes within max_pipe_distance from the robot base.
        """
        close_pipes = []
        for pipe in pipes:
            distance = self.get_pipe_distance_from_base(pipe.position)
            if distance is not None and distance <= self.max_view_distance:
                close_pipes.append(pipe)
        return close_pipes

    # --- System 2 functions ---

    def sort_pipes_along_line(self, cluster):
        """
        Project each pipe onto the RANSAC line *after* forcing that line’s
        direction to point toward robot-right in the odom frame.
        The resulting sorted list will always run left→right from the robot’s view.
        """
        origin_point, unit_direction = cluster["line_model"]
        try:
            msg = Vector3Stamped()
            msg.header.frame_id = self.base_link_frame
            msg.header.stamp = rospy.Time(0)
            msg.vector.x, msg.vector.y, msg.vector.z = 0, 1, 0

            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_link_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            y_odom = tf2_geometry_msgs.do_transform_vector3(msg, transform).vector
            robot_y = np.array([y_odom.x, y_odom.y])
            robot_unit_y = robot_y / np.linalg.norm(robot_y)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"TF failed getting robot Y in odom: {e}")
            # fallback: assume odom Y-axis ≡ robot Y-axis
            robot_y = np.array([0.0, 1.0])

        robot_right = -robot_unit_y
        # Flip RANSAC line direction if it isn't pointing roughy toward robot right
        if np.dot(unit_direction, robot_right) < 0:
            unit_direction = -unit_direction

        projections = []
        for pipe in cluster["pipes"]:
            pipe_position_xy = np.array([pipe.position.x, pipe.position.y])
            projection_scalar = np.dot(
                (pipe_position_xy - origin_point), unit_direction
            )
            # projection_scalar represents how far along the line the pipe is located
            projections.append((projection_scalar, pipe))
        projections.sort(key=lambda item: item[0])
        sorted_pipes = [pipe for (_, pipe) in projections]
        return {"pipes": sorted_pipes, "line_model": (origin_point, unit_direction)}

    def validate_cluster(self, sorted_cluster):
        """
        Check the colors and spacing of the sorted pipes.
        Returns dict with 'pipes', 'model', and flags: 'is_complete', 'has_red', 'has_white_left', 'has_white_right'.
        """
        pipes = sorted_cluster["pipes"]
        num_pipes = len(pipes)
        colors = [p.color for p in pipes]
        flags = {
            "has_red": "red" in colors,
            "has_white_left": colors[0] == "white" if pipes else False,
            "has_white_right": colors[-1] == "white" if pipes else False,
        }
        flags["is_complete"] = (
            flags["has_red"]
            and flags["has_white_left"]
            and flags["has_white_right"]
            and (len(pipes) == 3)
        )
        return {"pipes": pipes, "line_model": sorted_cluster["line_model"], **flags}

    def create_gate_object(self, info):
        """
        Create a Gate object from a validated cluster info dict.
        Assumes info is from validate_cluster and is complete (3 pipes: white, red, white, in order).
        """
        if not info.get("is_complete", False):
            return None
        pipes = info["pipes"]
        if len(pipes) != 3:
            return None
        # By validate_cluster: [white, red, white]
        white_left = pipes[0]
        red = pipes[1]
        white_right = pipes[2]
        _, direction = info["line_model"]
        return Gate(white_left, red, white_right, direction)

    # --- System 3 functions ---
    def compute_navigation_targets(self, gate):
        """
        For a complete Gate, compute the navigation target Pose.
        The target is at the middle of the selected passage pipes.
        Its orientation is normal to the gate direction, pointing away from the robot.
        """

        if self.navigation_mode == "left":
            pipe_A = gate.white_left
            pipe_B = gate.red
        elif self.navigation_mode == "right":
            pipe_A = gate.red
            pipe_B = gate.white_right
        else:
            rospy.logwarn_throttle(
                5.0,
                f"Unknown navigation_mode: {self.navigation_mode}. Defaulting to left.",
            )
            pipe_A = gate.white_left
            pipe_B = gate.red
        pipe_A_pos = np.array([pipe_A.position.x, pipe_A.position.y])
        pipe_B_pos = np.array([pipe_B.position.x, pipe_B.position.y])
        midpoint_pos = (pipe_A_pos + pipe_B_pos) / 2
        waypoint = Pose()
        waypoint.position.x = midpoint_pos[0]
        waypoint.position.y = midpoint_pos[1]
        waypoint.position.z = (
            0.0  # Assuming 2D navigation for slalom, Z can be refined later
        )

        # Orientation
        # TODO: Calculate proper orientation: normal to the gate (gate.direction),
        # and pointing "away" from the robot.
        # gate.direction is a 2D unit vector [dx, dy]. A normal could be (-dy, dx) or (dy, -dx).
        # The "away" part needs the robot's current pose relative to the gate to determine.
        # For now, using a placeholder (identity quaternion: no rotation).
        waypoint.orientation.x = 0.0
        waypoint.orientation.y = 0.0
        waypoint.orientation.z = 0.0
        waypoint.orientation.w = 1.0

        return [waypoint]  # Return a list containing the single Pose target

    def publish_waypoints(self, targets_list):
        """Publish all gate targets as a concatenated PoseArray."""
        pa = PoseArray()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = self.odom_frame

        marker_array = MarkerArray()
        marker_id_counter = 0

        for (
            targets_for_one_gate
        ) in targets_list:  # targets_list is a list of [pose1, pose2] from each gate
            for pose_target in targets_for_one_gate:
                pa.poses.append(pose_target)

                marker = Marker()
                marker.header.frame_id = self.odom_frame
                marker.header.stamp = pa.header.stamp  # Use same timestamp
                marker.ns = "slalom_navigation_targets"
                marker.id = marker_id_counter
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.pose = pose_target
                marker.scale.x = 0.5  # Arrow length
                marker.scale.y = 0.1  # Arrow width
                marker.scale.z = 0.1  # Arrow height
                marker.color.a = 1.0  # Alpha
                marker.color.r = 0.0
                marker.color.g = 1.0  # Green
                marker.color.b = 0.0
                marker.lifetime = rospy.Duration(5.0)  # Disappear after 5 seconds
                marker_array.markers.append(marker)
                marker_id_counter += 1

        self.centers_pub.publish(pa)
        if marker_array.markers:  # Only publish if there are markers
            self.centers_marker_pub.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("slalom_processor")
    SlalomProcessorNode()
    rospy.spin()
