#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import (
    PoseArray,
    Pose,
)
from auv_msgs.msg import Pipes, Pipe
from visualization_msgs.msg import Marker, MarkerArray
from auv_navigation.slalom import (
    cluster_pipes_with_ransac,
    Gate,
    create_gate_object,
    compute_navigation_targets,
    filter_pipes_within_distance,
    sort_pipes_along_line,
    validate_cluster,
)
from typing import List, Optional, Dict, Any, Tuple


class SlalomProcessorNode:
    def __init__(self) -> None:
        # Frame names and parameters
        self.base_link_frame: str = rospy.get_param(
            "~robot_base_frame", "taluy/base_link"
        )
        self.odom_frame: str = rospy.get_param("~odom_frame", "odom")
        self.max_view_distance: float = rospy.get_param(
            "~max_view_distance", 5.0
        )  # Max distance to consider pipes
        self.ransac_iterations: int = rospy.get_param(
            "~ransac_iters", 500
        )  # RANSAC iterations in clustering
        self.line_distance_threshold: float = rospy.get_param(
            "~line_tolerance", 0.1
        )  # Distance threshold for line fitting
        self.min_pipe_cluster_size: int = rospy.get_param(
            "~min_inliers", 2
        )  # Minimum pipes to form a cluster
        self.navigation_mode: str = rospy.get_param(
            "~navigation_mode", "left"
        )  # Navigation mode: 'left' or 'right'
        self.red_white_distance: float = rospy.get_param(
            "~red_white_distance", 1.5
        )  # Expected distance between red and white pipes
        self.gate_angle_tolerance_degrees: float = rospy.get_param(
            "~gate_angle_tolerance_degrees", 15.0
        )
        self.gate_angle_cos_threshold: float = np.cos(
            np.deg2rad(self.gate_angle_tolerance_degrees)
        )  # threshold for angle comparison

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pipe_sub = rospy.Subscriber(
            "/slalom_pipes", Pipes, self.callback_pipes, queue_size=1
        )
        self.centers_pub = rospy.Publisher("/slalom/centers", PoseArray, queue_size=10)
        self.centers_marker_pub = rospy.Publisher(
            "/slalom/centers_markers", MarkerArray, queue_size=10
        )

    def callback_pipes(self, msg: Pipes) -> None:
        rospy.logdebug(
            "[SlalomProcessorNode] Received Pipes message with %d pipes",
            len(msg.pipes),
        )

        # Get current robot pose in odom frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_link_frame, rospy.Time(0)
            )
            robot_pose = Pose()
            robot_pose.position.x = transform.transform.translation.x
            robot_pose.position.y = transform.transform.translation.y
            robot_pose.position.z = transform.transform.translation.z
            robot_pose.orientation = transform.transform.rotation
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0, f"[SlalomProcessorNode] Could not get robot pose: {e}"
            )
            return  # Cannot proceed without robot pose

        # 1. Filter pipes based on distance from robot base
        valid_pipes = filter_pipes_within_distance(
            pipes=msg.pipes,
            max_view_distance=self.max_view_distance,
            odom_frame=self.odom_frame,
            base_link_frame=self.base_link_frame,
            tf_buffer=self.tf_buffer,
        )

        # 2. Cluster pipes with RANSAC
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
        rospy.logdebug(
            "[SlalomProcessorNode] Found %d raw clusters (gate candidates)",
            len(raw_clusters),
        )

        # 3. Sort pipes along the RANSAC line
        sorted_clusters = [
            sort_pipes_along_line(
                cluster=c,
                base_link_frame=self.base_link_frame,
                odom_frame=self.odom_frame,
                tf_buffer=self.tf_buffer,
            )
            for c in raw_clusters
        ]

        # 4. Validate clusters
        validated = [validate_cluster(sorted_cluster=c) for c in sorted_clusters]
        rospy.logdebug("[SlalomProcessorNode] Validated clusters: %d", len(validated))
        # 5. Create Gate objects from validated clusters
        gates = [create_gate_object(info=c) for c in validated]
        num_gates = sum(1 for g in gates if g is not None)
        rospy.logdebug(
            "[SlalomProcessorNode] Created %d complete Gate objects", num_gates
        )

        # 6. Compute navigation targets from gate objects
        targets = [
            compute_navigation_targets(
                gate=g,
                navigation_mode=self.navigation_mode,
                robot_pose=robot_pose,  # Pass the robot's pose
            )
            for g in gates
            if g is not None
        ]
        rospy.logdebug(
            "[SlalomProcessorNode] Computed navigation targets for %d gates",
            len(targets),
        )
        # 7. Publish the targets as PoseArray and markers
        self.publish_waypoints(targets)

    def publish_waypoints(self, targets_list: List[List[Pose]]) -> None:
        """Publish all gate targets as PoseArray and markers."""
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
