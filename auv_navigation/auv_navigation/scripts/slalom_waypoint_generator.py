#!/usr/bin/env python
import rospy
import numpy as np
import random
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from your_package.msg import (
    DetectedPipes,
    DetectedPipe,
)  # replace with your actual package


class Gate:
    """
    Represents a complete or partial slalom gate with left white, red, right white pipes.
    Each attribute is a DetectedPipe or a placeholder if guessed.
    """

    def __init__(self, white_left, red, white_right, direction):
        self.white_left = white_left
        self.red = red
        self.white_right = white_right
        self.direction = direction  # unit 2D vector along the gate


class SlalomProcessorNode:
    def __init__(self):
        # --- Parameters ---
        self.base_link_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.max_pipe_distance = rospy.get_param("~max_view_distance", 5.0)
        self.ransac_iterations = rospy.get_param("~ransac_iters", 500)
        self.line_distance_threshold = rospy.get_param("~line_tolerance", 0.1)
        self.min_pipe_cluster_size = rospy.get_param("~min_inliers", 2)
        self.navigation_mode = rospy.get_param(
            "~navigation_mode", "left"
        )  # 'left' or 'right'

        # TF2 for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers & Publishers
        self.pipe_sub = rospy.Subscriber(
            "/slalom_pipes", DetectedPipes, self.cb_detected_pipes, queue_size=1
        )
        self.centers_pub = rospy.Publisher("/slalom/centers", PoseArray, queue_size=10)

    def cb_detected_pipes(self, msg):
        # System 1
        valid_pipes = self.filter_pipes_within_distance(msg.detected_pipes)
        raw_clusters = self.cluster_pipes_with_ransac(valid_pipes)

        # System 2
        sorted_clusters = [self.sort_pipes_along_line(c) for c in raw_clusters]
        validated = [self.validate_cluster(c) for c in sorted_clusters]
        gates = [self.complete_gate(c) for c in validated]

        # System 3 (publish navigation targets)
        targets = [self.compute_navigation_targets(g) for g in gates]
        self.publish_centers(targets)

    def get_pipe_distance_from_base(self, position):
        """
        Transforms a pipe position (odom) to the robot's base frame and computes its distance to the origin.
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
    def filter_pipes_within_distance(self, detected_pipes):
        """
        Returns only those pipes within max_pipe_distance from the robot base.
        """
        close_pipes = []
        for pipe in detected_pipes:
            distance = self.get_pipe_distance_from_base(pipe.position)
            if distance is not None and distance <= self.max_pipe_distance:
                close_pipes.append(pipe)
        return close_pipes

    def cluster_pipes_with_ransac(self, pipes):
        """
        Uses a RANSAC-like approach to cluster pipes lying approximately on the same line (gate).
        Returns a list of clusters, each with member pipes and the fitted line.
        """
        unassigned_pipes = list(pipes)
        gate_clusters = []

        while len(unassigned_pipes) >= self.min_pipe_cluster_size:
            best_inliers = []
            best_line_model = None

            for _ in range(self.ransac_iterations):
                # Pick two unique pipes to define a line
                pipe_a, pipe_b = random.sample(unassigned_pipes, 2)
                pos_a = np.array([pipe_a.position.x, pipe_a.position.y])
                pos_b = np.array([pipe_b.position.x, pipe_b.position.y])
                line_vec = pos_b - pos_a
                length = np.linalg.norm(line_vec)
                if length == 0:
                    continue
                unit_direction = line_vec / length

                inliers = []
                for candidate_pipe in unassigned_pipes:
                    candidate_pos = np.array(
                        [candidate_pipe.position.x, candidate_pipe.position.y]
                    )
                    # Project candidate point onto the line
                    projected = (
                        pos_a
                        + np.dot(candidate_pos - pos_a, unit_direction) * unit_direction
                    )
                    distance_to_line = np.linalg.norm(candidate_pos - projected)
                    if distance_to_line < self.line_distance_threshold:
                        inliers.append(candidate_pipe)

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_line_model = (pos_a, unit_direction)

            if len(best_inliers) < self.min_pipe_cluster_size:
                break

            gate_clusters.append({"pipes": best_inliers, "line_model": best_line_model})
            # Remove clustered pipes for next iteration
            unassigned_pipes = [p for p in unassigned_pipes if p not in best_inliers]
        return gate_clusters

    # --- System 2 functions ---

    def sort_pipes_along_line(self, cluster):
        """
        Given a raw cluster {pipes, model=(point, dir)}, project each pipe onto the line
        and return a dict with 'pipes' sorted along the line and keep the same model.
        """
        origin, dir_unit = cluster["model"]
        projections = []
        for p in cluster["pipes"]:
            coord = np.array([p.position.x, p.position.y])
            t = np.dot((coord - origin), dir_unit)
            projections.append((t, p))
        projections.sort(key=lambda x: x[0])
        sorted_pipes = [p for (_, p) in projections]
        return {"pipes": sorted_pipes, "model": cluster["model"]}

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
        return {"pipes": pipes, "model": sorted_cluster["model"], **flags}

    def complete_gate(self, info):
        """
        Given validated info, fill in missing pipes using known spacing and orientation.
        Returns a Gate object with white_left, red, white_right, and direction.
        """
        origin, dir_unit = info["model"]
        pipes = info["pipes"]
        # Initialize placeholders
        w_left = r = w_right = None
        # Assign seen pipes
        for p in pipes:
            if p.color == "red":
                r = p
            elif w_left is None:
                w_left = p
            else:
                w_right = p
                #! Assumes first white pipe is left, second is right
        # Guess missing
        if r and not w_left:
            # red seen only: place whites at ±d along dir
            pos = np.array([r.position.x, r.position.y])
            w_left = self._make_fake_pipe(
                pos - dir_unit * self.red_white_spacing, "white"
            )
            w_right = self._make_fake_pipe(
                pos + dir_unit * self.red_white_spacing, "white"
            )
        elif r and w_left and not w_right:
            # red + one white
            pos_r = np.array([r.position.x, r.position.y])
            pos_w = np.array([w_left.position.x, w_left.position.y])
            # determine which side is missing
            side = np.sign(np.dot((pos_w - pos_r), dir_unit))
            missing_pos = pos_r - side * dir_unit * self.red_white_spacing
            w_right = self._make_fake_pipe(missing_pos, "white")
        # More cases (only whites, etc.) can be added similarly
        return Gate(w_left, r, w_right, dir_unit)

    def _make_fake_pipe(self, pos_xy, color):
        """Create a DetectedPipe-like placeholder at pos_xy with given color."""
        fake = DetectedPipe()
        fake.color = color
        fake.position.x, fake.position.y = float(pos_xy[0]), float(pos_xy[1])
        fake.position.z = 0.0
        return fake

    # --- System 3 functions ---
    def compute_navigation_targets(self, gate):
        """
        For a complete Gate, return the two Pose targets based on navigation_mode.
        """
        p1 = gate.white_left if self.navigation_mode == "left" else gate.red
        p2 = gate.red if self.navigation_mode == "left" else gate.white_right
        pose1 = self._pipe_to_pose(p1)
        pose2 = self._pipe_to_pose(p2)
        return [pose1, pose2]

    def publish_centers(self, targets_list):
        """Publish all gate targets as a concatenated PoseArray."""
        pa = PoseArray()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = self.odom_frame
        for targets in targets_list:
            for pose in targets:
                pa.poses.append(pose)
        self.centers_pub.publish(pa)

    def _pipe_to_pose(self, pipe):
        """Convert a DetectedPipe to a geometry_msgs/Pose (position only)."""
        pose = Pose()
        pose.position.x = pipe.position.x
        pose.position.y = pipe.position.y
        pose.position.z = pipe.position.z
        return pose


if __name__ == "__main__":
    rospy.init_node("slalom_processor")
    SlalomProcessorNode()
    rospy.spin()
