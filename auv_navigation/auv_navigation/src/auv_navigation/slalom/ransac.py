import rospy
import numpy as np
import random
import tf2_ros  # For exceptions
from geometry_msgs.msg import Vector3Stamped

# from auv_msgs.msg import Pipe # Assuming Pipe objects are passed in and have a 'position' attribute
from typing import Tuple, List, Optional


def cluster_pipes_with_ransac(
    pipes: list,  # List of Pipe objects
    min_pipe_cluster_size: int,
    base_link_frame: str,
    odom_frame: str,
    tf_buffer,  # tf2_ros.Buffer
    ransac_iterations: int,
    gate_angle_cos_threshold: float,
    line_distance_threshold: float,
) -> list:
    """
    Uses a RANSAC-like approach to cluster pipes lying approximately on the same line (gate).
    Filters lines based on their orientation relative to the robot's Y-axis.
    Returns a list of clusters, each with member pipes and the fitted line.
    """

    unassigned_pipes = list(
        pipes
    )  # Working copy of pipes that have not been assigned to a cluster
    gate_clusters = []

    do_directional_check, robot_y_axis_odom = _compute_robot_y_axis(
        len(unassigned_pipes),
        min_pipe_cluster_size,
        base_link_frame,
        odom_frame,
        tf_buffer,
    )

    while len(unassigned_pipes) >= min_pipe_cluster_size:
        best_inliers: list = []
        best_line_model = None

        for _ in range(ransac_iterations):

            model = _sample_line_model(unassigned_pipes)
            if model is None:
                break
            pos_a, unit_direction = model

            # Orientation pruning
            if do_directional_check and robot_y_axis_odom is not None:
                if (
                    abs(np.dot(unit_direction, robot_y_axis_odom))
                    < gate_angle_cos_threshold
                ):
                    continue

            inliers = _find_inliers(
                unassigned_pipes,
                pos_a,
                unit_direction,
                line_distance_threshold,
            )

            if len(inliers) > len(best_inliers):
                best_inliers, best_line_model = inliers, model

        if len(best_inliers) < min_pipe_cluster_size:
            break

        # Only add if a valid model was found for the inliers
        if best_line_model is not None:
            gate_clusters.append({"pipes": best_inliers, "line_model": best_line_model})
            # Remove clustered pipes for next iteration
            unassigned_pipes = [p for p in unassigned_pipes if p not in best_inliers]
        else:
            # If no model was found, but we have enough inliers (edge case),
            # we can't form a cluster. Break to avoid issues.
            # Or, treat these pipes as unassignable for this iteration.
            # For safety, let's just break if best_line_model is None and we expected one.
            # This implies that even if inliers were found, no line could be fit.
            rospy.logwarn_throttle(
                5.0,
                "[RANSAC] Found sufficient inliers but no line model. Skipping cluster.",
            )
            break  # Or, could remove best_inliers from unassigned_pipes and continue,
            # but breaking is safer if this state is unexpected.

    return gate_clusters


def _compute_robot_y_axis(
    num_pipes: int,
    min_pipe_cluster_size: int,
    base_link_frame: str,
    odom_frame: str,
    tf_buffer,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Return (do_directional_check, robot_y_axis_2d).

    If there aren’t enough pipes, or TF fails, the first item is False and the
    second is None.  Otherwise the 2‑D unit vector of the robot’s body‑frame
    Y‑axis projected into *odom* is returned.
    """

    if num_pipes < min_pipe_cluster_size:
        return False, None

    vec_stamped = Vector3Stamped()
    vec_stamped.header.stamp = rospy.Time(0)
    vec_stamped.header.frame_id = base_link_frame
    vec_stamped.vector.x = 0.0
    vec_stamped.vector.y = 1.0  # body‑frame Y
    vec_stamped.vector.z = 0.0

    try:
        transformed = tf_buffer.transform(
            vec_stamped,
            odom_frame,
            timeout=rospy.Duration(0.1),
        )
        vec = np.array([transformed.vector.x, transformed.vector.y])
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            rospy.logwarn_throttle(
                10.0,
                "Robot Y‑axis projected into odom is near‑zero; skipping direction check.",
            )
            return False, None
        return True, vec / norm
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
        tf2_ros.TransformException,
    ) as exc:
        rospy.logwarn_throttle(
            10.0,
            f"TF transform failed for robot Y‑axis: {exc}.  Skipping directional check.",
        )
        return False, None


#! Naming is poor?
def _sample_line_model(unassigned_pipes: List, max_attempts: int = 10):
    """Randomly pick two distinct pipes and return (pos_a, unit_direction).

    Returns None if a non‑degenerate pair cannot be found after *max_attempts*.
    """
    if len(unassigned_pipes) < 2:
        return None

    for _ in range(max_attempts):
        pipe_a, pipe_b = random.sample(unassigned_pipes, 2)
        pos_a = np.array([pipe_a.position.x, pipe_a.position.y])
        pos_b = np.array([pipe_b.position.x, pipe_b.position.y])
        vec = pos_b - pos_a
        length = np.linalg.norm(vec)
        if length > 0:
            return pos_a, vec / length
    return None


def _find_inliers(
    pipes: List,
    pos_a: np.ndarray,
    unit_dir: np.ndarray,
    threshold: float,
) -> List:
    """Return list of pipes closer than *threshold* to the infinite line."""
    inliers = []
    for p in pipes:
        pos = np.array([p.position.x, p.position.y])
        proj = pos_a + np.dot(pos - pos_a, unit_dir) * unit_dir
        if np.linalg.norm(pos - proj) < threshold:
            inliers.append(p)
    return inliers
