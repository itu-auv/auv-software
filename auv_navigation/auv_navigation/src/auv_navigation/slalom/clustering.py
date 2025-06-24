import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Vector3Stamped
from auv_msgs.msg import Pipe
from typing import Dict, Any


def sort_pipes_along_line(
    cluster: Dict[str, Any],
    base_link_frame: str,
    odom_frame: str,
    tf_buffer: tf2_ros.Buffer,
) -> Dict[str, Any]:
    """
    Project each pipe onto the RANSAC line *after* forcing that line’s
    direction to point toward robot-right in the odom frame.
    The resulting sorted list will always run left→right from the robot’s view.
    """
    origin_point, unit_direction = cluster["line_model"]
    try:
        msg = Vector3Stamped()
        msg.header.frame_id = base_link_frame
        msg.header.stamp = rospy.Time(0)
        msg.vector.x, msg.vector.y, msg.vector.z = 0, 1, 0

        transform = tf_buffer.lookup_transform(
            odom_frame,
            base_link_frame,
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
        projection_scalar = np.dot((pipe_position_xy - origin_point), unit_direction)
        # projection_scalar represents how far along the line the pipe is located
        projections.append((projection_scalar, pipe))
    projections.sort(key=lambda item: item[0])
    sorted_pipes = [pipe for (_, pipe) in projections]
    return {"pipes": sorted_pipes, "line_model": (origin_point, unit_direction)}


def validate_cluster(sorted_cluster: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check the colors of the sorted pipes to determine gate characteristics.
    Returns dict with 'pipes', 'line_model', and flags: 'is_complete', 'has_red', 'has_white_left', 'has_white_right'.
    Assumes pipes are sorted from left to right from robot's perspective.
    """
    pipes = sorted_cluster["pipes"]
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
