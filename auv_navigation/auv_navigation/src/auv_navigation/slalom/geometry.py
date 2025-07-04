import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from auv_msgs.msg import Pipe
from typing import List, Optional, Any


def get_pipe_distance_from_base(
    position: Any, odom_frame: str, base_link_frame: str, tf_buffer: tf2_ros.Buffer
) -> Optional[float]:
    """
    Transforms a pipe position (odom) to the robot's base frame and computes its distance to the robot base.
    """
    try:
        pipe_point = PointStamped()
        pipe_point.header.frame_id = odom_frame
        pipe_point.header.stamp = rospy.Time(0)
        pipe_point.point.x = position.x
        pipe_point.point.y = position.y
        pipe_point.point.z = position.z

        transform = tf_buffer.lookup_transform(
            base_link_frame,
            odom_frame,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        transformed_point = tf2_geometry_msgs.do_transform_point(pipe_point, transform)

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


def filter_pipes_within_distance(
    pipes: List[Pipe],
    max_view_distance: float,
    odom_frame: str,
    base_link_frame: str,
    tf_buffer: tf2_ros.Buffer,
) -> List[Pipe]:
    """
    Returns only those pipes within max_view_distance from the robot base.
    """
    close_pipes = []
    for pipe in pipes:
        distance = get_pipe_distance_from_base(
            pipe.position, odom_frame, base_link_frame, tf_buffer
        )
        if distance is not None and distance <= max_view_distance:
            close_pipes.append(pipe)
    return close_pipes
