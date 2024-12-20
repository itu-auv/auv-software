import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rospy
import tf2_ros

from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PathPlanner:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_straight_path(self, source_frame, target_frame, angle_offset=0.0, num_points=20):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            trans = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            )
            rot = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )

            # Apply angle offset
            roll, pitch, yaw = euler_from_quaternion(rot)
            yaw += angle_offset
            new_rot = quaternion_from_euler(roll, pitch, yaw)

            path = Path()
            path.header.frame_id = target_frame
            for i in range(num_points):
                pose = PoseStamped()
                pose.header.frame_id = target_frame
                pose.pose.position.x = trans[0] * (i / (num_points - 1))
                pose.pose.position.y = trans[1] * (i / (num_points - 1))
                pose.pose.position.z = trans[2] * (i / (num_points - 1))
                pose.pose.orientation.x = new_rot[0]
                pose.pose.orientation.y = new_rot[1]
                pose.pose.orientation.z = new_rot[2]
                pose.pose.orientation.w = new_rot[3]
                path.poses.append(pose)

            return path

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform between {source_frame} and {target_frame}: {str(e)}")
            return None