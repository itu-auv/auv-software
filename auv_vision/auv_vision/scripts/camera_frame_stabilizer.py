#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import angles


class CameraFrameStabilizer:
    def __init__(self):
        rospy.init_node("camera_frame_stabilizer", anonymous=True)
        rospy.loginfo("Camera frame stabilizer node started.")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.odom_subscriber = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, queue_size=1
        )

        self.parent_frame = rospy.get_param(
            "~parent_frame", "taluy/base_link/front_camera_optical_link"
        )
        self.child_frame = rospy.get_param(
            "~child_frame", "camera_optical_link_stabilized"
        )
        rospy.loginfo(
            "Publishing transform from '%s' to '%s'",
            self.parent_frame,
            self.child_frame,
        )

    def odom_callback(self, odom_msg):
        q = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(q)

        correction_rad = angles.shortest_angular_distance(0.0, roll)

        t = TransformStamped()

        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        q_rot = quaternion_from_euler(correction_rad, 0, 0)
        t.transform.rotation.x = q_rot[0]
        t.transform.rotation.y = q_rot[1]
        t.transform.rotation.z = q_rot[2]
        t.transform.rotation.w = q_rot[3]

        self.tf_broadcaster.sendTransform(t)


if __name__ == "__main__":
    try:
        node = CameraFrameStabilizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera frame stabilizer node terminated.")
