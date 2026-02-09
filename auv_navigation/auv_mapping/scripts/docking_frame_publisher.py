#!/usr/bin/env python3
"""
Docking Frame Publisher

Publishes target frames for the docking mission.

Frames published (ArUco based, toggle: toggle_docking_trajectory):
- docking_puck_target: Final docking position (0.5m above docking_station by default)
"""

import numpy as np
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from dynamic_reconfigure.server import Server

from auv_mapping.cfg import DockingTrajectoryConfig


class DockingFramePublisher:
    def __init__(self):
        rospy.init_node("docking_frame_publisher")

        self.trajectory_enabled = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.parent_frame = rospy.get_param("~parent_frame", "docking_station")
        self.puck_target_frame = rospy.get_param(
            "~puck_target_frame", "docking_puck_target"
        )

        self.puck_offset_x = 0.0
        self.puck_offset_y = 0.0
        self.puck_offset_z = 0.5

        self.reconfigure_server = Server(
            DockingTrajectoryConfig, self.reconfigure_callback
        )

        self.toggle_trajectory_service = rospy.Service(
            "toggle_docking_trajectory", SetBool, self.handle_toggle_trajectory_service
        )

        rospy.loginfo("Docking Frame Publisher initialized")

    def reconfigure_callback(self, config, level):
        self.puck_offset_x = config.puck_offset_x
        self.puck_offset_y = config.puck_offset_y
        self.puck_offset_z = config.puck_offset_z
        rospy.loginfo("Docking trajectory parameters updated via dynamic reconfigure")
        return config

    def handle_toggle_trajectory_service(self, req):
        self.trajectory_enabled = req.data
        message = (
            f"Docking trajectory frame publishing set to: {self.trajectory_enabled}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def get_pose(self, transform: TransformStamped) -> Pose:
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose, stamp: rospy.Time
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def apply_offset_to_pose(
        self, base_pose: Pose, offset_x: float, offset_y: float, offset_z: float
    ) -> Pose:
        offset = np.array([offset_x, offset_y, offset_z, 0.0])
        q = [
            base_pose.orientation.x,
            base_pose.orientation.y,
            base_pose.orientation.z,
            base_pose.orientation.w,
        ]
        rot = tf.transformations.quaternion_matrix(q)
        rotated = rot.dot(offset)

        new_pose = Pose()
        new_pose.position.x = base_pose.position.x + rotated[0]
        new_pose.position.y = base_pose.position.y + rotated[1]
        new_pose.position.z = base_pose.position.z + rotated[2]
        new_pose.orientation = base_pose.orientation
        return new_pose

    def publish_docking_frames(self):
        try:
            docking_station_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.parent_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0, f"TF lookup for {self.parent_frame} failed: {e}"
            )
            return

        base_pose = self.get_pose(docking_station_tf)

        current_time = rospy.Time.now()

        puck_pose = self.apply_offset_to_pose(
            base_pose,
            self.puck_offset_x,
            self.puck_offset_y,
            self.puck_offset_z,
        )
        puck_transform = self.build_transform_message(
            self.puck_target_frame, puck_pose, current_time
        )

        self.tf_broadcaster.sendTransform(puck_transform)

    def spin(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if self.trajectory_enabled:
                self.publish_docking_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DockingFramePublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
