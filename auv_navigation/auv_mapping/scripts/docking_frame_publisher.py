#!/usr/bin/env python3
"""
Docking Frame Publisher

Publishes target frames for the docking mission.

Frames published (Phase B - ArUco based, toggle: toggle_docking_trajectory):
- docking_approach_target: Approach position (1m above docking_station by default)
- docking_puck_target: Final docking position (0.5m above docking_station by default)

Frames published (Phase A - YOLO based, toggle: toggle_docking_approach_frame):
- docking_station_approach: Position at docking_station_link with orientation facing from robot to station
"""

import numpy as np
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from dynamic_reconfigure.server import Server

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from auv_mapping.cfg import DockingTrajectoryConfig


class DockingFramePublisher:
    def __init__(self):
        rospy.init_node("docking_frame_publisher")

        # Enable flags for different frame types
        self.trajectory_enabled = False  # Phase B: ArUco-based target frames
        self.approach_frame_enabled = False  # Phase A: YOLO-based approach frame

        # TF2 buffer for frame lookups
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Service proxy to set object transforms (provided by object_map_tf_server_node)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        # Frame configuration - Phase B (ArUco-based)
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.parent_frame = rospy.get_param("~parent_frame", "docking_station")
        self.approach_target_frame = rospy.get_param(
            "~approach_target_frame", "docking_approach_target"
        )
        self.puck_target_frame = rospy.get_param(
            "~puck_target_frame", "docking_puck_target"
        )

        # Frame configuration - Phase A (YOLO-based)
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.docking_station_frame = rospy.get_param(
            "~docking_station_frame", "docking_station_link"
        )
        self.station_approach_frame = rospy.get_param(
            "~station_approach_frame", "docking_station_approach"
        )

        # Default offsets (will be updated by dynamic reconfigure)
        self.approach_offset_x = 0.0
        self.approach_offset_y = 0.0
        self.approach_offset_z = 1.0
        self.puck_offset_x = 0.0
        self.puck_offset_y = 0.0
        self.puck_offset_z = 0.5

        # Dynamic reconfigure server
        self.reconfigure_server = Server(
            DockingTrajectoryConfig, self.reconfigure_callback
        )

        # Enable/disable services
        self.toggle_trajectory_service = rospy.Service(
            "toggle_docking_trajectory", SetBool, self.handle_toggle_trajectory_service
        )
        self.toggle_approach_frame_service = rospy.Service(
            "toggle_docking_approach_frame",
            SetBool,
            self.handle_toggle_approach_frame_service,
        )

        rospy.loginfo("Docking Frame Publisher initialized")

    def reconfigure_callback(self, config, level):
        """Callback for dynamic reconfigure parameters."""
        self.approach_offset_x = config.approach_offset_x
        self.approach_offset_y = config.approach_offset_y
        self.approach_offset_z = config.approach_offset_z
        self.puck_offset_x = config.puck_offset_x
        self.puck_offset_y = config.puck_offset_y
        self.puck_offset_z = config.puck_offset_z
        rospy.loginfo("Docking trajectory parameters updated via dynamic reconfigure")
        return config

    def handle_toggle_trajectory_service(self, req):
        """Service callback to enable/disable Phase B (ArUco) trajectory frames."""
        self.trajectory_enabled = req.data
        message = f"Docking trajectory frame publishing set to: {self.trajectory_enabled}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_toggle_approach_frame_service(self, req):
        """Service callback to enable/disable Phase A (YOLO) approach frame."""
        self.approach_frame_enabled = req.data
        message = f"Docking approach frame publishing set to: {self.approach_frame_enabled}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def get_pose(self, transform: TransformStamped) -> Pose:
        """Convert a TransformStamped to a Pose."""
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        """Build a TransformStamped message."""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform: TransformStamped):
        """Send transform via the set_object_transform service."""
        req = SetObjectTransformRequest()
        req.transform = transform
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(
                    f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def apply_offset_to_pose(
        self, base_pose: Pose, offset_x: float, offset_y: float, offset_z: float
    ) -> Pose:
        """
        Apply an offset to a pose. The offset is in the parent frame's coordinate system.
        Since docking_station is oriented with Z pointing up, offsets are straightforward.
        """
        new_pose = Pose()
        new_pose.position.x = base_pose.position.x + offset_x
        new_pose.position.y = base_pose.position.y + offset_y
        new_pose.position.z = base_pose.position.z + offset_z
        new_pose.orientation = base_pose.orientation
        return new_pose

    def publish_docking_frames(self):
        """Publish the docking target frames relative to docking_station."""
        try:
            # Look up docking_station transform in odom frame
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

        # Create approach target frame
        approach_pose = self.apply_offset_to_pose(
            base_pose,
            self.approach_offset_x,
            self.approach_offset_y,
            self.approach_offset_z,
        )
        approach_transform = self.build_transform_message(
            self.approach_target_frame, approach_pose
        )
        self.send_transform(approach_transform)

        # Create puck target frame
        puck_pose = self.apply_offset_to_pose(
            base_pose,
            self.puck_offset_x,
            self.puck_offset_y,
            self.puck_offset_z,
        )
        puck_transform = self.build_transform_message(
            self.puck_target_frame, puck_pose
        )
        self.send_transform(puck_transform)

    def publish_approach_frame(self):
        """
        Publish approach frame for Phase A (YOLO-based docking approach).

        Creates a frame at docking_station_link position with orientation
        facing from the robot toward the station. This allows the AUV to
        approach while facing the target.

        Pattern follows torpedo_frame_publisher.py create_target_frame().
        """
        try:
            # Look up robot and docking station positions in odom frame
            robot_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
            station_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.docking_station_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0, f"TF lookup for approach frame failed: {e}"
            )
            return

        # Get positions
        robot_pos = np.array(
            [
                robot_tf.transform.translation.x,
                robot_tf.transform.translation.y,
            ]
        )
        station_pos = np.array(
            [
                station_tf.transform.translation.x,
                station_tf.transform.translation.y,
            ]
        )

        # Compute direction from robot to station
        direction = station_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            # Too close, skip publishing to avoid division by zero
            rospy.logwarn_throttle(
                5.0, "Robot and station too close, skipping approach frame"
            )
            return

        # Compute yaw angle facing toward the station
        facing_yaw = np.arctan2(direction[1], direction[0])
        quaternion = tf.transformations.quaternion_from_euler(0, 0, facing_yaw)

        # Create pose at station position with facing orientation
        approach_pose = Pose()
        approach_pose.position.x = station_tf.transform.translation.x
        approach_pose.position.y = station_tf.transform.translation.y
        approach_pose.position.z = station_tf.transform.translation.z
        approach_pose.orientation.x = quaternion[0]
        approach_pose.orientation.y = quaternion[1]
        approach_pose.orientation.z = quaternion[2]
        approach_pose.orientation.w = quaternion[3]

        # Build and send transform
        approach_transform = self.build_transform_message(
            self.station_approach_frame, approach_pose
        )
        self.send_transform(approach_transform)

    def spin(self):
        """Main loop."""
        rate = rospy.Rate(10.0)  # 10 Hz
        while not rospy.is_shutdown():
            if self.trajectory_enabled:
                self.publish_docking_frames()
            if self.approach_frame_enabled:
                self.publish_approach_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DockingFramePublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
