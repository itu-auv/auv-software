#!/usr/bin/env python3
"""
Valve Trajectory Publisher
--------------------------
Reads the `tac/valve` TF (Kalman-filtered output of the keypoint pose pipeline)
and publishes two derived frames via the `set_object_transform` service
(Object Map TF Server):

  1. valve_approach_frame  — Offset along the valve's 3D face normal by
     `approach_offset` metres, oriented to face into the valve (full
     roll/pitch/yaw). The robot aligns to this frame perpendicular to the
     valve face from a comfortable distance.

  2. valve_contact_frame   — Offset along the same 3D normal by
     `contact_offset` metres, with the same facing orientation. The robot
     aligns to this frame for the final contact pose.

Params:
  ~valve_frame    (str, default "tac/valve")    — source frame to read
  ~approach_frame (str, default "valve_approach_frame")
  ~contact_frame  (str, default "valve_contact_frame")
  ~start_enabled  (bool, default False)         — when True, both phases
      are enabled at startup and published continuously without anything
      calling the SetBool services. Used by the debug launch for bag-replay
      visualization. Production smach behavior leaves this False.
"""

import numpy as np
import rospy
import tf.transformations
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_matrix

from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
)

from auv_mapping.cfg import ValveTrajectoryConfig


class ValveTrajectoryPublisherNode:
    def __init__(self):
        rospy.init_node("valve_trajectory_publisher_node")

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Object map service (same pattern as torpedo_frame_publisher)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        rospy.loginfo("Waiting for set_object_transform service...")
        self.set_object_transform_service.wait_for_service()

        # Frame names
        self.odom_frame = "odom"
        self.valve_frame = rospy.get_param("~valve_frame", "tac/valve")
        self.approach_frame = rospy.get_param("~approach_frame", "valve_approach_frame")
        self.contact_frame = rospy.get_param("~contact_frame", "valve_contact_frame")

        # Phase enable flags (toggled via SetBool services or ~start_enabled)
        start_enabled = bool(rospy.get_param("~start_enabled", False))
        self.enable_approach = start_enabled
        self.enable_contact = start_enabled

        # Default offsets — overwritten immediately by dynamic reconfigure
        self.approach_offset = 0.75
        self.contact_offset = 0.625

        # Dynamic reconfigure
        self.reconfigure_server = Server(
            ValveTrajectoryConfig, self.reconfigure_callback
        )

        # Enable/disable services
        self.set_enable_approach_service = rospy.Service(
            "set_transform_valve_approach_frame",
            SetBool,
            self.handle_enable_approach_service,
        )
        self.set_enable_contact_service = rospy.Service(
            "set_transform_valve_contact_frame",
            SetBool,
            self.handle_enable_contact_service,
        )

        rospy.loginfo(
            "Valve trajectory publisher started "
            f"(valve_frame={self.valve_frame}, start_enabled={start_enabled})"
        )

    # =========================================================================
    #  Helpers
    # =========================================================================
    def get_pose(self, transform: TransformStamped) -> Pose:
        """TransformStamped → Pose."""
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        """Pose → TransformStamped, parented to odom."""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def get_valve_surface_normal_3d(self, valve_tf):
        """Extract the full 3D valve surface normal and compute an orientation
        quaternion that faces INTO the valve.

        The keypoint pipeline sets `tac/valve`'s X-axis to the face normal
        pointing OUT of the valve face (toward the camera). We return that
        unit normal plus a quaternion whose X-axis points INTO the valve
        (opposite of the outward normal), giving the robot the correct
        roll, pitch, and yaw to face the valve.

        Returns (normal_3d, orientation_quat) or None if the normal is
        degenerate.
        """
        q = [
            valve_tf.transform.rotation.x,
            valve_tf.transform.rotation.y,
            valve_tf.transform.rotation.z,
            valve_tf.transform.rotation.w,
        ]

        # Surface normal = X-axis of the valve frame, in odom coordinates.
        rot_matrix = quaternion_matrix(q)
        normal_3d = rot_matrix[:3, 0].copy()

        norm = np.linalg.norm(normal_3d)
        if norm < 1e-6:
            rospy.logwarn_throttle(5.0, "Valve surface normal is degenerate!")
            return None

        normal_3d = normal_3d / norm

        # The robot's X-axis should point INTO the valve (opposite of
        # the outward normal).
        facing_dir = -normal_3d

        # Build a rotation matrix whose X-axis = facing_dir.
        # Choose an "up" hint to derive Y and Z axes.
        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(facing_dir, up_hint)) > 0.99:
            # facing_dir is nearly vertical — fall back to Y-up hint.
            up_hint = np.array([0.0, 1.0, 0.0])

        y_axis = np.cross(up_hint, facing_dir)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(facing_dir, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # 4x4 rotation matrix for tf.transformations
        rot = np.eye(4)
        rot[:3, 0] = facing_dir
        rot[:3, 1] = y_axis
        rot[:3, 2] = z_axis

        orientation_quat = tf.transformations.quaternion_from_matrix(rot)

        return normal_3d, orientation_quat

    def send_transform(self, transform):
        """Send a transform via the set_object_transform service."""
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def _create_offset_frame(self, child_frame_id: str, offset_distance: float):
        """Build a frame at `offset_distance` along the valve's 3D surface
        normal, oriented to face into the valve. Publishes via the
        set_object_transform service.
        """
        try:
            valve_tf = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.valve_frame,
                rospy.Time(0),
                rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Valve TF lookup failed: {e}")
            return

        valve_pose = self.get_pose(valve_tf)
        valve_pos = np.array(
            [valve_pose.position.x, valve_pose.position.y, valve_pose.position.z]
        )

        result = self.get_valve_surface_normal_3d(valve_tf)
        if result is None:
            return
        normal_3d, orientation_quat = result

        # Step out along the full 3D surface normal.
        target_pos = valve_pos + (normal_3d * offset_distance)

        pose = Pose()
        pose.position.x = target_pos[0]
        pose.position.y = target_pos[1]
        pose.position.z = target_pos[2]

        pose.orientation.x = orientation_quat[0]
        pose.orientation.y = orientation_quat[1]
        pose.orientation.z = orientation_quat[2]
        pose.orientation.w = orientation_quat[3]

        transform = self.build_transform_message(child_frame_id, pose)
        self.send_transform(transform)

    # =========================================================================
    #  Phase publishers
    # =========================================================================
    def create_approach_frame(self):
        self._create_offset_frame(self.approach_frame, self.approach_offset)

    def create_contact_frame(self):
        self._create_offset_frame(self.contact_frame, self.contact_offset)

    # =========================================================================
    #  Service handlers
    # =========================================================================
    def handle_enable_approach_service(self, req):
        self.enable_approach = req.data
        message = f"Valve approach frame publish is set to: {self.enable_approach}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_enable_contact_service(self, req):
        self.enable_contact = req.data
        message = f"Valve contact frame publish is set to: {self.enable_contact}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    # =========================================================================
    #  Dynamic reconfigure
    # =========================================================================
    def reconfigure_callback(self, config, level):
        self.approach_offset = config.approach_offset
        self.contact_offset = config.contact_offset
        rospy.loginfo(
            "Valve trajectory params updated: "
            f"approach={self.approach_offset:.2f}m, "
            f"contact={self.contact_offset:.2f}m"
        )
        return config

    # =========================================================================
    #  Main loop
    # =========================================================================
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.enable_approach:
                self.create_approach_frame()
            if self.enable_contact:
                self.create_contact_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ValveTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
