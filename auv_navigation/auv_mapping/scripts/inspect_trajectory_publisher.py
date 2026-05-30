#!/usr/bin/env python3
"""Publishes 8 inspection frames around the valve, each yawed toward the
rectangle centre so the robot faces inward when aligned. The rectangle
centre and the frames' XY positions use the valve's full orientation, but
every frame is flattened to a single Z that sits a configurable distance
above the valve's Z. A SetBool service toggles publishing of all frames.
"""

import threading

import numpy as np
import rospy
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_from_euler, quaternion_matrix

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from auv_mapping.cfg import InspectTrajectoryConfig


# Geometry from tac_valve/model.sdf. Desk mesh AABB is X: -0.640..+0.825,
# Y: ±1.242, so desk origin sits +0.0925 m in X from the table-top centre.
# valve_front is at (0.580, 0.555, 1.4205) in desk with RPY (0, pi, 0); the
# Ry(pi) rotation maps valve X = desk -X, valve Y = desk +Y, valve Z = desk -Z.
RECT_CENTER_IN_DESK = np.array([0.0925, 0.0, 1.4205])
VALVE_IN_DESK = np.array([0.580, -0.555, 1.4205])

BASE_LONG_SIDE = 2.485
BASE_SHORT_SIDE = 1.464

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _desk_offset_to_valve_local(desk_vec):
    """Inverse of Ry(pi): (x,y,z) -> (-x, y, -z)."""
    return np.array([-desk_vec[0], desk_vec[1], -desk_vec[2]])


def _build_valve_local_offsets(long_side_extension: float):
    """Long side grows by `long_side_extension` m; short scales proportionally."""
    long_side = BASE_LONG_SIDE + long_side_extension
    scale = long_side / BASE_LONG_SIDE
    short_side = BASE_SHORT_SIDE * scale
    long_half = long_side / 2.0
    short_half = short_side / 2.0

    # Offsets from rectangle centre in desk frame (short=X, long=Y).
    # Order: mid -X, BL, mid -Y, BR, mid +X, TR, mid +Y, TL.
    offsets_in_desk = [
        (-short_half, 0.0, 0.0),
        (-short_half, -long_half, 0.0),
        (0.0, -long_half, 0.0),
        (+short_half, -long_half, 0.0),
        (+short_half, 0.0, 0.0),
        (+short_half, +long_half, 0.0),
        (0.0, +long_half, 0.0),
        (-short_half, +long_half, 0.0),
    ]
    return [
        _desk_offset_to_valve_local(
            (RECT_CENTER_IN_DESK + np.array(delta)) - VALVE_IN_DESK
        )
        for delta in offsets_in_desk
    ]


class InspectTrajectoryPublisherNode:
    def __init__(self):
        rospy.init_node("inspect_trajectory_publisher_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        # Inspect mission is front-only: the rectangle geometry below
        # (RECT_CENTER_IN_DESK, VALVE_IN_DESK) is hard-coded to the front
        # panel's layout. valve_bottom would need a different rectangle.
        self.valve_frame = rospy.get_param("~valve_frame", "tac/valve_front")

        self._offsets_lock = threading.Lock()
        self.frame_offsets = []
        self.height_above_valve = 1.0
        self.center_offset_valve_local = _desk_offset_to_valve_local(
            RECT_CENTER_IN_DESK - VALVE_IN_DESK
        )

        self.enable_publishing = False
        self.set_enable_service = rospy.Service(
            "set_transform_inspect_frames",
            SetBool,
            self.handle_enable_service,
        )

        self.reconfigure_server = Server(
            InspectTrajectoryConfig, self.reconfigure_callback
        )

    def reconfigure_callback(self, config, level):
        offsets = _build_valve_local_offsets(config.long_side_extension)
        with self._offsets_lock:
            self.frame_offsets = [
                (f"inspect_frame_{i}", off) for i, off in enumerate(offsets)
            ]
            self.height_above_valve = config.height_above_valve
        return config

    def _lookup_in_odom(self, frame_id, label):
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame, frame_id, rospy.Time.now(), rospy.Duration(4.0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"{label} TF lookup failed: {e}")
            return None

    @staticmethod
    def _translation(tf_stamped):
        t = tf_stamped.transform.translation
        return np.array([t.x, t.y, t.z])

    @staticmethod
    def _rotation(tf_stamped):
        r = tf_stamped.transform.rotation
        return quaternion_matrix([r.x, r.y, r.z, r.w])[:3, :3]

    def _broadcast(self, child_frame_id, xyz, quat=IDENTITY_QUAT):
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = child_frame_id
        msg.transform.translation.x = float(xyz[0])
        msg.transform.translation.y = float(xyz[1])
        msg.transform.translation.z = float(xyz[2])
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])

        req = SetObjectTransformRequest(transform=msg)
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logerr(
                    f"Failed to set transform for {child_frame_id}: {resp.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def publish_inspect_frames(self):
        valve_tf = self._lookup_in_odom(self.valve_frame, "Valve")
        if valve_tf is None:
            return

        R = self._rotation(valve_tf)
        valve_pos = self._translation(valve_tf)
        center_pos = valve_pos + R @ self.center_offset_valve_local

        with self._offsets_lock:
            frame_offsets = list(self.frame_offsets)
            height_above_valve = self.height_above_valve

        if not frame_offsets:
            return

        # All frames share one Z, a configurable distance above the valve's Z.
        # Only the XY of each frame (and the centre for the inward yaw) uses
        # the valve's full orientation.
        frame_z = valve_pos[2] + height_above_valve

        for child_frame_id, offset_valve_local in frame_offsets:
            target_pos = valve_pos + R @ np.asarray(offset_valve_local)
            target_pos[2] = frame_z
            yaw = float(
                np.arctan2(
                    center_pos[1] - target_pos[1],
                    center_pos[0] - target_pos[0],
                )
            )
            self._broadcast(
                child_frame_id,
                target_pos,
                quaternion_from_euler(0.0, 0.0, yaw),
            )

    def handle_enable_service(self, req):
        self.enable_publishing = req.data
        message = f"Inspect frames publishing is set to: {self.enable_publishing}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.enable_publishing:
                self.publish_inspect_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = InspectTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
