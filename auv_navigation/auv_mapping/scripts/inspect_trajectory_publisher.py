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
# valve_front is at (0.580, 0.555, 1.4205) in the desk frame with RPY
# (0, pi, 0). That Ry(pi) is the *orientation* of the valve TF frame relative
# to the desk; _desk_offset_to_valve_local() below uses it to convert
# desk-frame offsets into the valve's local frame. The valve POSITION is kept
# in plain desk coordinates (it is subtracted from other desk-frame points
# *before* the rotation is applied), so it must NOT be sign-flipped.
RECT_CENTER_IN_DESK = np.array([0.0925, 0.0, 1.4205])

# The valve is mounted on the front (+X) face of the desk at one of two
# lateral positions that mirror across the desk's long (Y) axis:
#   left  -> (0.580, +0.555, 1.4205)
#   right -> (0.580, -0.555, 1.4205)
# Only the desk-frame Y sign differs between the two layouts; X (distance along
# the short axis) and Z (height) are identical. The smach layer selects the
# side by calling the matching left/right enable service (see
# handle_enable_service).
VALVE_IN_DESK_X = 0.580
VALVE_IN_DESK_ABS_Y = 0.555
VALVE_IN_DESK_Z = 1.4205

BASE_LONG_SIDE = 2.485
BASE_SHORT_SIDE = 1.464

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _valve_in_desk(valve_side: str):
    """Valve position in the desk frame for the given side ('left'/'right').

    Only the desk-frame Y sign differs between the two layouts: the valve sits
    on the front face at +0.555 (left) or -0.555 (right) along the desk's long
    axis. X (short-axis offset) and Z (height) are identical for both.
    """
    y = VALVE_IN_DESK_ABS_Y if valve_side == "left" else -VALVE_IN_DESK_ABS_Y
    return np.array([VALVE_IN_DESK_X, y, VALVE_IN_DESK_Z])


def _desk_offset_to_valve_local(desk_vec):
    """Inverse of Ry(pi): (x,y,z) -> (-x, y, -z)."""
    return np.array([-desk_vec[0], desk_vec[1], -desk_vec[2]])


def _build_valve_local_offsets(long_side_extension: float, valve_in_desk):
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
            (RECT_CENTER_IN_DESK + np.array(delta)) - valve_in_desk
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
        # Geometry is side-dependent and stays unbuilt until smach selects a
        # side via one of the left/right enable services. valve_side stays
        # None (and frame_offsets empty) until then, so nothing is published.
        self.valve_side = None
        self.valve_in_desk = None
        self.center_offset_valve_local = None
        self.frame_offsets = []
        self.long_side_extension = 0.0
        self.height_above_valve = 1.0

        self.enable_publishing = False
        # Two enable services; the one smach calls selects the valve side.
        self.set_enable_left_service = rospy.Service(
            "set_transform_inspect_frames_left",
            SetBool,
            lambda req: self.handle_enable_service("left", req),
        )
        self.set_enable_right_service = rospy.Service(
            "set_transform_inspect_frames_right",
            SetBool,
            lambda req: self.handle_enable_service("right", req),
        )

        self.reconfigure_server = Server(
            InspectTrajectoryConfig, self.reconfigure_callback
        )

    def reconfigure_callback(self, config, level):
        with self._offsets_lock:
            self.long_side_extension = config.long_side_extension
            self.height_above_valve = config.height_above_valve
            # Rebuild only once a side is known; before that, stay empty.
            if self.valve_in_desk is not None:
                self._rebuild_offsets_locked()
        return config

    def _rebuild_offsets_locked(self):
        """Recompute the 8 frame offsets. Caller must hold _offsets_lock."""
        offsets = _build_valve_local_offsets(
            self.long_side_extension, self.valve_in_desk
        )
        self.frame_offsets = [
            (f"inspect_frame_{i}", off) for i, off in enumerate(offsets)
        ]

    def _apply_valve_side(self, valve_side):
        """Set the valve side and (re)build all side-dependent geometry."""
        with self._offsets_lock:
            self.valve_side = valve_side
            self.valve_in_desk = _valve_in_desk(valve_side)
            self.center_offset_valve_local = _desk_offset_to_valve_local(
                RECT_CENTER_IN_DESK - self.valve_in_desk
            )
            self._rebuild_offsets_locked()

    def _lookup_in_odom(self, frame_id, label):
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame, frame_id, rospy.Time(0), rospy.Duration(4.0)
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

        with self._offsets_lock:
            frame_offsets = list(self.frame_offsets)
            height_above_valve = self.height_above_valve
            center_offset_valve_local = self.center_offset_valve_local

        if not frame_offsets or center_offset_valve_local is None:
            return

        center_pos = valve_pos + R @ center_offset_valve_local

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

    def handle_enable_service(self, valve_side, req):
        if req.data:
            # Selecting the side rebuilds geometry mirrored for left/right.
            self._apply_valve_side(valve_side)
            self.enable_publishing = True
            message = f"Inspect frames publishing enabled (valve on {valve_side})"
        else:
            self.enable_publishing = False
            message = "Inspect frames publishing disabled"
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
