#!/usr/bin/env python3
"""Publishes 8 inspection frames around the valve, each yawed toward the
rectangle centre so the robot faces inward when aligned. Two companion
frames (identity orientation) are published above mission_start_link and
above inspect_frame_0, sharing one vertical offset param. A SetBool
service toggles publishing of all frames.
"""

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_from_euler, quaternion_matrix

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


# Geometry from tac_valve/model.sdf. Desk mesh AABB is X: -0.640..+0.825,
# Y: ±1.242, so desk origin sits +0.0925 m in X from the table-top centre.
# valve_front is at (0.580, 0.555, 1.4205) in desk with RPY (0, pi, 0); the
# Ry(pi) rotation maps valve X = desk -X, valve Y = desk +Y, valve Z = desk -Z.
RECT_CENTER_IN_DESK = np.array([0.0925, 0.0, 1.4205])
VALVE_IN_DESK = np.array([0.580, 0.555, 1.4205])

# Rectangle grown by +1 m on the long side, scaled proportionally on the short.
LONG_SIDE = 2.485 + 1.0
SCALE = LONG_SIDE / 2.485
SHORT_SIDE = 1.464 * SCALE
LONG_HALF = LONG_SIDE / 2.0
SHORT_HALF = SHORT_SIDE / 2.0

# Offsets from rectangle centre in desk frame (short=X, long=Y).
# Order: mid -X, BL, mid -Y, BR, mid +X, TR, mid +Y, TL.
_FRAME_OFFSETS_FROM_CENTER_DESK = [
    (-SHORT_HALF, 0.0, 0.0),
    (-SHORT_HALF, -LONG_HALF, 0.0),
    (0.0, -LONG_HALF, 0.0),
    (+SHORT_HALF, -LONG_HALF, 0.0),
    (+SHORT_HALF, 0.0, 0.0),
    (+SHORT_HALF, +LONG_HALF, 0.0),
    (0.0, +LONG_HALF, 0.0),
    (-SHORT_HALF, +LONG_HALF, 0.0),
]

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _desk_offset_to_valve_local(desk_vec):
    """Inverse of Ry(pi): (x,y,z) -> (-x, y, -z)."""
    return np.array([-desk_vec[0], desk_vec[1], -desk_vec[2]])


def _build_valve_local_offsets():
    return [
        _desk_offset_to_valve_local(
            (RECT_CENTER_IN_DESK + np.array(delta)) - VALVE_IN_DESK
        )
        for delta in _FRAME_OFFSETS_FROM_CENTER_DESK
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
        self.valve_frame = rospy.get_param("~valve_frame", "tac/valve")

        self.mission_start_frame = rospy.get_param(
            "~mission_start_frame", "mission_start_link"
        )
        self.mission_start_top_frame = rospy.get_param(
            "~mission_start_top_frame", "inspect_start_top"
        )
        self.frame_0_top_frame = rospy.get_param(
            "~frame_0_top_frame", "inspect_frame_0_top"
        )
        self.mission_start_top_offset = rospy.get_param(
            "~mission_start_top_offset", 1.0
        )

        self.frame_offsets = [
            (f"inspect_frame_{i}", off)
            for i, off in enumerate(_build_valve_local_offsets())
        ]
        self.center_offset_valve_local = _desk_offset_to_valve_local(
            RECT_CENTER_IN_DESK - VALVE_IN_DESK
        )

        self.enable_publishing = False
        self.set_enable_service = rospy.Service(
            "set_transform_inspect_frames",
            SetBool,
            self.handle_enable_service,
        )

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

        for child_frame_id, offset_valve_local in self.frame_offsets:
            target_pos = valve_pos + R @ np.asarray(offset_valve_local)
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

        frame_0_pos = valve_pos + R @ np.asarray(self.frame_offsets[0][1])
        self._broadcast(
            self.frame_0_top_frame,
            frame_0_pos + np.array([0.0, 0.0, self.mission_start_top_offset]),
        )

    def publish_mission_start_top_frame(self):
        ms_tf = self._lookup_in_odom(self.mission_start_frame, "Mission start")
        if ms_tf is None:
            return
        pos = self._translation(ms_tf) + np.array(
            [0.0, 0.0, self.mission_start_top_offset]
        )
        self._broadcast(self.mission_start_top_frame, pos)

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
                self.publish_mission_start_top_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = InspectTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
