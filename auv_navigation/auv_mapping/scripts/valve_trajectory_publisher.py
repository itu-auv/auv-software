#!/usr/bin/env python3
"""Single node publishing approach/engage target frames for both front and
bottom valves.

For each valve, target frames are offset along the valve's normal direction
(full 3-D, so the approach standoff clears the structure the valve sits in).

Orientation is intentionally **level** — ``roll = pitch = 0`` — because the
vehicle cannot hold pitch in the sea (no hydrostatics / pitch PID).  Only the
yaw is commanded, chosen per valve:

* **Front valve**: yaw faces the valve — the horizontal projection of the
  valve normal.  Robust because the normal comes from the planar fit of the
  whole bolt ring; it is invariant to the keypoint-ordering ambiguity (which
  only rotates the frame *about* the normal).
* **Bottom valve**: yaw = ``mission_start_link`` heading — a fixed,
  operator-chosen heading set by pre-positioning within 0.5 m of the valve.
  The bottom valve normal is ~vertical, so its horizontal projection is
  degenerate; and the PnP rotation-about-normal is unreliable.  The bottom
  gripper points straight down (-Z) and the jaws rotate independently to meet
  the handle, so the body only needs to hold this safe heading.

Two target frames per valve:
  - approach: standoff distance from valve
  - engage:   at the valve surface

Each valve has an independent ``SetBool`` service to toggle its frame
publishing (needed for the freeze/unfreeze pattern in the SMACH layer).
"""

import numpy as np
import rospy
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_from_euler, quaternion_matrix

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from auv_mapping.cfg import ValveTrajectoryConfig


# Below this horizontal magnitude a vector is too vertical to extract a
# meaningful yaw from — fall back to the mission heading.
_HORIZONTAL_EPS = 0.1


def _horizontal_yaw(vec):
    """Yaw (rad) of a 3-D vector's horizontal projection, or None when the
    vector is too close to vertical to define a heading."""
    if np.hypot(vec[0], vec[1]) < _HORIZONTAL_EPS:
        return None
    return float(np.arctan2(vec[1], vec[0]))


class _ValveConfig:
    """Per-valve runtime state: frame names, offsets, and enable service."""

    def __init__(
        self,
        valve_frame,
        approach_frame,
        engage_frame,
        approach_offset,
        engage_offset,
        service_name,
        yaw_from_valve_normal,
    ):
        self.valve_frame = valve_frame
        self.approach_frame = approach_frame
        self.engage_frame = engage_frame
        self.approach_offset = approach_offset
        self.engage_offset = engage_offset
        # True  -> face the valve (front): yaw from horizontal valve normal.
        # False -> hold mission heading (bottom): yaw from mission_start_link.
        self.yaw_from_valve_normal = yaw_from_valve_normal
        self.enabled = False

        self._service = rospy.Service(service_name, SetBool, self._handle_enable)

    def _handle_enable(self, req):
        self.enabled = req.data
        msg = f"Valve ({self.valve_frame}) publishing set to: {self.enabled}"
        rospy.loginfo(msg)
        return SetBoolResponse(success=True, message=msg)


class ValveTrajectoryPublisherNode:
    def __init__(self):
        rospy.init_node("valve_trajectory_publisher_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.mission_start_frame = rospy.get_param(
            "~mission_start_frame", "mission_start_link"
        )

        # ---- Front valve ----
        # Faces the valve: yaw from the (reliable) horizontal valve normal.
        self.front = _ValveConfig(
            valve_frame=rospy.get_param("~front_valve_frame", "tac/valve_front"),
            approach_frame=rospy.get_param(
                "~front_approach_frame", "valve_front_approach_target"
            ),
            engage_frame=rospy.get_param(
                "~front_engage_frame", "valve_front_engage_target"
            ),
            approach_offset=rospy.get_param("~front_approach_offset", 0.3),
            engage_offset=rospy.get_param("~front_engage_offset", 0.0),
            service_name=rospy.get_param(
                "~front_service", "set_publishing_valve_front"
            ),
            yaw_from_valve_normal=True,
        )

        # ---- Bottom valve ----
        # Holds the operator-chosen mission heading: the valve normal is
        # ~vertical (no usable horizontal projection) and the PnP yaw is
        # unreliable, so the body keeps a known-safe heading while the
        # downward gripper engages and its jaws rotate independently.
        self.bottom = _ValveConfig(
            valve_frame=rospy.get_param("~bottom_valve_frame", "tac/valve_bottom"),
            approach_frame=rospy.get_param(
                "~bottom_approach_frame", "valve_bottom_approach_target"
            ),
            engage_frame=rospy.get_param(
                "~bottom_engage_frame", "valve_bottom_engage_target"
            ),
            approach_offset=rospy.get_param("~bottom_approach_offset", 0.3),
            engage_offset=rospy.get_param("~bottom_engage_offset", 0.0),
            service_name=rospy.get_param(
                "~bottom_service", "set_publishing_valve_bottom"
            ),
            yaw_from_valve_normal=False,
        )

        self.reconfigure_server = Server(
            ValveTrajectoryConfig, self.reconfigure_callback
        )

    # ------------------------------------------------------------------ #
    #  Dynamic reconfigure
    # ------------------------------------------------------------------ #
    def reconfigure_callback(self, config, level):
        self.front.approach_offset = config.front_approach_offset
        self.front.engage_offset = config.front_engage_offset
        self.bottom.approach_offset = config.bottom_approach_offset
        self.bottom.engage_offset = config.bottom_engage_offset
        return config

    # ------------------------------------------------------------------ #
    #  TF helpers
    # ------------------------------------------------------------------ #
    def _lookup_tf(self, frame_id, label):
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                frame_id,
                rospy.Time(0),
                rospy.Duration(4.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"{label} TF lookup failed: {e}")
            return None

    def _send_transform(self, child_frame_id, position, quaternion):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.x = float(quaternion[0])
        t.transform.rotation.y = float(quaternion[1])
        t.transform.rotation.z = float(quaternion[2])
        t.transform.rotation.w = float(quaternion[3])

        req = SetObjectTransformRequest(transform=t)
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {child_frame_id}: " f"{resp.message}"
            )

    # ------------------------------------------------------------------ #
    #  Per-valve computation & publishing
    # ------------------------------------------------------------------ #
    def _publish_valve(self, valve_cfg, mission_yaw):
        valve_tf = self._lookup_tf(valve_cfg.valve_frame, valve_cfg.valve_frame)
        if valve_tf is None:
            return

        valve_pos = np.array(
            [
                valve_tf.transform.translation.x,
                valve_tf.transform.translation.y,
                valve_tf.transform.translation.z,
            ]
        )
        q_valve = [
            valve_tf.transform.rotation.x,
            valve_tf.transform.rotation.y,
            valve_tf.transform.rotation.z,
            valve_tf.transform.rotation.w,
        ]
        R_valve = quaternion_matrix(q_valve)[:3, :3]
        valve_normal = R_valve[:, 0]  # +X axis of valve frame

        # Pick the level (roll=pitch=0) heading for this valve.
        if valve_cfg.yaw_from_valve_normal:
            # Face the valve: heading along the horizontal valve normal.
            yaw = _horizontal_yaw(-valve_normal)
            if yaw is None:
                rospy.logwarn_throttle(
                    2.0,
                    f"[{valve_cfg.valve_frame}] valve normal too vertical for a "
                    f"facing yaw; falling back to mission heading",
                )
                yaw = mission_yaw
        else:
            # Hold the operator-chosen mission heading.
            yaw = mission_yaw

        q_target = quaternion_from_euler(0.0, 0.0, yaw)

        rospy.loginfo_throttle(
            2.0,
            f"[{valve_cfg.valve_frame}] pos={np.round(valve_pos, 3)}, "
            f"normal={np.round(valve_normal, 3)}, yaw={yaw:.3f}",
        )

        for target_frame, offset in [
            (valve_cfg.approach_frame, valve_cfg.approach_offset),
            (valve_cfg.engage_frame, valve_cfg.engage_offset),
        ]:
            # Position offset stays along the full 3-D valve normal so the
            # standoff clears the structure even though the frame is level.
            target_pos = valve_pos + valve_normal * offset

            rospy.loginfo_throttle(
                2.0,
                f"[{target_frame}] pos={np.round(target_pos, 3)}, " f"offset={offset}",
            )
            self._send_transform(target_frame, target_pos, q_target)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not (self.front.enabled or self.bottom.enabled):
                rate.sleep()
                continue

            # Lookup mission_start_link once per cycle for the bottom-valve
            # (and fallback) heading.
            ms_tf = self._lookup_tf(self.mission_start_frame, "mission_start_link")
            if ms_tf is None:
                rate.sleep()
                continue

            q_ms = [
                ms_tf.transform.rotation.x,
                ms_tf.transform.rotation.y,
                ms_tf.transform.rotation.z,
                ms_tf.transform.rotation.w,
            ]
            mission_start_R = quaternion_matrix(q_ms)[:3, :3]
            # Heading = yaw of mission_start X axis in the horizontal plane.
            mission_yaw = float(
                np.arctan2(mission_start_R[1, 0], mission_start_R[0, 0])
            )

            rospy.loginfo_throttle(2.0, f"[mission_start] yaw={mission_yaw:.3f}")

            if self.front.enabled:
                self._publish_valve(self.front, mission_yaw)
            if self.bottom.enabled:
                self._publish_valve(self.bottom, mission_yaw)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = ValveTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
