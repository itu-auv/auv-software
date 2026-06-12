#!/usr/bin/env python3
"""Single node publishing approach/engage target frames for both front and
bottom valves.

For each valve, target frames are offset along the valve's normal direction.
Orientations are built directly in yaw-locked ZYX form
``R_target = Rz(psi) * Ry(beta) * Rx(gamma)`` (the ``tf`` ``'sxyz'`` Euler
convention), so the commanded yaw is *exactly* the mission heading at any
tilt -- no look-at construction, no body-frame correction quaternion.

* **Front valve**: body +X axis points at the valve (``f = -n``), roll = 0.
  Yaw faces the valve (horizontal projection of ``f``), falling back to the
  mission heading when ``f`` is too vertical.
* **Bottom valve**: body -Z axis points along the valve normal (``R . z = n``),
  yaw locked to ``mission_start_link`` heading. Pitch/roll absorb the table
  tilt.

A ``~tilt_clamp_deg`` param (default 45 deg) bounds how far the valve normal
may tilt from nominal before the orientation is clamped (azimuth preserved);
beyond 90 deg (bottom valve only, normal pointing down) the frame is rejected
for that cycle and the last published frames are held.

See ``auv_navigation/auv_mapping/docs/valve_pitched_trajectory_plan.md`` for
the full derivation and ``valve_frame_visualizer.html`` for an interactive
validator.

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
# meaningful yaw from -- fall back to the mission heading. Same constant as
# the active (level) publisher.
_HORIZONTAL_EPS = 0.1

_DEFAULT_TILT_CLAMP_DEG = 60.0


def _bottom_target(n, psi):
    """Yaw-locked frame for the bottom valve.

    Requirement: body -Z onto -n, i.e. ``R . z_hat == n``, with yaw = psi
    exactly (by construction of the ZYX order)::

        n' = Rz(-psi) . n
        gamma = -asin(n'_y)   # roll absorbs sideways tilt
        beta  =  atan2(n'_x, n'_z)  # pitch absorbs fore/aft tilt
    """
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    n_prime = np.array(
        [
            cos_psi * n[0] + sin_psi * n[1],
            -sin_psi * n[0] + cos_psi * n[1],
            n[2],
        ]
    )
    gamma = -np.arcsin(np.clip(n_prime[1], -1.0, 1.0))
    beta = np.arctan2(n_prime[0], n_prime[2])
    quat = quaternion_from_euler(gamma, beta, psi)
    return quat, beta, gamma


def _front_target(n, mission_yaw):
    """Yaw-locked frame for the front valve.

    Requirement: body +X onto f = -n (face the valve), roll = 0 -- the
    roll channel about the flange normal is the handle position and is
    never used::

        h = hypot(f_x, f_y)
        psi = atan2(f_y, f_x)   if h >= _HORIZONTAL_EPS, else mission yaw
        beta = atan2(-f_z, h)
    """
    f = -n
    h = np.hypot(f[0], f[1])
    if h < _HORIZONTAL_EPS:
        psi = mission_yaw
    else:
        psi = np.arctan2(f[1], f[0])
    beta = np.arctan2(-f[2], h)
    quat = quaternion_from_euler(0.0, beta, psi)
    return quat, beta, psi


def _clamp_bottom_normal(n, clamp_rad):
    """Tilt-clamp policy for the bottom-valve normal (nominal = +Z).

    Returns ``(n_used_or_None, tilt, status)`` where ``status`` is one of
    ``"OK"``, ``"CLAMPED"`` (azimuth preserved, tilt clamped) or
    ``"REJECT"`` (normal points down -- flipped, do not publish).
    """
    tilt = np.arccos(np.clip(n[2], -1.0, 1.0))
    if tilt > np.pi / 2:
        return None, tilt, "REJECT"
    if tilt <= clamp_rad:
        return n, tilt, "OK"
    az = np.arctan2(n[1], n[0])
    n_clamped = np.array(
        [
            np.sin(clamp_rad) * np.cos(az),
            np.sin(clamp_rad) * np.sin(az),
            np.cos(clamp_rad),
        ]
    )
    return n_clamped, tilt, "CLAMPED"


def _clamp_front_normal(n, clamp_rad):
    """Tilt-clamp policy for the front-valve normal elevation (nominal =
    horizontal). Elevation is bounded to +/-90 deg, so there is no reject
    case here -- only ``"OK"`` or ``"CLAMPED"``."""
    elevation = np.arcsin(np.clip(n[2], -1.0, 1.0))
    if abs(elevation) <= clamp_rad:
        return n, elevation, "OK"
    az = np.arctan2(n[1], n[0])
    e = clamp_rad * np.sign(elevation)
    n_clamped = np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])
    return n_clamped, elevation, "CLAMPED"


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
        is_bottom,
    ):
        self.valve_frame = valve_frame
        self.approach_frame = approach_frame
        self.engage_frame = engage_frame
        self.approach_offset = approach_offset
        self.engage_offset = engage_offset
        self.is_bottom = is_bottom
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
        self.tilt_clamp_rad = np.radians(
            rospy.get_param("~tilt_clamp_deg", _DEFAULT_TILT_CLAMP_DEG)
        )

        # ---- Front valve ----
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
            is_bottom=False,
        )

        # ---- Bottom valve ----
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
            is_bottom=True,
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

        if valve_cfg.is_bottom:
            n_used, tilt, status = _clamp_bottom_normal(
                valve_normal, self.tilt_clamp_rad
            )
            if status == "REJECT":
                rospy.logwarn_throttle(
                    2.0,
                    f"[{valve_cfg.valve_frame}] normal tilt "
                    f"{np.degrees(tilt):.1f} deg exceeds 90 deg (flipped "
                    f"normal) - rejecting, holding last published frames",
                )
                return
            q_target, beta, gamma = _bottom_target(n_used, mission_yaw)
            psi = mission_yaw
        else:
            n_used, tilt, status = _clamp_front_normal(
                valve_normal, self.tilt_clamp_rad
            )
            q_target, beta, psi = _front_target(n_used, mission_yaw)
            gamma = 0.0

        if status == "CLAMPED":
            rospy.logwarn_throttle(
                2.0,
                f"[{valve_cfg.valve_frame}] normal tilt {np.degrees(tilt):.1f} "
                f"deg exceeds clamp {np.degrees(self.tilt_clamp_rad):.1f} deg - "
                f"clamped (azimuth preserved)",
            )

        rospy.loginfo_throttle(
            2.0,
            f"[{valve_cfg.valve_frame}] pos={np.round(valve_pos, 3)}, "
            f"normal={np.round(valve_normal, 3)}, tilt={np.degrees(tilt):.1f} deg, "
            f"status={status}, psi={np.degrees(psi):.1f} deg, "
            f"beta={np.degrees(beta):.1f} deg, gamma={np.degrees(gamma):.1f} deg",
        )

        for target_frame, offset in [
            (valve_cfg.approach_frame, valve_cfg.approach_offset),
            (valve_cfg.engage_frame, valve_cfg.engage_offset),
        ]:
            # Position offset stays along the full 3-D (unclamped) normal so
            # the standoff clears the structure even when the orientation
            # has been tilt-clamped.
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

            # Lookup mission_start_link once per cycle for the mission
            # heading (bottom-valve yaw, and front-valve fallback yaw).
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
            # This is the ZYX yaw of the snapshot by construction, so it is
            # exact even if the vehicle was rocking when it was taken.
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
