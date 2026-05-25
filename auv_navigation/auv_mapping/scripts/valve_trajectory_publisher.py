#!/usr/bin/env python3
"""Single node publishing approach/engage target frames for both front and
bottom valves.

For each valve, target frames are offset along the valve's normal direction.
Orientations use a gimbal-lock-free "look-at" algorithm: the frame's X axis
points at the valve, free rotation resolved via ``mission_start_link``.

An optional body-frame correction quaternion is applied after look-at to
re-orient the frame to match the AUV's gripper convention:

* **Front valve** (no correction): X already faces the valve — matches the
  front gripper which points along base_link X.
* **Bottom valve** (Ry +90 deg correction): rotates the look-at frame so
  that -Z faces the valve and X points forward — matches the bottom gripper
  which hangs below base_link along -Z.  Correctly handles table tilt.

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
from tf.transformations import (
    quaternion_from_euler,
    quaternion_from_matrix,
    quaternion_matrix,
    quaternion_multiply,
)

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from auv_mapping.cfg import ValveTrajectoryConfig


_PARALLEL_THRESHOLD = 0.1  # sin(~6 deg) — below this, axis is too parallel

# Ry(-pi/2): maps look-at X→-Z, Z→X so that +Z faces up and -Z faces
# the valve — matching base_link convention for a bottom gripper.
_BOTTOM_CORRECTION_QUAT = quaternion_from_euler(0.0, -np.pi / 2, 0.0)
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _look_at_quaternion(forward, ref_rotation_matrix, hint_axis=None, label=""):
    """Quaternion whose X axis is *forward*, free rotation resolved by
    ``ref_rotation_matrix``.

    Algorithm (gimbal-lock-free — no Euler angles):
      1. Pick a Z-hint from ``ref_rotation_matrix``.  If ``hint_axis`` is
         given (0=X, 1=Y, 2=Z) use that axis directly — the caller knows
         which mission-start axis carries the information it cares about.
         Otherwise fall back to priority order Z -> X -> Y.
      2. Y = normalise(Z_hint x forward)   — perpendicular to both.
      3. Z = forward x Y                   — completes right-handed frame.
      4. Build the 3x3 rotation matrix [X Y Z] and convert to quaternion.
    """
    forward = forward / np.linalg.norm(forward)

    cross_mags = [
        np.linalg.norm(np.cross(forward, ref_rotation_matrix[:, i])) for i in range(3)
    ]

    if hint_axis is not None:
        chosen = hint_axis
    else:
        # Priority order: Z (level roll), X (heading yaw), Y (last resort).
        priority = [2, 0, 1]
        chosen = priority[0]
        for axis_idx in priority:
            if cross_mags[axis_idx] > _PARALLEL_THRESHOLD:
                chosen = axis_idx
                break

    z_hint = ref_rotation_matrix[:, chosen]

    rospy.loginfo_throttle(
        2.0,
        f"[look_at {label}] forward={np.round(forward, 3)}, "
        f"cross_mags=[{cross_mags[0]:.3f}, {cross_mags[1]:.3f}, {cross_mags[2]:.3f}], "
        f"chosen_axis={chosen}, z_hint={np.round(z_hint, 3)}",
    )

    y_axis = np.cross(z_hint, forward)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(forward, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    R = np.eye(4)
    R[:3, 0] = forward
    R[:3, 1] = y_axis
    R[:3, 2] = z_axis
    q = quaternion_from_matrix(R)

    rospy.loginfo_throttle(
        2.0,
        f"[look_at {label}] Y={np.round(y_axis, 3)}, "
        f"Z={np.round(z_axis, 3)}, q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]",
    )

    return q


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
        correction_quat=_IDENTITY_QUAT,
        hint_axis=None,
    ):
        self.valve_frame = valve_frame
        self.approach_frame = approach_frame
        self.engage_frame = engage_frame
        self.approach_offset = approach_offset
        self.engage_offset = engage_offset
        self.correction_quat = np.asarray(correction_quat)
        self.has_correction = not np.allclose(self.correction_quat, _IDENTITY_QUAT)
        self.hint_axis = hint_axis
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
        self.front = _ValveConfig(
            valve_frame=rospy.get_param("~front_valve_frame", "tac/valve_front"),
            approach_frame=rospy.get_param(
                "~front_approach_frame", "valve_front_approach_target"
            ),
            engage_frame=rospy.get_param(
                "~front_engage_frame", "valve_front_engage_target"
            ),
            approach_offset=rospy.get_param("~front_approach_offset", 1.0),
            engage_offset=rospy.get_param("~front_engage_offset", 0.0),
            service_name=rospy.get_param(
                "~front_service", "set_publishing_valve_front"
            ),
        )

        # ---- Bottom valve ----
        # hint_axis=0 (mission-start X) so the target frame inherits the
        # mission heading regardless of valve pitch.  The default Z-hint
        # becomes degenerate when the valve normal is nearly vertical.
        self.bottom = _ValveConfig(
            valve_frame=rospy.get_param("~bottom_valve_frame", "tac/valve_bottom"),
            approach_frame=rospy.get_param(
                "~bottom_approach_frame", "valve_bottom_approach_target"
            ),
            engage_frame=rospy.get_param(
                "~bottom_engage_frame", "valve_bottom_engage_target"
            ),
            approach_offset=rospy.get_param("~bottom_approach_offset", 1.0),
            engage_offset=rospy.get_param("~bottom_engage_offset", 0.0),
            service_name=rospy.get_param(
                "~bottom_service", "set_publishing_valve_bottom"
            ),
            correction_quat=_BOTTOM_CORRECTION_QUAT,
            hint_axis=0,
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
    def _publish_valve(self, valve_cfg, mission_start_R):
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

        rospy.loginfo_throttle(
            2.0,
            f"[{valve_cfg.valve_frame}] pos={np.round(valve_pos, 3)}, "
            f"normal={np.round(valve_normal, 3)}, "
            f"has_correction={valve_cfg.has_correction}",
        )

        for target_frame, offset in [
            (valve_cfg.approach_frame, valve_cfg.approach_offset),
            (valve_cfg.engage_frame, valve_cfg.engage_offset),
        ]:
            target_pos = valve_pos + valve_normal * offset
            forward = -valve_normal  # point from offset position toward valve

            q_target = _look_at_quaternion(
                forward,
                mission_start_R,
                hint_axis=valve_cfg.hint_axis,
                label=target_frame,
            )

            # Apply body-frame correction so the target frame matches the
            # gripper / base_link axis convention for this valve type.
            if valve_cfg.has_correction:
                q_target = quaternion_multiply(q_target, valve_cfg.correction_quat)
                rospy.loginfo_throttle(
                    2.0,
                    f"[{target_frame}] after correction: "
                    f"q=[{q_target[0]:.4f}, {q_target[1]:.4f}, "
                    f"{q_target[2]:.4f}, {q_target[3]:.4f}]",
                )

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

            # Lookup mission_start_link once per cycle.
            ms_tf = self._lookup_tf(self.mission_start_frame, "mission_start_link")
            if ms_tf is None:
                rate.sleep()
                continue

            q_ms = np.array(
                [
                    ms_tf.transform.rotation.x,
                    ms_tf.transform.rotation.y,
                    ms_tf.transform.rotation.z,
                    ms_tf.transform.rotation.w,
                ]
            )
            mission_start_R = quaternion_matrix(q_ms)[:3, :3]

            rospy.loginfo_throttle(
                2.0,
                f"[mission_start] X={np.round(mission_start_R[:, 0], 3)}, "
                f"Y={np.round(mission_start_R[:, 1], 3)}, "
                f"Z={np.round(mission_start_R[:, 2], 3)}",
            )

            if self.front.enabled:
                self._publish_valve(self.front, mission_start_R)
            if self.bottom.enabled:
                self._publish_valve(self.bottom, mission_start_R)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = ValveTrajectoryPublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
