#!/usr/bin/env python3
"""
Gripper roll tracker.

Drives the gripper servo so its fingers track the *roll* of a target TF frame
about a chosen axis, relative to the vehicle body (taluy/base_link). The body
and the gripper mount share roll, so base_link is used as the reference.

Servo mapping (per auv-hardware-interface): the published value is a raw PWM
pulse in microseconds, full travel 500..2500 us over -180..+180 deg
(=> 2000/360 = 5.5556 us/deg). A library clamps the reachable pulse to
[550, 2399] us. We treat 550 us as 0 deg (fingers stacked over z, one above /
one below), giving a usable servo travel of ~[0, 332.8] deg.

Symmetry: the gripper looks identical when flipped 180 deg, so any desired roll
has two equivalent servo positions (theta and theta+180). We reduce the desired
roll modulo 180, then pick whichever mechanical position is reachable and
closest to the current one (no needless flips). Note: because the travel spans
~333 deg (not a full 360), there is a small unreachable band just below 0 deg;
crossing roll = 0 there forces one 180 flip, which for a symmetric gripper just
swaps the two fingers.

Configuration (fully param-driven; no topic input needed):
    ~target_frame : TF frame whose roll to follow (e.g. tac/valve_front).
    ~axis         : axis in the target frame the fingers spin around
                    (default [1,0,0] = the valve flange normal).
    The optional ~roll_input topic (geometry_msgs/Vector3Stamped) overrides
    both at runtime: header.frame_id is the target frame (empty -> keep the
    param), vector is the axis. A zero vector is ignored.

Tracking starts passively as soon as the target frame appears in TF.

Valve-turn support (the reason this node exists):
  * ~set_turn_direction (auv_msgs/RotateGripper): tell the tracker which way
    the valve handle will be turned ("cw"/"ccw" as seen facing the valve;
    degrees = required headroom, 0 keeps the current/default 90). From then on
    TRACKING always parks the servo on a 180-deg branch with at least that
    much travel left in the turn direction, so the turn is always possible
    without a finger flip. Worst case a branch with ~153 deg headroom exists,
    so 90-deg turns are always satisfiable.
  * ~rotate (auv_msgs/RotateGripper): latch the current servo target, ramp to
    (latched +/- degrees) at ~turn_rate_dps, then HOLD (mod-180 folding and
    live tracking disabled — never flip while gripping the handle). The valve
    physically travels only NOMINAL_TURN_DEG (90); any request beyond that is
    treated as OVERDRIVE — the ramp pushes past the hard stop to guarantee the
    turn completes, then relaxes back to the nominal target so the servo does
    not stall against the stop while holding. Refuses if the overdriven target
    would leave the reachable window or no roll was ever tracked. Blocks until
    the ramp (out and back) completes; returns failure if interrupted.
  * ~hold (std_srvs/Trigger): latch the servo at its current tracked angle
    without rotating (stop live tracking, freeze in place). Used to lock the
    gripper roll before a maneuver so it stops chasing the target; resume with
    ~resume_tracking.
  * ~resume_tracking (std_srvs/Trigger): back to live tracking (after the
    jaws have released the handle, or after a ~hold).

Sign conventions (two independent calibration knobs):
    ~invert   : flips the sign of the *tracked* roll (servo vs perception).
    ~ccw_sign : servo-degrees per CCW handle-degree (+1 or -1): maps the
                operator's "handle cw/ccw as you face the valve" to the servo
                direction. Calibrate both on the bench once per gripper.

Run from the HOST with the AUV env (ROS_MASTER_URI -> orin):
    AUV; rosrun auv_control gripper_roll_tracker.py        (or python3 <path>)
"""
import math
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import UInt16
from std_srvs.srv import Trigger, TriggerResponse

from auv_msgs.srv import RotateGripper, RotateGripperResponse


def quat_to_matrix(q):
    """q = (x, y, z, w) -> 3x3 rotation matrix (orientation of frame in parent)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def twist_angle_about_axis(q, axis_unit):
    """Swing-twist: signed rotation (rad) of quaternion q about axis_unit.

    q = (x, y, z, w); axis_unit is a unit vector in the same frame as q's
    vector part (the parent/base frame here). Returns angle in (-pi, pi].
    """
    qvec = np.array([q[0], q[1], q[2]])
    proj = float(np.dot(qvec, axis_unit))
    return 2.0 * math.atan2(proj, q[3])


def _parse_direction(direction):
    """'ccw' -> +1, 'cw' -> -1 (handle sense, facing the valve), else None."""
    d = direction.strip().lower()
    if d == "ccw":
        return 1.0
    if d == "cw":
        return -1.0
    return None


class GripperRollTracker:
    MODE_TRACKING = "tracking"
    MODE_LATCHED = "latched"

    # Physical travel of the valve handle (deg). A rotate request beyond this
    # is overdrive: ramp past the hard stop, then relax back to this nominal
    # so the servo doesn't keep straining against the stop while holding.
    NOMINAL_TURN_DEG = 90.0

    def __init__(self):
        # Frames
        self.reference_frame = rospy.get_param("~reference_frame", "taluy/base_link")
        self.default_target = rospy.get_param("~target_frame", "")

        # Servo mapping / limits
        self.zero_pulse = int(rospy.get_param("~zero_pulse", 550))  # us at 0 deg
        self.min_pulse = int(rospy.get_param("~min_pulse", 550))
        self.max_pulse = int(rospy.get_param("~max_pulse", 2399))
        self.us_per_deg = float(rospy.get_param("~us_per_deg", 2000.0 / 360.0))

        # Calibration knobs
        self.offset_deg = float(rospy.get_param("~offset_deg", 0.0))
        self.invert = bool(rospy.get_param("~invert", False))
        # Servo-degrees per CCW handle-degree (+1/-1); see module docstring.
        self.ccw_sign = float(rospy.get_param("~ccw_sign", 1.0))
        self.ccw_sign = 1.0 if self.ccw_sign >= 0.0 else -1.0

        # Valve-turn behaviour
        self.turn_rate_dps = float(rospy.get_param("~turn_rate_dps", 45.0))
        self.headroom_deg = float(rospy.get_param("~turn_headroom_deg", 90.0))

        # Optional feature-vector mode: define roll geometrically (no quaternion
        # zero/sign ambiguity). feature_vector is a direction fixed in the
        # target frame (e.g. the object's grasp line); reference_vector is the
        # zero direction fixed in the reference frame. Roll is the signed angle
        # from reference_vector to feature_vector, about the axis, both
        # projected onto the plane perpendicular to the axis. If feature_vector
        # is empty, falls back to swing-twist on the axis alone.
        fv = rospy.get_param("~feature_vector", [])
        rv = rospy.get_param("~reference_vector", [0.0, 0.0, 1.0])
        self.feature_vec = np.array(fv, dtype=float) if fv else None
        self.reference_vec = np.array(rv, dtype=float)
        if np.linalg.norm(self.reference_vec) > 1e-9:
            self.reference_vec /= np.linalg.norm(self.reference_vec)

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        topic = rospy.get_param("~gripper_topic", "/taluy/actuators/gripper/set_angle")
        in_topic = rospy.get_param(
            "~roll_input", "/taluy/actuators/gripper/roll_target"
        )

        # Max reachable servo angle (deg) from the zero (550) reference.
        self.theta_max = (self.max_pulse - self.zero_pulse) / self.us_per_deg

        # ---- Mutable state (guarded by _lock) ----
        self._lock = threading.Lock()
        self.last_servo = None  # last commanded servo angle (deg), for continuity
        self._mode = self.MODE_TRACKING
        self._turning = False  # a rotate ramp is in flight
        self._turn_servo_sign = None  # +1/-1 expected turn dir in SERVO degrees
        self._target_frame = self.default_target if self.default_target else None

        # Default axis from params (in the target frame); topic can override.
        axis = np.array(rospy.get_param("~axis", [1.0, 0.0, 0.0]), dtype=float)
        if np.linalg.norm(axis) > 1e-9:
            self._axis_msg = axis / np.linalg.norm(axis)
        else:
            self._axis_msg = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(topic, UInt16, queue_size=10)
        self.sub = rospy.Subscriber(in_topic, Vector3Stamped, self._axis_cb)

        rospy.Service(
            "~set_turn_direction", RotateGripper, self._handle_set_turn_direction
        )
        rospy.Service("~rotate", RotateGripper, self._handle_rotate)
        rospy.Service("~hold", Trigger, self._handle_hold)
        rospy.Service("~resume_tracking", Trigger, self._handle_resume_tracking)

        rospy.loginfo(
            "gripper_roll_tracker: ref=%s target=%s topic=%s theta_max=%.1f deg "
            "(zero=%dus, clamp [%d,%d]) headroom=%.0f deg rate=%.0f deg/s",
            self.reference_frame,
            self._target_frame,
            topic,
            self.theta_max,
            self.zero_pulse,
            self.min_pulse,
            self.max_pulse,
            self.headroom_deg,
            self.turn_rate_dps,
        )

    # ------------------------------------------------------------------ #
    #  Inputs
    # ------------------------------------------------------------------ #
    def _axis_cb(self, msg):
        v = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        if np.linalg.norm(v) < 1e-6:
            rospy.logwarn_throttle(5.0, "roll axis is ~zero; ignoring")
            return
        frame = msg.header.frame_id if msg.header.frame_id else self.default_target
        if not frame:
            rospy.logwarn_throttle(
                5.0, "no target frame (empty frame_id and no ~target_frame param)"
            )
            return
        with self._lock:
            self._target_frame = frame
            self._axis_msg = v / np.linalg.norm(v)

    # ------------------------------------------------------------------ #
    #  Roll measurement
    # ------------------------------------------------------------------ #
    def _desired_roll_deg(self):
        """Return roll (deg) of target about the commanded axis, or None."""
        with self._lock:
            axis = self._axis_msg
            target_frame = self._target_frame
        if axis is None or target_frame is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.reference_frame, target_frame, rospy.Time(0), rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                10.0, "TF %s<-%s failed: %s", self.reference_frame, target_frame, e
            )
            return None

        r = tf.transform.rotation
        q = (r.x, r.y, r.z, r.w)
        # Axis is given in the target frame; express it in the reference frame.
        rot = quat_to_matrix(q)  # orientation of target in reference
        axis_ref = rot.dot(axis)
        n = np.linalg.norm(axis_ref)
        if n < 1e-9:
            return None
        axis_ref /= n

        if self.feature_vec is not None:
            # Geometric roll: signed angle from reference_vec to feature_vec
            # about axis_ref, both projected onto the plane perpendicular to it.
            f_base = rot.dot(self.feature_vec)
            f_p = f_base - np.dot(f_base, axis_ref) * axis_ref
            r_p = self.reference_vec - np.dot(self.reference_vec, axis_ref) * axis_ref
            if np.linalg.norm(f_p) < 1e-9 or np.linalg.norm(r_p) < 1e-9:
                rospy.logwarn_throttle(
                    2.0, "feature/reference vector parallel to axis; roll undefined"
                )
                return None
            f_p /= np.linalg.norm(f_p)
            r_p /= np.linalg.norm(r_p)
            cross = np.cross(r_p, f_p)
            roll = math.degrees(math.atan2(np.dot(axis_ref, cross), np.dot(r_p, f_p)))
        else:
            roll = math.degrees(twist_angle_about_axis(q, axis_ref))

        if self.invert:
            roll = -roll
        return roll + self.offset_deg

    # ------------------------------------------------------------------ #
    #  Roll -> servo (TRACKING mode)
    # ------------------------------------------------------------------ #
    def _roll_to_servo(self, roll_deg):
        """Map desired roll to a reachable servo angle in [0, theta_max] using
        the gripper's 180-deg symmetry.

        Candidates: the (one or two) positions in [0, theta_max] congruent to
        the desired roll mod 180. When a turn direction is set, candidates
        without ~turn_headroom_deg of travel left in that direction are
        dropped (for headroom <= ~153 deg a feasible branch always exists).
        Among the survivors, the one nearest the last command wins —
        hysteresis with a built-in 90-deg deadband, so no chattering; forced
        180 flips happen only when the current branch becomes infeasible or
        runs off a hard edge. On the first command we start near mid-travel
        for maximum headroom in both directions."""
        eff = roll_deg % 180.0  # symmetric: theta and theta+180 are identical
        candidates = [eff]
        if eff + 180.0 <= self.theta_max:
            candidates.append(eff + 180.0)

        with self._lock:
            turn_sign = self._turn_servo_sign
            ref = self.last_servo
        if ref is None:
            ref = self.theta_max / 2.0

        if turn_sign is not None:
            if turn_sign > 0.0:
                feasible = [
                    c for c in candidates if c <= self.theta_max - self.headroom_deg
                ]
            else:
                feasible = [c for c in candidates if c >= self.headroom_deg]
            if not feasible:
                rospy.logwarn_throttle(
                    5.0,
                    "no servo branch with %.0f deg headroom (roll=%.1f); "
                    "tracking without the headroom guarantee",
                    self.headroom_deg,
                    roll_deg,
                )
                feasible = candidates
        else:
            feasible = candidates

        return min(feasible, key=lambda c: abs(c - ref))

    def _servo_to_pulse(self, servo_deg):
        pulse = int(round(self.zero_pulse + servo_deg * self.us_per_deg))
        return max(self.min_pulse, min(self.max_pulse, pulse))

    def _publish_servo(self, servo_deg):
        """Publish a servo angle and record it as the continuity reference.
        Caller must hold _lock."""
        pulse = self._servo_to_pulse(servo_deg)
        self.last_servo = servo_deg
        self.pub.publish(UInt16(data=pulse))
        rospy.logdebug("servo=%.1f -> pulse=%d", servo_deg, pulse)

    # ------------------------------------------------------------------ #
    #  Services
    # ------------------------------------------------------------------ #
    def _handle_set_turn_direction(self, req):
        handle_sign = _parse_direction(req.direction)
        if handle_sign is None:
            return RotateGripperResponse(
                success=False,
                message=f"direction must be 'cw' or 'ccw', got '{req.direction}'",
            )
        if req.degrees < 0.0:
            return RotateGripperResponse(
                success=False, message="headroom degrees must be >= 0"
            )
        with self._lock:
            self._turn_servo_sign = self.ccw_sign * handle_sign
            if req.degrees > 0.0:
                self.headroom_deg = req.degrees
            msg = (
                f"turn direction set: handle {req.direction.lower()} "
                f"(servo sign {self._turn_servo_sign:+.0f}), "
                f"headroom {self.headroom_deg:.0f} deg"
            )
        rospy.loginfo("gripper_roll_tracker: %s", msg)
        return RotateGripperResponse(success=True, message=msg)

    def _ramp_to(self, target, step, rate):
        """Ramp the servo to `target` at the configured rate. Returns
        (ok, reason); reason is set when the ramp was interrupted."""
        while not rospy.is_shutdown():
            with self._lock:
                if self._mode != self.MODE_LATCHED:
                    return False, "turn interrupted by resume_tracking"
                remaining = target - self.last_servo
                if abs(remaining) <= step:
                    self._publish_servo(target)
                    return True, ""
                self._publish_servo(self.last_servo + math.copysign(step, remaining))
            rate.sleep()
        return False, "node shutdown"

    def _handle_rotate(self, req):
        """Latch the current servo target and ramp to (latched +/- degrees).

        The valve only travels NOMINAL_TURN_DEG (90); anything beyond is
        overdrive: ramp the full request (push through the hard stop), then
        relax back to the nominal target and hold there — undershoot
        insurance without a sustained servo stall.

        Blocks until the ramp (out and back) finishes. Mod-180 folding is
        disabled — the jaws are assumed to be gripping the handle, so the
        overdriven target must stay inside the physical window or the call is
        refused outright."""
        handle_sign = _parse_direction(req.direction)
        if handle_sign is None:
            return RotateGripperResponse(
                success=False,
                message=f"direction must be 'cw' or 'ccw', got '{req.direction}'",
            )
        if req.degrees <= 0.0:
            return RotateGripperResponse(
                success=False, message="rotate degrees must be > 0"
            )

        nominal = min(req.degrees, self.NOMINAL_TURN_DEG)
        overdrive = req.degrees - nominal

        with self._lock:
            if self._turning:
                return RotateGripperResponse(
                    success=False, message="a turn is already in progress"
                )
            if self.last_servo is None:
                return RotateGripperResponse(
                    success=False,
                    message="no roll tracked yet (never saw the target frame)",
                )
            start = self.last_servo
            servo_sign = self.ccw_sign * handle_sign
            target_full = start + servo_sign * req.degrees
            target_hold = start + servo_sign * nominal
            if not (0.0 <= target_full <= self.theta_max):
                return RotateGripperResponse(
                    success=False,
                    message=(
                        f"target {target_full:.1f} deg outside [0, "
                        f"{self.theta_max:.1f}] (start {start:.1f}); refusing "
                        f"to flip while gripped — check pre-positioning"
                    ),
                )
            self._mode = self.MODE_LATCHED
            self._turning = True

        rospy.loginfo(
            "gripper_roll_tracker: turning handle %s %.1f deg "
            "(nominal %.0f + overdrive %.0f; servo %.1f -> %.1f, "
            "hold at %.1f, %.0f deg/s)",
            req.direction.lower(),
            req.degrees,
            nominal,
            overdrive,
            start,
            target_full,
            target_hold,
            self.turn_rate_dps,
        )

        # Ramp (runs in this service handler thread; spin() holds off
        # publishing while _turning is set).
        step = self.turn_rate_dps / self.rate_hz
        rate = rospy.Rate(self.rate_hz)
        try:
            ok, reason = self._ramp_to(target_full, step, rate)
            if ok and overdrive > 0.0:
                # Relax back off the hard stop to the nominal target.
                ok, reason = self._ramp_to(target_hold, step, rate)
        finally:
            with self._lock:
                self._turning = False

        if not ok:
            return RotateGripperResponse(success=False, message=reason)
        return RotateGripperResponse(
            success=True,
            message=(
                f"turn complete ({nominal:.0f} deg + {overdrive:.0f} deg "
                f"overdrive), holding at {target_hold:.1f} deg"
            ),
        )

    def _handle_hold(self, req):
        """Freeze the servo at its current tracked angle without rotating
        (latch in place). Unlike `rotate`, this does not ramp the handle — it
        just stops live tracking so the gripper roll stays put through the
        approach/engage maneuver. Refuses if no roll was ever tracked (nothing
        to hold). `resume_tracking` unlatches it."""
        with self._lock:
            if self.last_servo is None:
                return TriggerResponse(
                    success=False,
                    message="no roll tracked yet (never saw the target frame)",
                )
            if self._turning:
                return TriggerResponse(
                    success=False, message="a turn is in progress; cannot hold"
                )
            was = self._mode
            self._mode = self.MODE_LATCHED
            held = self.last_servo
        rospy.loginfo(
            "gripper_roll_tracker: hold (was %s), latched at %.1f deg", was, held
        )
        return TriggerResponse(
            success=True, message=f"holding at {held:.1f} deg (was {was})"
        )

    def _handle_resume_tracking(self, req):
        with self._lock:
            was = self._mode
            self._mode = self.MODE_TRACKING
        rospy.loginfo("gripper_roll_tracker: resume tracking (was %s)", was)
        return TriggerResponse(success=True, message=f"tracking resumed (was {was})")

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self._lock:
                mode = self._mode
                turning = self._turning
            if mode == self.MODE_TRACKING:
                roll = self._desired_roll_deg()
                if roll is not None:
                    servo = self._roll_to_servo(roll)
                    with self._lock:
                        if self._mode == self.MODE_TRACKING:
                            self._publish_servo(servo)
            elif not turning:
                # LATCHED hold: keep refreshing the held target.
                with self._lock:
                    if self.last_servo is not None and not self._turning:
                        self._publish_servo(self.last_servo)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("gripper_roll_tracker")
    GripperRollTracker().spin()
