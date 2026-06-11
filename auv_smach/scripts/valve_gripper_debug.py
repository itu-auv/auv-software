#!/usr/bin/env python3
"""Hand-held valve gripper debug state machine.

Bench/pool test for the gripper roll tracking + turn pipeline WITHOUT moving
the vehicle (no align controller, no depth — gripper logic only). Run it after
tac_sea.launch (which starts the keypoint pipeline and the two gripper roll
trackers), then take the valve in your hands and move it toward the gripper:

  1. SET_TURN_DIRECTION — tells the tracker the handle turn direction, so
     tracking keeps enough servo headroom for the turn (same as valve.py).
  2. CHECK_ENGAGEMENT — the tracker keeps spinning the jaws to match the
     handle while you move the valve; this state watches the gripper<->valve
     TF error SPLIT IN THE VALVE FRAME (same state as valve.py):
       perpendicular (|x|, along the flange normal) < ~perpendicular_threshold
       in-plane (sqrt(y²+z²), across the valve face) < ~in_plane_threshold
     both held for ~confirm_duration. No timeout — the operator is holding
     the valve; Ctrl-C is the way out.
  3. ROTATE_VALVE — latched, ramped turn of ~degrees in ~direction (blocks
     until the ramp completes), exactly as in valve.py. The valve physically
     travels only 90 deg: anything beyond is overdrive — the tracker pushes
     past the stop, then relaxes back to 90 and holds there (no stall).
  4. WAIT_FOR_DISENGAGEMENT + RESUME_TRACKING (optional, default on) — resume
     is gated on you PULLING THE VALVE OUT of the jaws: only once the
     gripper<->valve TF separation exceeds ~disengage_distance for
     ~disengage_confirm seconds is live tracking re-enabled. Resuming while
     still gripped would let the tracker's headroom logic flip the jaws 180
     deg and crank the valve right back (the turn consumed the headroom on
     the current servo branch). If you Ctrl-C before pulling the valve out,
     the tracker just stays safely latched — resume manually with
         rosservice call /taluy/gripper_roll_tracker_<valve>/resume_tracking
     (Note: the gate needs the camera to SEE the valve move away; the object
     map rebroadcasts the last pose, so with the valve out of view the
     separation never grows — use the manual service in that case.)

The turn direction has NO default — it must be given explicitly:

    rosrun auv_smach valve_gripper_debug.py _valve:=front _direction:=cw
    rosrun auv_smach valve_gripper_debug.py _valve:=bottom _direction:=ccw \\
        _degrees:=110 _perpendicular_threshold:=0.02 \\
        _in_plane_threshold:=0.05 _confirm_duration:=0.3

(NOTE the leading underscores — `name:=value` without one is a ROS name
remap, not a param, and silently does nothing.)

Direction convention: "cw"/"ccw" is the sense the valve HANDLE turns, as seen
facing the valve face (looking down at it, for the bottom valve).
"""

import math
import sys

import rospy
import smach
import tf2_ros

from auv_smach.tf_utils import get_tf_buffer
from auv_smach.valve import (
    CheckValveEngagementState,
    ResumeTrackingState,
    RotateGripperState,
    SetGripperTurnDirectionState,
)


class WaitForDisengagementState(smach.State):
    """Block until the gripper<->valve TF separation exceeds `distance` for
    `confirm_duration` seconds — i.e. the operator has pulled the valve out of
    the jaws. Gating resume_tracking on this (instead of a timer) is what
    prevents the tracker from flipping the jaws 180 deg while still gripped:
    the completed turn consumed the headroom on the current servo branch, so
    live tracking would immediately jump to the other branch. No timeout —
    Ctrl-C leaves the tracker safely latched."""

    def __init__(
        self,
        gripper_frame: str,
        valve_frame: str,
        distance: float,
        confirm_duration: float,
        rate_hz: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.gripper_frame = gripper_frame
        self.valve_frame = valve_frame
        self.distance = distance
        self.confirm_duration = confirm_duration
        self.rate_hz = rate_hz
        self.tf_buffer = get_tf_buffer()

    def execute(self, userdata):
        rospy.loginfo(
            "[WAIT_FOR_DISENGAGEMENT] pull the valve out of the jaws: waiting "
            "for |%s <-> %s| > %.2f m for %.1f s before resuming tracking "
            "(Ctrl-C keeps the tracker latched; resume manually if the camera "
            "can't see the valve move away)",
            self.gripper_frame,
            self.valve_frame,
            self.distance,
            self.confirm_duration,
        )
        rate = rospy.Rate(self.rate_hz)
        clear_since = None
        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            separation = None
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.valve_frame,
                    self.gripper_frame,
                    rospy.Time(0),
                    rospy.Duration(0.1),
                )
                t = tf.transform.translation
                separation = math.sqrt(t.x**2 + t.y**2 + t.z**2)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    5.0, "[WAIT_FOR_DISENGAGEMENT] TF not available: %s", e
                )

            if separation is None or separation <= self.distance:
                clear_since = None
                if separation is not None:
                    rospy.loginfo_throttle(
                        1.0,
                        "[WAIT_FOR_DISENGAGEMENT] d=%.3f m (still close)",
                        separation,
                    )
            else:
                if clear_since is None:
                    clear_since = rospy.Time.now()
                held = (rospy.Time.now() - clear_since).to_sec()
                rospy.loginfo_throttle(
                    0.5,
                    "[WAIT_FOR_DISENGAGEMENT] d=%.3f m (clear, %.1f/%.1f s)",
                    separation,
                    held,
                    self.confirm_duration,
                )
                if held >= self.confirm_duration:
                    rospy.loginfo(
                        "[WAIT_FOR_DISENGAGEMENT] disengaged at d=%.3f m",
                        separation,
                    )
                    return "succeeded"

            rate.sleep()
        return "preempted"


def main():
    rospy.init_node("valve_gripper_debug_sm")

    # -- Required: turn direction (deliberately no default) --
    direction = rospy.get_param("~direction", "").strip().lower()
    if direction not in ("cw", "ccw"):
        rospy.logfatal(
            "~direction is required: pass _direction:=cw or _direction:=ccw "
            "(handle turn sense as seen facing the valve)"
        )
        sys.exit(1)

    valve = rospy.get_param("~valve", "front").strip().lower()
    if valve not in ("front", "bottom"):
        rospy.logfatal("~valve must be 'front' or 'bottom', got '%s'", valve)
        sys.exit(1)

    vehicle_ns = rospy.get_param("~vehicle_namespace", "taluy")
    # 90 = the valve's physical travel; the excess (default 20) is overdrive,
    # relaxed back by the tracker after the stop is hit.
    degrees = float(rospy.get_param("~degrees", 110.0))
    perpendicular_threshold = float(rospy.get_param("~perpendicular_threshold", 0.02))
    in_plane_threshold = float(rospy.get_param("~in_plane_threshold", 0.05))
    confirm_duration = float(rospy.get_param("~confirm_duration", 0.3))
    resume = bool(rospy.get_param("~resume", True))
    disengage_distance = float(rospy.get_param("~disengage_distance", 0.15))
    disengage_confirm = float(rospy.get_param("~disengage_confirm", 1.0))

    # Same frames/namespaces as main_state_machine.py / tac_sea.launch.
    valve_frame = f"tac/valve_{valve}"
    gripper_frame = f"{vehicle_ns}/base_link/valve_gripper_{valve}_link"
    tracker_namespace = f"/{vehicle_ns}/gripper_roll_tracker_{valve}"

    rospy.loginfo(
        "valve_gripper_debug: valve=%s direction=%s degrees=%.1f "
        "(tracker=%s, %s <-> %s, perp<%.3fm in-plane<%.3fm for %.1fs)",
        valve,
        direction,
        degrees,
        tracker_namespace,
        gripper_frame,
        valve_frame,
        perpendicular_threshold,
        in_plane_threshold,
        confirm_duration,
    )

    sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])
    with sm:
        smach.StateMachine.add(
            "SET_TURN_DIRECTION",
            SetGripperTurnDirectionState(
                tracker_namespace=tracker_namespace,
                direction=direction,
                headroom_deg=degrees,
            ),
            transitions={
                "succeeded": "CHECK_ENGAGEMENT",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        smach.StateMachine.add(
            "CHECK_ENGAGEMENT",
            CheckValveEngagementState(
                gripper_frame=gripper_frame,
                valve_frame=valve_frame,
                perpendicular_threshold=perpendicular_threshold,
                in_plane_threshold=in_plane_threshold,
                confirm_duration=confirm_duration,
                timeout=None,  # operator-held: wait forever, Ctrl-C to stop
            ),
            transitions={
                "succeeded": "ROTATE_VALVE",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        smach.StateMachine.add(
            "ROTATE_VALVE",
            RotateGripperState(
                tracker_namespace=tracker_namespace,
                direction=direction,
                degrees=degrees,
            ),
            transitions={
                "succeeded": "WAIT_FOR_DISENGAGEMENT" if resume else "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        if resume:
            smach.StateMachine.add(
                "WAIT_FOR_DISENGAGEMENT",
                WaitForDisengagementState(
                    gripper_frame=gripper_frame,
                    valve_frame=valve_frame,
                    distance=disengage_distance,
                    confirm_duration=disengage_confirm,
                ),
                transitions={
                    "succeeded": "RESUME_TRACKING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESUME_TRACKING",
                ResumeTrackingState(tracker_namespace=tracker_namespace),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    outcome = sm.execute()
    rospy.loginfo("valve_gripper_debug: finished with outcome '%s'", outcome)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
