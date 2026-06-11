#!/usr/bin/env python3
"""Hand-held valve gripper debug state machine.

Bench/pool test for the gripper roll tracking + turn pipeline WITHOUT moving
the vehicle (no align controller, no depth — gripper logic only). Run it after
tac_sea.launch (which starts the keypoint pipeline and the two gripper roll
trackers), then take the valve in your hands and move it toward the gripper:

  1. SET_TURN_DIRECTION — tells the tracker the handle turn direction, so
     tracking keeps enough servo headroom for the turn (same as valve.py).
  2. WAIT_FOR_PROXIMITY — the tracker keeps spinning the jaws to match the
     handle while you move the valve; this state watches the TF distance
     between the valve frame and the gripper link. When they stay within
     ~distance_threshold for ~confirm_duration, the grip is assumed.
  3. ROTATE_VALVE — latched, ramped turn of ~degrees in ~direction (blocks
     until the ramp completes), exactly as in valve.py.
  4. RESUME_TRACKING (optional, default on) — after ~resume_delay seconds the
     tracker goes back to live tracking so the script can be re-run.

The turn direction has NO default — it must be given explicitly:

    rosrun auv_smach valve_gripper_debug.py _valve:=front _direction:=cw
    rosrun auv_smach valve_gripper_debug.py _valve:=bottom _direction:=ccw \\
        _degrees:=90 _distance_threshold:=0.15 _confirm_duration:=1.0

Direction convention: "cw"/"ccw" is the sense the valve HANDLE turns, as seen
facing the valve face (looking down at it, for the bottom valve).
"""

import sys

import numpy as np
import rospy
import smach
import tf2_ros
from std_srvs.srv import Trigger

from auv_smach.initialize import DelayState
from auv_smach.valve import RotateGripperState, SetGripperTurnDirectionState


class WaitForProximityState(smach.State):
    """Poll TF until source and target frames stay within distance_threshold
    of each other for confirm_duration seconds. No timeout — the operator is
    holding the valve; Ctrl-C (preempt/shutdown) is the way out."""

    def __init__(
        self,
        source_frame: str,
        target_frame: str,
        distance_threshold: float,
        confirm_duration: float,
        rate_hz: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.distance_threshold = distance_threshold
        self.confirm_duration = confirm_duration
        self.rate_hz = rate_hz
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def execute(self, userdata):
        rospy.loginfo(
            "[WAIT_FOR_PROXIMITY] move the valve to the gripper: waiting for "
            "|%s <-> %s| < %.2f m for %.1f s",
            self.source_frame,
            self.target_frame,
            self.distance_threshold,
            self.confirm_duration,
        )
        rate = rospy.Rate(self.rate_hz)
        within_since = None
        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            distance = None
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.source_frame,
                    self.target_frame,
                    rospy.Time(0),
                    rospy.Duration(0.1),
                )
                t = tf.transform.translation
                distance = float(np.linalg.norm([t.x, t.y, t.z]))
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    5.0, "[WAIT_FOR_PROXIMITY] TF not available yet: %s", e
                )

            if distance is not None:
                if distance < self.distance_threshold:
                    if within_since is None:
                        within_since = rospy.Time.now()
                    held = (rospy.Time.now() - within_since).to_sec()
                    rospy.loginfo_throttle(
                        0.5,
                        "[WAIT_FOR_PROXIMITY] d=%.3f m (within, %.1f/%.1f s)",
                        distance,
                        held,
                        self.confirm_duration,
                    )
                    if held >= self.confirm_duration:
                        rospy.loginfo(
                            "[WAIT_FOR_PROXIMITY] grip assumed at d=%.3f m",
                            distance,
                        )
                        return "succeeded"
                else:
                    within_since = None
                    rospy.loginfo_throttle(
                        1.0, "[WAIT_FOR_PROXIMITY] d=%.3f m", distance
                    )

            rate.sleep()
        return "preempted"


class ResumeTrackingState(smach.State):
    """Call the tracker's resume_tracking Trigger service."""

    def __init__(self, tracker_namespace: str):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.service_name = f"{tracker_namespace}/resume_tracking"

    def execute(self, userdata):
        try:
            rospy.wait_for_service(self.service_name, timeout=5.0)
            resp = rospy.ServiceProxy(self.service_name, Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logerr("[RESUME_TRACKING] %s failed: %s", self.service_name, e)
            return "aborted"
        rospy.loginfo("[RESUME_TRACKING] %s", resp.message)
        return "succeeded" if resp.success else "aborted"


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
    degrees = float(rospy.get_param("~degrees", 90.0))
    distance_threshold = float(rospy.get_param("~distance_threshold", 0.15))
    confirm_duration = float(rospy.get_param("~confirm_duration", 1.0))
    resume = bool(rospy.get_param("~resume", True))
    resume_delay = float(rospy.get_param("~resume_delay", 3.0))

    # Same frames/namespaces as main_state_machine.py / tac_sea.launch.
    valve_frame = f"tac/valve_{valve}"
    gripper_frame = f"{vehicle_ns}/base_link/valve_gripper_{valve}_link"
    tracker_namespace = f"/{vehicle_ns}/gripper_roll_tracker_{valve}"

    rospy.loginfo(
        "valve_gripper_debug: valve=%s direction=%s degrees=%.1f "
        "(tracker=%s, %s <-> %s, d<%.2fm for %.1fs)",
        valve,
        direction,
        degrees,
        tracker_namespace,
        gripper_frame,
        valve_frame,
        distance_threshold,
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
                "succeeded": "WAIT_FOR_PROXIMITY",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        smach.StateMachine.add(
            "WAIT_FOR_PROXIMITY",
            WaitForProximityState(
                source_frame=gripper_frame,
                target_frame=valve_frame,
                distance_threshold=distance_threshold,
                confirm_duration=confirm_duration,
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
                "succeeded": "WAIT_BEFORE_RESUME" if resume else "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        if resume:
            smach.StateMachine.add(
                "WAIT_BEFORE_RESUME",
                DelayState(delay_time=resume_delay),
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
