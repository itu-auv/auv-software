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
  4. RESUME_TRACKING (optional, default on) — after ~resume_delay seconds the
     tracker goes back to live tracking so the script can be re-run.

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

import sys

import rospy
import smach
from std_srvs.srv import Trigger

from auv_smach.initialize import DelayState
from auv_smach.valve import (
    CheckValveEngagementState,
    RotateGripperState,
    SetGripperTurnDirectionState,
)


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
    # 90 = the valve's physical travel; the excess (default 20) is overdrive,
    # relaxed back by the tracker after the stop is hit.
    degrees = float(rospy.get_param("~degrees", 110.0))
    perpendicular_threshold = float(rospy.get_param("~perpendicular_threshold", 0.02))
    in_plane_threshold = float(rospy.get_param("~in_plane_threshold", 0.05))
    confirm_duration = float(rospy.get_param("~confirm_duration", 0.3))
    resume = bool(rospy.get_param("~resume", True))
    resume_delay = float(rospy.get_param("~resume_delay", 3.0))

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
