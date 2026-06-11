from .initialize import *
import math

import rospy
import smach
import smach_ros
import tf2_ros
from std_srvs.srv import SetBool, SetBoolRequest, Trigger

from auv_msgs.srv import RotateGripper, RotateGripperRequest
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SetAlignControllerTargetState,
    SetDepthState,
)
from auv_smach.initialize import DelayState
from auv_smach.tf_utils import get_tf_buffer


class _SetBoolServiceState(smach_ros.ServiceState):
    """Generic SetBool wrapper for a trajectory publisher's
    frame-publishing toggle."""

    def __init__(self, service_name: str, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            service_name,
            SetBool,
            request=SetBoolRequest(data=req),
        )


def _rotate_gripper_response_cb(userdata, response):
    """Map the service-level success flag onto the smach outcome (a refusal —
    e.g. not enough headroom — must abort the task, not 'succeed')."""
    if response.success:
        return "succeeded"
    rospy.logerr(f"RotateGripper service refused: {response.message}")
    return "aborted"


class SetGripperTurnDirectionState(smach_ros.ServiceState):
    """Tell a gripper roll tracker which way the valve handle will be turned
    ("cw"/"ccw" as seen facing the valve). From then on the tracker only parks
    the servo on branches with enough headroom to make that turn without a
    mid-grip finger flip. Call once, early — turn-readiness is a standing
    property of tracking, not a pre-engage ritual."""

    def __init__(
        self, tracker_namespace: str, direction: str, headroom_deg: float = 0.0
    ):
        smach_ros.ServiceState.__init__(
            self,
            f"{tracker_namespace}/set_turn_direction",
            RotateGripper,
            request=RotateGripperRequest(direction=direction, degrees=headroom_deg),
            response_cb=_rotate_gripper_response_cb,
        )


class RotateGripperState(smach_ros.ServiceState):
    """Turn the gripped valve handle: latches the tracked roll and ramps the
    servo by `degrees` in `direction` (handle sense, facing the valve). The
    service blocks until the ramp completes, so this state returns when the
    turn is physically done. Refusals (insufficient headroom, nothing tracked
    yet, turn already in flight) come back as 'aborted'."""

    def __init__(self, tracker_namespace: str, direction: str, degrees: float):
        smach_ros.ServiceState.__init__(
            self,
            f"{tracker_namespace}/rotate",
            RotateGripper,
            request=RotateGripperRequest(direction=direction, degrees=degrees),
            response_cb=_rotate_gripper_response_cb,
        )


class ResumeTrackingState(smach.State):
    """Call the tracker's resume_tracking Trigger service (back to live
    tracking after a latched turn). Safe whenever the jaws are NOT on the
    handle — e.g. at the very start of a valve task, where it clears a
    latched tracker left over from a previous mission run. Never call it
    while still gripped: the completed turn consumed the servo headroom on
    the current branch, so live tracking would flip the jaws 180 deg and
    crank the valve back."""

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


class CheckValveEngagementState(smach.State):
    """Poll TF until the gripper sits on the valve, with the position error
    split in the VALVE frame (+X is always the flange normal, for both the
    front and the bottom valve):

      perpendicular error = |x|        — engagement depth along the normal;
                                         strict: the jaws must reach the handle.
      in-plane error      = sqrt(y²+z²) — lateral offset across the valve face;
                                         loose: the jaws straddle a long handle.

    Both must hold for `confirm_duration` seconds (a single excursion resets
    the clock). `timeout=None` waits forever (bench use — the operator holds
    the valve, Ctrl-C is the way out); a finite timeout returns 'succeeded'
    with a loud warning (graceful degradation, same convention as
    CheckAlignmentState — attempting the turn slightly out of tolerance still
    beats guaranteed zero points)."""

    def __init__(
        self,
        gripper_frame: str,
        valve_frame: str,
        perpendicular_threshold: float = 0.02,
        in_plane_threshold: float = 0.05,
        confirm_duration: float = 0.3,
        timeout: float = None,
        rate_hz: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.gripper_frame = gripper_frame
        self.valve_frame = valve_frame
        self.perpendicular_threshold = perpendicular_threshold
        self.in_plane_threshold = in_plane_threshold
        self.confirm_duration = confirm_duration
        self.timeout = timeout
        self.rate_hz = rate_hz
        self.tf_buffer = get_tf_buffer()

    def _get_errors(self):
        """(perpendicular, in_plane) gripper position error in the valve
        frame, or (None, None) when TF is unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.valve_frame,
                self.gripper_frame,
                rospy.Time(0),
                rospy.Duration(0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0,
                "[CHECK_ENGAGEMENT] TF %s<-%s failed: %s",
                self.valve_frame,
                self.gripper_frame,
                e,
            )
            return None, None
        t = tf.transform.translation
        return abs(t.x), math.hypot(t.y, t.z)

    def execute(self, userdata):
        rospy.loginfo(
            "[CHECK_ENGAGEMENT] waiting for %s on %s: perp<%.3f m, "
            "in-plane<%.3f m for %.1f s%s",
            self.gripper_frame,
            self.valve_frame,
            self.perpendicular_threshold,
            self.in_plane_threshold,
            self.confirm_duration,
            "" if self.timeout is None else f" (timeout {self.timeout:.0f} s)",
        )
        rate = rospy.Rate(self.rate_hz)
        start_time = rospy.Time.now()
        within_since = None
        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            if (
                self.timeout is not None
                and (rospy.Time.now() - start_time).to_sec() >= self.timeout
            ):
                rospy.logwarn(
                    "[CHECK_ENGAGEMENT] timeout after %.0f s WITHOUT confirmed "
                    "engagement — continuing anyway (graceful degradation)",
                    self.timeout,
                )
                return "succeeded"

            perp, in_plane = self._get_errors()
            if perp is None:
                within_since = None
            elif (
                perp < self.perpendicular_threshold
                and in_plane < self.in_plane_threshold
            ):
                if within_since is None:
                    within_since = rospy.Time.now()
                held = (rospy.Time.now() - within_since).to_sec()
                rospy.loginfo_throttle(
                    0.5,
                    "[CHECK_ENGAGEMENT] perp=%.3f in-plane=%.3f m "
                    "(within, %.1f/%.1f s)",
                    perp,
                    in_plane,
                    held,
                    self.confirm_duration,
                )
                if held >= self.confirm_duration:
                    rospy.loginfo(
                        "[CHECK_ENGAGEMENT] engaged: perp=%.3f m in-plane=%.3f m",
                        perp,
                        in_plane,
                    )
                    return "succeeded"
            else:
                within_since = None
                rospy.loginfo_throttle(
                    1.0,
                    "[CHECK_ENGAGEMENT] perp=%.3f m in-plane=%.3f m",
                    perp,
                    in_plane,
                )

            rate.sleep()
        return "preempted"


class ValveTaskState(smach.State):
    """Aligns to a valve in two phases: approach → engage.

    A single trajectory publisher node (one per valve) emits both target
    frames simultaneously (approach + engage).  All frames are enabled
    once at the start via a single service call.

    Approach phase uses a Concurrence to stabilise the valve TF:
      1. AlignFrame drives toward the approach target.
      2. After a settle delay the publisher is disabled ("frozen") — the
         object map keeps broadcasting the last-computed TF, giving the
         alignment a rock-solid static target to converge on.
      When AlignFrame succeeds the Concurrence exits, regardless of
      whether the freeze has fired yet.

    After approach, the publisher is re-enabled (unfrozen) and the
    engage phase aligns the valve gripper link to the engage target.
    Engagement is confirmed by CheckValveEngagementState, which splits the
    gripper<->valve error in the valve frame: strict along the flange
    normal (the jaws must reach the handle), loose across the valve face
    (the jaws straddle a long handle). The align controller keeps holding
    the engage pose through the check and the turn.

    If `turn_direction` is set ("cw"/"ccw", handle sense facing the valve),
    the gripper roll tracker is told the direction up front (so tracking
    stays turn-ready with enough servo headroom) and, after engaging, the
    handle is turned `turn_degrees` while the align controller holds the
    engage pose.

    Parameterised so a single class covers both front and bottom valves —
    they differ only in the frame names, gripper link, and tracker namespace.
    """

    def __init__(
        self,
        valve_depth,
        gripper_frame: str = "taluy/base_link/valve_gripper_front_link",
        valve_frame: str = "tac/valve_front",
        approach_target_frame: str = "valve_front_approach_target",
        engage_target_frame: str = "valve_front_engage_target",
        publisher_service: str = "set_publishing_valve_front",
        freeze_delay: float = 5.0,
        tracker_namespace: str = "gripper_roll_tracker_front",
        turn_direction: str = "",
        turn_degrees: float = 90.0,
        engage_perpendicular_threshold: float = 0.02,
        engage_in_plane_threshold: float = 0.05,
        engage_confirm_duration: float = 0.3,
        engage_timeout: float = 30.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.state_machine.set_initial_state(
            ["RESUME_TRACKING" if turn_direction else "SET_VALVE_DEPTH"]
        )

        # -- Build the approach Concurrence --
        # Child 1: align to approach target (loose thresholds, long timeout)
        # Child 2: delay → freeze publisher (disable it so TF stops updating)
        # The Concurrence exits only when ALIGN_TO_APPROACH finishes; the
        # freeze branch runs to completion (or is preempted if alignment
        # finishes before the settle delay) without ending the Concurrence.
        approach_concurrence = smach.Concurrence(
            outcomes=["succeeded", "preempted", "aborted"],
            default_outcome="aborted",
            outcome_map={
                "succeeded": {"ALIGN_TO_APPROACH": "succeeded"},
                "preempted": {"ALIGN_TO_APPROACH": "preempted"},
                "aborted": {"ALIGN_TO_APPROACH": "aborted"},
            },
            child_termination_cb=lambda outcomes: outcomes["ALIGN_TO_APPROACH"]
            is not None,
        )
        with approach_concurrence:
            smach.Concurrence.add(
                "ALIGN_TO_APPROACH",
                AlignFrame(
                    source_frame=gripper_frame,
                    target_frame=approach_target_frame,
                    dist_threshold=0.05,
                    yaw_threshold=0.01,
                    confirm_duration=1.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    use_frame_depth=True,
                ),
            )

            freeze_sequence = smach.StateMachine(
                outcomes=["succeeded", "preempted", "aborted"]
            )
            with freeze_sequence:
                smach.StateMachine.add(
                    "WAIT_FOR_SETTLE",
                    DelayState(delay_time=freeze_delay),
                    transitions={
                        "succeeded": "FREEZE_PUBLISHER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                # Once frozen, this branch is done — it just returns
                # "succeeded" and idles while ALIGN_TO_APPROACH keeps running;
                # the Concurrence's child_termination_cb ignores this outcome.
                smach.StateMachine.add(
                    "FREEZE_PUBLISHER",
                    _SetBoolServiceState(publisher_service, req=False),
                    transitions={
                        "succeeded": "succeeded",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.Concurrence.add("FREEZE_SEQUENCE", freeze_sequence)

        # -- Wire the top-level state machine --
        with self.state_machine:
            if turn_direction:
                # Un-latch the tracker first (no-op on a fresh run): a
                # previous run's turn leaves it LATCHED holding the turned
                # handle angle, which would otherwise persist into this run.
                # Safe here — the vehicle is far from the valve, jaws empty.
                smach.StateMachine.add(
                    "RESUME_TRACKING",
                    ResumeTrackingState(tracker_namespace=tracker_namespace),
                    transitions={
                        "succeeded": "SET_TURN_DIRECTION",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                # Tell the roll tracker the turn direction: from here on,
                # tracking permanently keeps enough headroom for the turn.
                smach.StateMachine.add(
                    "SET_TURN_DIRECTION",
                    SetGripperTurnDirectionState(
                        tracker_namespace=tracker_namespace,
                        direction=turn_direction,
                        headroom_deg=turn_degrees,
                    ),
                    transitions={
                        "succeeded": "SET_VALVE_DEPTH",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth),
                transitions={
                    "succeeded": "ENABLE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Enable all trajectory frames at once.
            smach.StateMachine.add(
                "ENABLE_PUBLISHER",
                _SetBoolServiceState(publisher_service, req=True),
                transitions={
                    "succeeded": "WAIT_FOR_FRAMES",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRAMES",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "APPROACH_WITH_FREEZE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Approach phase (with concurrent freeze) --------
            smach.StateMachine.add(
                "APPROACH_WITH_FREEZE",
                approach_concurrence,
                transitions={
                    "succeeded": "UNFREEZE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Re-enable publisher so engage sees live TF.
            smach.StateMachine.add(
                "UNFREEZE_PUBLISHER",
                _SetBoolServiceState(publisher_service, req=True),
                transitions={
                    "succeeded": "SET_ENGAGE_ALIGN_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Engage phase --------
            # The gripper frame (not base_link centre) drives toward the
            # engage target; the align controller keeps holding it from here
            # through the turn. Engagement itself is judged against the
            # actual valve frame with the split thresholds (see
            # CheckValveEngagementState), not the controller's own error.
            smach.StateMachine.add(
                "SET_ENGAGE_ALIGN_TARGET",
                SetAlignControllerTargetState(
                    source_frame=gripper_frame,
                    target_frame=engage_target_frame,
                    max_linear_velocity=0.05,
                    max_angular_velocity=0.05,
                    use_depth=True,
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
                    perpendicular_threshold=engage_perpendicular_threshold,
                    in_plane_threshold=engage_in_plane_threshold,
                    confirm_duration=engage_confirm_duration,
                    timeout=engage_timeout,
                ),
                transitions={
                    "succeeded": "DISABLE_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "DISABLE_PUBLISHER",
                _SetBoolServiceState(publisher_service, req=False),
                transitions={
                    "succeeded": (
                        "ROTATE_VALVE" if turn_direction else "CANCEL_ALIGN_CONTROLLER"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # -------- Turn phase --------
            # The align controller is still holding the engage pose; the
            # rotate service latches the tracked roll and ramps the servo.
            # It blocks until the turn is physically complete.
            if turn_direction:
                smach.StateMachine.add(
                    "ROTATE_VALVE",
                    RotateGripperState(
                        tracker_namespace=tracker_namespace,
                        direction=turn_direction,
                        degrees=turn_degrees,
                    ),
                    transitions={
                        "succeeded": "CANCEL_ALIGN_CONTROLLER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
