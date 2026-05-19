#!/usr/bin/env python3
"""Pinger-aware mission runner.

Same `test_states:=a,b,c` convention as test_follow_waypoints — tokens run in
order, comma-separated. Extra token types:

    init            -> InitializeState()
    path<N>         -> DynamicPathExecutionState for that GUI-drawn path
                       (unconditional — exactly like test_follow_waypoints)
    <task>          -> a task state: octagon, torpedo, bin, gate, slalom
                       (params from ~tasks.<task>.params)
    <pinger token>  -> the ONE conditional block (default name
                       "octagon_torpedo_pinger").

KEY POINT — the pinger token does NOT decide at startup. It is built as a
real in-graph decision: a PingerDecisionState runs *when the sequence
reaches it* (i.e. after gate/bin/etc. have already revealed the object
frames), waits for octagon/torpedo positions up to a timeout, then branches:

    ...prev... -> DECIDE ->(octagon nearer) OCT_octagon -> OCT_path1 -> OCT_torpedo -> ...next...
                        \->(torpedo nearer) TOR_torpedo -> TOR_path2 -> TOR_octagon -/

Both branches rejoin at whatever token follows. Everything else flows
normally; only the octagon<->torpedo ordering is conditional.

Example:
    test_states:="init, gate, path3, bin, octagon_torpedo_pinger"

`dry_run:=true` swaps tasks/paths for stubs (the decision still runs for
real, so it needs the positions resolvable).

See config/pinger_mission_example.yaml for the schema.
"""

import rospy
import smach

from auv_smach.initialize import InitializeState
from auv_smach.octagon import OctagonTaskState, OctagonFramePublisherServiceState
from auv_smach.torpedo import TorpedoTaskState, TorpedoTargetFramePublisherServiceState
from auv_smach.bin import BinTaskState
from auv_smach.gate import NavigateThroughGateState
from auv_smach.slalom import NavigateThroughSlalomState
from auv_smach.waypoints import DynamicPathExecutionState
from auv_smach.pinger_decision import PingerDecisionState

from auv_smach.tf_utils import get_base_link
from auv_smach.common import (
    AlignFrame,
    SetDepthState,
    DynamicPathState,
    SearchForPropState,
    SetDetectionState,
    SetDetectionFocusState,
    CancelAlignControllerState,
    CheckForTransformState,
)
from auv_smach.initialize import DelayState

from auv_msgs.srv import SetDepth, SetDepthRequest
import tf2_ros


class SyncDepthToCurrentState(smach.State):
    """Reads the current AUV depth via TF and calls ``set_depth`` with that
    value so the depth controller tracks the *actual* depth instead of a
    stale target.  Always returns ``succeeded``."""

    def __init__(self, frame_id: str = "odom"):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.frame_id = frame_id
        from auv_smach.tf_utils import get_tf_buffer
        self.tf_buffer = get_tf_buffer()
        self.base_frame = get_base_link()

    def execute(self, userdata):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frame_id, self.base_frame, rospy.Time(0), rospy.Duration(4.0)
            )
            current_z = transform.transform.translation.z

            rospy.wait_for_service("set_depth", timeout=5.0)
            srv = rospy.ServiceProxy("set_depth", SetDepth)
            req = SetDepthRequest()
            req.target_depth = current_z
            req.frame_id = self.frame_id
            req.max_velocity = 0.0
            srv(req)
            rospy.loginfo(
                f"[SyncDepthToCurrentState] Depth synced to current: {current_z:.3f}m"
            )
        except Exception as e:
            rospy.logwarn(f"[SyncDepthToCurrentState] Failed to sync depth: {e}")
        return "succeeded"


# ---------------------------------------------------------------------------
# Align-only task variants (used when align_only:=true).
#
# Purpose: validate the pinger octa/torpe selection + path1/path2 flow
# WITHOUT performing the tasks. After the pinger picks the near object the
# AUV approaches it and aligns, then succeeds (no bottle grab / firing).
#
# These mirror the EARLY part of the real octagon/torpedo tasks: focus the
# detector, search for the prop, then enable the object-frame publisher.
# That detection step is what puts octagon_link / torpedo_target into the
# object map in the first place — InitializeState's CLEAR_OBJECT_MAP wipes
# the map, so without re-detecting here the pinger/approach frames never
# come back. (The vision pipeline runs in sim: sim_bbox_node ->
# camera_detection_pose_estimator -> object_map_tf_server; the launch
# remaps set_front_camera_focus -> vision/... so the focus call resolves.)
#
# Flow: focus -> search -> enable frame publisher -> wait -> set depth ->
#       DynamicPath toward object -> AlignFrame -> succeeded.
#
# It stops right BEFORE the part the real task would do next (bottle grab /
# torpedo firing). The real mission code (octagon.py / torpedo.py) is NOT
# touched — this lives only in the test runner. Kept deliberately close to
# the real task's early states so behaviour matches the full mission.
# ---------------------------------------------------------------------------


class OctagonAlignOnlyState(smach.State):
    """Octagon align-only: focus -> search -> enable frame -> approach ->
    align -> succeed.

    Mirrors the early states of the real OctagonTaskState (detection focus +
    search + frame publisher) so octagon_link enters the object map exactly
    like in the full mission, then approaches octagon_closer_link and aligns.
    Stops before the bottle grab / surfacing.
    """

    def __init__(self, octagon_depth: float, animal: str = "sawfish", **_ignored):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_INITIAL_DEPTH",
                SetDepthState(depth=-1.2, depth_threshold=0.2, timeout=10.0),
                transitions={
                    "succeeded": "FOCUS_ON_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_OCTAGON",
                SetDetectionFocusState(focus_object="octagon"),
                transitions={
                    "succeeded": "FIND_AIM_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AIM_OCTAGON",
                SearchForPropState(
                    look_at_frame="octagon_link",
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ENABLE_OCTAGON_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_OCTAGON_FRAME_PUBLISHER",
                OctagonFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_OCTAGON_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_OCTAGON_FRAME",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "SET_OCTAGON_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(depth=octagon_depth, depth_threshold=0.2, timeout=5.0),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame="octagon_closer_link",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_CLOSE_APPROACH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="octagon_closer_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=60.0,
                    # Hold position while aligned; we cancel explicitly in the
                    # next state so the controller doesn't get yanked from
                    # octagon_closer_link straight to path1's far first
                    # waypoint (that jump is what made the AUV lurch/oscillate
                    # on the transition).
                    cancel_on_success=False,
                ),
                transitions={
                    # Real task continues to bottle grab here; we stop instead.
                    "succeeded": "SYNC_DEPTH_AFTER_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SYNC_DEPTH_AFTER_ALIGN",
                SyncDepthToCurrentState(),
                transitions={
                    "succeeded": "CANCEL_ALIGN_AFTER_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                # Cleanly release the octagon align target BEFORE path1 starts.
                # path1 will set its own align target from scratch; cancelling
                # here (instead of letting path1 abruptly retarget the live
                # controller) removes the sudden octagon->far-waypoint setpoint
                # jump that caused the post-align oscillation.
                "CANCEL_ALIGN_AFTER_OCTAGON",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "STABILIZE_AFTER_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                # Brief settle so the AUV is steady (depth held by the
                # set_depth call above) before path1 grabs control.
                "STABILIZE_AFTER_OCTAGON",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo(
            "[test_pinger_mission] OCTAGON align-only: focus -> search -> "
            "enable frame -> approach -> align to octagon_closer_link -> "
            "succeed (no bottle grab)."
        )
        outcome = self.state_machine.execute()
        return "preempted" if outcome is None else outcome


class TorpedoAlignOnlyState(smach.State):
    """Torpedo align-only: focus -> search -> enable frame -> approach ->
    align -> succeed.

    Mirrors the early states of the real TorpedoTaskState (front camera +
    detection focus + search + frame publisher) so torpedo_target enters the
    object map exactly like in the full mission, then approaches and aligns.
    Stops before the realsense / hole detection / firing.
    """

    def __init__(
        self,
        torpedo_map_depth,
        torpedo_target_frame="torpedo_target",
        **_ignored,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA_FOCUS",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "FOCUS_ON_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_TORPEDO",
                SetDetectionFocusState(focus_object="torpedo"),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoTargetFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "SET_TORPEDO_MAP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_MAP_DEPTH",
                SetDepthState(depth=torpedo_map_depth, depth_threshold=0.2, timeout=10.0),
                transitions={
                    "succeeded": "FIND_AND_AIM_TORPEDO_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_TORPEDO_MAP",
                SearchForPropState(
                    look_at_frame="torpedo_map_link",
                    alignment_frame="torpedo_map_travel_start",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame=self.base_link,
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_FRAME",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "PATH_TO_TORPEDO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_TO_TORPEDO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame=torpedo_target_frame,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_CLOSE_APPROACH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=torpedo_target_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=10.0,
                    # Hold while aligned; cancelled explicitly in the next
                    # state (same reasoning as OctagonAlignOnlyState — avoids
                    # the abrupt retarget jump into the next path).
                    cancel_on_success=False,
                ),
                transitions={
                    # Real task continues to firing here; we stop instead.
                    "succeeded": "SYNC_DEPTH_AFTER_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SYNC_DEPTH_AFTER_ALIGN",
                SyncDepthToCurrentState(),
                transitions={
                    "succeeded": "CANCEL_ALIGN_AFTER_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_AFTER_TORPEDO",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "STABILIZE_AFTER_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "STABILIZE_AFTER_TORPEDO",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo(
            "[test_pinger_mission] TORPEDO align-only: focus -> search -> "
            "enable frame -> approach -> align to torpedo_target -> "
            "succeed (no firing)."
        )
        outcome = self.state_machine.execute()
        return "preempted" if outcome is None else outcome


# ---------------------------------------------------------------------------
# Pre-survey states (run BEFORE the pinger decision).
#
# PingerDecisionState compares pinger distance to octagon_link / torpedo_target
# — but those are object-map frames that only exist once the AUV has *seen*
# the object (detection class -> object_map_tf_server). In the real mission
# the AUV passes by both on the way and they get mapped naturally; the pinger
# token then just picks the order. In this isolated test there is no such
# fly-by, so we explicitly look at each object once first (focus + search,
# same detection path as the real tasks) to put both into the object map.
#
# Survey == the early detection states of the real task, WITHOUT enabling the
# target-frame publisher, approaching, or aligning. Just: see it, map it.
# The pinger decision then runs with both frames available, exactly as its
# docstring assumes ("the underlying object must already be in the map").
# ---------------------------------------------------------------------------


class OctagonSurveyState(smach.State):
    """Look at the octagon once so octagon_link enters the object map.

    focus=octagon -> SearchForProp(octagon_link). No approach / align /
    frame publisher — just enough for detection to map it.
    """

    def __init__(self, octagon_depth: float = -1.2, animal: str = "sawfish",
                 **_ignored):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        with self.state_machine:
            # If octagon_link is already in the object map (seen during an
            # earlier task), skip the whole survey — it's pure overhead.
            # 'succeeded' = frame live -> skip; 'aborted' = not mapped within
            # the short probe -> run the real survey.
            smach.StateMachine.add(
                "CHECK_OCTAGON_ALREADY_MAPPED",
                CheckForTransformState(
                    source_frame=self.base_link,
                    target_frame="octagon_link",
                    timeout=1.5,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "aborted": "SET_OCTAGON_SURVEY_DEPTH",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_SURVEY_DEPTH",
                SetDepthState(depth=-1.2, depth_threshold=0.2, timeout=10.0),
                transitions={
                    "succeeded": "FOCUS_ON_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_OCTAGON",
                SetDetectionFocusState(focus_object="octagon"),
                transitions={
                    "succeeded": "FIND_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_OCTAGON",
                SearchForPropState(
                    look_at_frame="octagon_link",
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_OCTAGON_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_OCTAGON_MAP",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo(
            "[test_pinger_mission] OCTAGON survey: look at octagon so "
            "octagon_link enters the object map (no approach)."
        )
        outcome = self.state_machine.execute()
        return "preempted" if outcome is None else outcome


class TorpedoSurveyState(smach.State):
    """Look at the torpedo once so torpedo_target enters the object map.

    front camera + focus=torpedo -> SearchForProp(torpedo_map_link). No
    approach / align / frame publisher — just enough for detection to map it.
    """

    def __init__(self, torpedo_map_depth=-1.25, **_ignored):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        with self.state_machine:
            # Skip the survey if torpedo_map_link is already mapped (seen
            # during an earlier task). 'succeeded' = frame live -> skip;
            # 'aborted' = not mapped within the short probe -> real survey.
            smach.StateMachine.add(
                "CHECK_TORPEDO_ALREADY_MAPPED",
                CheckForTransformState(
                    source_frame=self.base_link,
                    target_frame="torpedo_map_link",
                    timeout=1.5,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "aborted": "ENABLE_FRONT_CAMERA_FOCUS",
                    "preempted": "preempted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA_FOCUS",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "FOCUS_ON_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_TORPEDO",
                SetDetectionFocusState(focus_object="torpedo"),
                transitions={
                    "succeeded": "SET_TORPEDO_SURVEY_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_SURVEY_DEPTH",
                SetDepthState(
                    depth=torpedo_map_depth, depth_threshold=0.2, timeout=10.0
                ),
                transitions={
                    "succeeded": "FIND_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_TORPEDO",
                SearchForPropState(
                    look_at_frame="torpedo_map_link",
                    alignment_frame="torpedo_map_travel_start",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame=self.base_link,
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_MAP",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo(
            "[test_pinger_mission] TORPEDO survey: look at torpedo so "
            "torpedo_target enters the object map (no approach)."
        )
        outcome = self.state_machine.execute()
        return "preempted" if outcome is None else outcome


TASK_STATE_CLASSES = {
    "octagon": OctagonTaskState,
    "torpedo": TorpedoTaskState,
    "bin": BinTaskState,
    "gate": NavigateThroughGateState,
    "slalom": NavigateThroughSlalomState,
}

# Align-only stand-ins, selected when align_only:=true. Only octagon/torpedo
# have one — other tasks fall back to their normal class.
ALIGN_ONLY_STATE_CLASSES = {
    "octagon": OctagonAlignOnlyState,
    "torpedo": TorpedoAlignOnlyState,
}

_PATH_KWARG_KEYS = (
    "final_align",
    "final_align_dist_threshold",
    "final_align_yaw_threshold",
    "final_align_timeout",
    "final_align_confirm_duration",
    "max_linear_velocity",
    "max_angular_velocity",
    "keep_orientation",
    "angle_offset",
    "wait_timeout",
)


class _StubState(smach.State):
    """Stand-in used in dry_run to validate flow without acting."""

    def __init__(self, label):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self._label = label

    def execute(self, userdata):
        rospy.loginfo("[test_pinger_mission] (dry_run) stub: %s", self._label)
        return "succeeded"


def _autodiscover_waypoints(path_name, reference_frame, max_waypoints):
    """Find pathN_wp1..wpK live in TF (published by the waypoint GUI).

    Same behaviour as test_follow_waypoints: we follow whatever the GUI
    actually drew, not a fixed count from the YAML. Scans wp1, wp2, ...
    against `reference_frame` and stops at the first gap.
    """
    from auv_smach.tf_utils import get_tf_buffer

    buffer = get_tf_buffer()
    frames = []
    for i in range(1, max_waypoints + 1):
        frame = f"{path_name}_wp{i}"
        try:
            buffer.lookup_transform(
                reference_frame, frame, rospy.Time(0), rospy.Duration(0.3)
            )
            frames.append(frame)
        except Exception:
            break
    return frames


def _build_path_state(path_name, paths_config, defaults, dry_run):
    if dry_run:
        return _StubState(f"PATH::{path_name}")

    cfg = (paths_config.get(path_name, {}) if paths_config else {}) or {}
    reference_frame = cfg.get("reference_frame", f"{path_name}_ref")

    # Waypoint source priority (matches test_follow_waypoints intent —
    # follow what the GUI drew, not a hardcoded count):
    #   1. explicit waypoint_frames in YAML  (full manual override)
    #   2. auto-discover from TF             (whatever the GUI published)
    #   3. YAML 'waypoints: N' count         (last-resort fallback only)
    # Previously (2) was missing, so a YAML 'waypoints: 3' forced wp1..wp3
    # even if the GUI only drew one point -> AUV chased a non-existent
    # path3_wp3 and went to the wrong place.
    waypoint_frames = cfg.get("waypoint_frames")
    if not waypoint_frames:
        max_wp = int(cfg.get("auto_max_waypoints", 20))
        discovered = _autodiscover_waypoints(
            path_name, reference_frame, max_wp
        )
        if discovered:
            rospy.loginfo(
                "[test_pinger_mission] %s: auto-discovered %d waypoint(s) "
                "from TF (GUI-drawn): %s",
                path_name,
                len(discovered),
                discovered,
            )
            waypoint_frames = discovered
    if not waypoint_frames and "waypoints" in cfg:
        count = int(cfg["waypoints"])
        if count < 1:
            raise ValueError(f"'{path_name}.waypoints' must be >= 1")
        waypoint_frames = [f"{path_name}_wp{i + 1}" for i in range(count)]
        rospy.logwarn(
            "[test_pinger_mission] %s: no TF waypoints found, falling back "
            "to YAML 'waypoints: %d' (%s). Draw '%s' in the waypoint GUI "
            "if this is not what you want.",
            path_name,
            count,
            waypoint_frames,
            path_name,
        )
    if not waypoint_frames:
        raise ValueError(
            f"No waypoints for '{path_name}'. Draw it in the waypoint GUI "
            f"(so {path_name}_wp* appear in TF under {reference_frame}), or "
            f"set ~paths.{path_name}.waypoint_frames / waypoints."
        )

    kwargs = dict(defaults)
    kwargs.update({k: cfg[k] for k in _PATH_KWARG_KEYS if k in cfg})

    rospy.loginfo(
        "[test_pinger_mission] %s: ref=%s, wps=%s",
        path_name,
        reference_frame,
        waypoint_frames,
    )
    return DynamicPathExecutionState(
        path_name=path_name,
        reference_frame=reference_frame,
        waypoint_frames=waypoint_frames,
        **kwargs,
    )


def _build_task_state(task_name, tasks_config, dry_run, align_only=False):
    if dry_run:
        return _StubState(f"TASK::{task_name}")
    state_cls = None
    if align_only:
        state_cls = ALIGN_ONLY_STATE_CLASSES.get(task_name)
        if state_cls is not None:
            rospy.loginfo(
                "[test_pinger_mission] %s -> ALIGN-ONLY variant "
                "(aligns then succeeds, task not performed).",
                task_name,
            )
    if state_cls is None:
        state_cls = TASK_STATE_CLASSES.get(task_name)
    if state_cls is None:
        raise KeyError(
            f"No state class for task '{task_name}'. "
            f"Known: {list(TASK_STATE_CLASSES)}"
        )
    params = (tasks_config.get(task_name, {}) or {}).get("params", {}) or {}
    return state_cls(**params)


def _add_simple_token(sm, token, label, next_label,
                      paths_config, tasks_config, defaults, dry_run,
                      align_only=False):
    """Add a non-conditional token (init / path<N> / task) to `sm`."""
    if token == "init":
        state = _StubState("INITIALIZE") if dry_run else InitializeState()
    elif token.startswith("path") and token[4:].isdigit():
        state = _build_path_state(token, paths_config, defaults, dry_run)
    elif token in TASK_STATE_CLASSES:
        state = _build_task_state(token, tasks_config, dry_run, align_only)
    else:
        raise KeyError(
            f"Unknown token '{token}'. Supported: 'init', path<N>, "
            f"tasks {list(TASK_STATE_CLASSES)}, or the pinger token."
        )
    smach.StateMachine.add(
        label,
        state,
        transitions={
            "succeeded": next_label,
            "preempted": "preempted",
            "aborted": "aborted",
        },
    )


def _add_pinger_block(sm, prefix, next_label, pinger_cfg,
                      paths_config, tasks_config, defaults, dry_run,
                      align_only=False):
    """Add the conditional octagon<->torpedo block.

    DECIDE branches to one of two linear chains; both rejoin at `next_label`.
    """
    if not pinger_cfg:
        raise KeyError("Pinger token used but '~pinger_select' is missing.")

    candidates = pinger_cfg.get("candidates")
    transitions = pinger_cfg.get("transitions")
    if not candidates or not transitions:
        raise KeyError(
            "pinger_select must define both 'candidates' and 'transitions'."
        )
    for key in ("octagon_to_torpedo", "torpedo_to_octagon"):
        if key not in transitions:
            raise KeyError(f"pinger_select.transitions missing '{key}'.")

    # The previous token transitions into `prefix`. A path token leaves the
    # align controller locked onto `dynamic_target`; once path following ends
    # that frame stops being broadcast and dies, so
    # reference_pose_publisher's control loop can no longer publish cmd_pose
    # and the next SetDepth (survey/task) freezes. So `prefix` is owned by a
    # CancelAlignControllerState that resets the controller to a plain odom
    # hold first, regardless of what (path / align / nothing) preceded.
    #
    # Pre-survey then runs BEFORE the decision so both object frames exist in
    # the map (PingerDecisionState needs octagon_link / torpedo_target — those
    # only appear once the AUV has seen each object). Surveys are skipped in
    # dry_run (frames come from gazebo_model) and when candidates use
    # gazebo_model (detection-independent).
    cancel_label = prefix
    survey_octagon_label = f"{prefix}_SURVEY_OCTAGON"
    survey_torpedo_label = f"{prefix}_SURVEY_TORPEDO"
    decide_label = f"{prefix}_DECIDE"

    def _uses_detection_survey():
        if dry_run:
            return False
        for cfg in candidates.values():
            if cfg.get("position_source", "tf_frame") == "tf_frame":
                return True
        return False

    survey_enabled = _uses_detection_survey()

    first_after_cancel = survey_octagon_label if survey_enabled else decide_label
    smach.StateMachine.add(
        cancel_label,
        CancelAlignControllerState(),
        transitions={
            "succeeded": first_after_cancel,
            "preempted": "preempted",
            "aborted": "aborted",
        },
    )

    # octagon-first chain: octagon -> <octagon_to_torpedo path> -> torpedo
    oct_path = transitions["octagon_to_torpedo"]
    oct_chain = [
        (f"{prefix}_OCT_1_OCTAGON", "octagon", "task"),
        (f"{prefix}_OCT_2_{oct_path.upper()}", oct_path, "path"),
        (f"{prefix}_OCT_3_TORPEDO", "torpedo", "task"),
    ]
    # torpedo-first chain: torpedo -> <torpedo_to_octagon path> -> octagon
    tor_path = transitions["torpedo_to_octagon"]
    tor_chain = [
        (f"{prefix}_TOR_1_TORPEDO", "torpedo", "task"),
        (f"{prefix}_TOR_2_{tor_path.upper()}", tor_path, "path"),
        (f"{prefix}_TOR_3_OCTAGON", "octagon", "task"),
    ]

    if survey_enabled:
        oct_params = (tasks_config.get("octagon", {}) or {}).get("params", {}) or {}
        tor_params = (tasks_config.get("torpedo", {}) or {}).get("params", {}) or {}
        smach.StateMachine.add(
            survey_octagon_label,
            OctagonSurveyState(**oct_params),
            transitions={
                "succeeded": survey_torpedo_label,
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )
        smach.StateMachine.add(
            survey_torpedo_label,
            TorpedoSurveyState(**tor_params),
            transitions={
                "succeeded": decide_label,
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

    smach.StateMachine.add(
        decide_label,
        PingerDecisionState(
            pinger_frame=pinger_cfg.get("pinger_frame", "pinger_link"),
            candidates=candidates,
            reference_frame=pinger_cfg.get("reference_frame", "odom"),
            wait_timeout=float(pinger_cfg.get("decision_wait_timeout", 30.0)),
            poll_timeout=float(pinger_cfg.get("tf_timeout", 2.0)),
        ),
        transitions={
            "octagon_first": oct_chain[0][0],
            "torpedo_first": tor_chain[0][0],
            "preempted": "preempted",
            "aborted": "aborted",
        },
    )

    for chain in (oct_chain, tor_chain):
        for i, (lbl, tok, kind) in enumerate(chain):
            nxt = chain[i + 1][0] if i + 1 < len(chain) else next_label
            if kind == "task":
                state = _build_task_state(tok, tasks_config, dry_run, align_only)
            else:
                state = _build_path_state(tok, paths_config, defaults, dry_run)
            smach.StateMachine.add(
                lbl,
                state,
                transitions={
                    "succeeded": nxt,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


def main() -> None:
    rospy.init_node("test_pinger_mission")

    raw = rospy.get_param("~test_states", "")
    if isinstance(raw, (list, tuple)):
        tokens = [str(t).strip() for t in raw]
    else:
        tokens = [t.strip() for t in str(raw).split(",")]
    tokens = [t for t in tokens if t]
    if not tokens:
        rospy.logerr("[test_pinger_mission] ~test_states is empty")
        return

    paths_config = rospy.get_param("~paths", None) or {}
    tasks_config = rospy.get_param("~tasks", None) or {}
    pinger_cfg = rospy.get_param("~pinger_select", None) or {}
    pinger_token = rospy.get_param("~pinger_token", "octagon_torpedo_pinger")
    dry_run = bool(rospy.get_param("~dry_run", False))
    align_only = bool(rospy.get_param("~align_only", False))

    defaults = dict(
        final_align=bool(rospy.get_param("~final_align", True)),
        wait_timeout=float(rospy.get_param("~wait_timeout", 120.0)),
    )
    mlv = rospy.get_param("~max_linear_velocity", None)
    mav = rospy.get_param("~max_angular_velocity", None)
    if mlv is not None:
        defaults["max_linear_velocity"] = float(mlv)
    if mav is not None:
        defaults["max_angular_velocity"] = float(mav)

    rospy.loginfo("[test_pinger_mission] Waiting 2s for TF buffer to fill...")
    rospy.sleep(2.0)

    # Build labels up-front so each token knows where it transitions next.
    labels = [f"{i + 1:02d}_{t.upper()}" for i, t in enumerate(tokens)]
    rospy.loginfo(
        "[test_pinger_mission] Token plan%s%s: %s",
        "  [DRY RUN]" if dry_run else "",
        "  [ALIGN-ONLY octagon/torpedo]" if (align_only and not dry_run) else "",
        " -> ".join(tokens),
    )

    sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])
    with sm:
        for i, (token, label) in enumerate(zip(tokens, labels)):
            next_label = labels[i + 1] if i + 1 < len(labels) else "succeeded"
            try:
                if token == pinger_token:
                    _add_pinger_block(
                        sm, label, next_label, pinger_cfg,
                        paths_config, tasks_config, defaults, dry_run,
                        align_only,
                    )
                else:
                    _add_simple_token(
                        sm, token, label, next_label,
                        paths_config, tasks_config, defaults, dry_run,
                        align_only,
                    )
            except (KeyError, ValueError) as exc:
                rospy.logerr(
                    "[test_pinger_mission] Failed to build '%s': %s",
                    token,
                    exc,
                )
                return

    rospy.loginfo("[test_pinger_mission] Executing...")
    outcome = sm.execute()
    rospy.loginfo("[test_pinger_mission] Outcome: %s", outcome)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
