#!/usr/bin/env python3
"""Modular test runner for GUI-drawn multi-path missions.

Mirrors the `test_states:=a,b,c` convention of start.launch:

    roslaunch auv_smach test_follow_waypoints.launch test_states:=path1,path2
    roslaunch auv_smach test_follow_waypoints.launch test_states:=init,path1,path2

Tokens recognised:
    init          -> InitializeState()
    path<N>       -> DynamicPathExecutionState for that path

For each `path<N>` token, the runner looks up its config in rosparam
`~paths.path<N>` (from a YAML passed via `paths_yaml:=...`). The config may
specify:
    waypoints:        integer count  (implies frames path<N>_wp1..path<N>_wpK)
    waypoint_frames:  explicit frame list
    reference_frame:  overrides default `path<N>_ref`
    final_align, final_align_dist_threshold, final_align_yaw_threshold,
    final_align_timeout, final_align_confirm_duration,
    max_linear_velocity, max_angular_velocity, keep_orientation,
    angle_offset, wait_timeout

If no config (or no `waypoints`/`waypoint_frames`) is provided, the runner
auto-discovers `path<N>_wp<M>` frames from TF — requires the GUI to be
publishing them (i.e. the composite reference is live in TF).
"""

import rospy
import smach
import tf2_ros

from auv_smach.waypoints import DynamicPathExecutionState
from auv_smach.initialize import InitializeState


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


_tf_buffer = None
_tf_listener = None


def _get_tf_buffer():
    global _tf_buffer, _tf_listener
    if _tf_buffer is None:
        _tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        _tf_listener = tf2_ros.TransformListener(_tf_buffer)
        rospy.loginfo("[test_follow_waypoints] Waiting 2s for TF buffer to fill...")
        rospy.sleep(2.0)
    return _tf_buffer


def _autodiscover_waypoints(path_name: str, max_waypoints: int, world_frame: str):
    buffer = _get_tf_buffer()
    frames = []
    for i in range(1, max_waypoints + 1):
        frame = f"{path_name}_wp{i}"
        try:
            buffer.lookup_transform(
                world_frame, frame, rospy.Time(0), rospy.Duration(0.3)
            )
            frames.append(frame)
        except Exception:
            break
    return frames


def _build_path_state(path_name, cfg, max_waypoints, world_frame, defaults):
    waypoint_frames = cfg.get("waypoint_frames")
    if not waypoint_frames and "waypoints" in cfg:
        count = int(cfg["waypoints"])
        if count < 1:
            raise ValueError(f"'{path_name}.waypoints' must be >= 1")
        waypoint_frames = [f"{path_name}_wp{i + 1}" for i in range(count)]

    if not waypoint_frames:
        rospy.loginfo(
            f"[test_follow_waypoints] Auto-discovering waypoints for {path_name} "
            f"(no 'waypoints'/'waypoint_frames' provided)..."
        )
        waypoint_frames = _autodiscover_waypoints(path_name, max_waypoints, world_frame)

    if not waypoint_frames:
        raise RuntimeError(
            f"No waypoints for '{path_name}'. Sources tried: ~paths.{path_name}, "
            f"/waypoint_gui/paths.{path_name}, TF scan. "
            f"Fix: (1) draw '{path_name}' in the GUI — if A/B aren't in TF yet, "
            f"enable 'Simulate missing A/B frames' in the GUI; or "
            f"(2) pass paths_yaml:=... with 'waypoints:' count."
        )

    reference_frame = cfg.get("reference_frame", f"{path_name}_ref")

    kwargs = dict(defaults)
    kwargs.update({k: cfg[k] for k in _PATH_KWARG_KEYS if k in cfg})

    rospy.loginfo(
        f"[test_follow_waypoints] {path_name}: ref={reference_frame}, "
        f"wps={waypoint_frames}"
    )

    return DynamicPathExecutionState(
        path_name=path_name,
        reference_frame=reference_frame,
        waypoint_frames=waypoint_frames,
        **kwargs,
    )


def _resolve_state(
    token, paths_config, gui_paths, max_waypoints, world_frame, defaults
):
    if token == "init":
        return InitializeState()
    if token.startswith("path") and token[4:].isdigit():
        cfg = paths_config.get(token, {}) if paths_config else {}
        if not cfg and gui_paths:
            cfg = gui_paths.get(token, {}) or {}
        return _build_path_state(token, cfg, max_waypoints, world_frame, defaults)
    return None


def main() -> None:
    rospy.init_node("test_follow_waypoints")

    raw = rospy.get_param("~test_states", "path1")
    if isinstance(raw, (list, tuple)):
        tokens = [str(t).strip() for t in raw]
    else:
        tokens = [t.strip() for t in str(raw).split(",")]
    tokens = [t for t in tokens if t]

    if not tokens:
        rospy.logerr("[test_follow_waypoints] ~test_states is empty")
        return

    paths_config = rospy.get_param("~paths", None) or {}
    gui_paths_ns = rospy.get_param("~gui_paths_rosparam", "/waypoint_gui/paths")
    gui_paths = rospy.get_param(gui_paths_ns, None) or {}
    if gui_paths and not paths_config:
        rospy.loginfo(
            f"[test_follow_waypoints] Using path configs published by GUI "
            f"at {gui_paths_ns}: {list(gui_paths.keys())}"
        )

    world_frame = rospy.get_param("~world_frame", "odom")
    max_waypoints = int(rospy.get_param("~auto_max_waypoints", 20))

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

    labels = [f"{i + 1:02d}_{t.upper()}" for i, t in enumerate(tokens)]

    sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])
    with sm:
        for i, (token, label) in enumerate(zip(tokens, labels)):
            next_label = labels[i + 1] if i + 1 < len(labels) else "succeeded"
            try:
                state = _resolve_state(
                    token,
                    paths_config,
                    gui_paths,
                    max_waypoints,
                    world_frame,
                    defaults,
                )
            except Exception as exc:  # noqa: BLE001
                rospy.logerr(
                    f"[test_follow_waypoints] Failed to build state for '{token}': {exc}"
                )
                return
            if state is None:
                rospy.logerr(
                    f"[test_follow_waypoints] Unknown token '{token}'. "
                    f"Supported: 'init' or 'path<N>'."
                )
                return
            smach.StateMachine.add(
                label,
                state,
                transitions={
                    "succeeded": next_label,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    rospy.loginfo(f"[test_follow_waypoints] Executing: {' → '.join(tokens)}")
    outcome = sm.execute()
    rospy.loginfo(f"[test_follow_waypoints] Outcome: {outcome}")


if __name__ == "__main__":
    main()
