import smach
import rospy

from auv_smach.common import (
    AlignFrame,
    CheckForTransformState,
    DynamicPathState,
)
from auv_smach.tf_utils import get_base_link


class FollowWaypointsState(smach.StateMachine):
    """Sequentially navigates through a list of TF frames using DynamicPathState.

    Designed as a drop-in sub-state for other state machines: construct with the
    ordered list of target frames (e.g. from the waypoint GUI) and add it with
    `smach.StateMachine.add(...)` like any other state.
    """

    def __init__(
        self,
        waypoint_frames,
        source_frame: str = None,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        keep_orientation: bool = False,
        angle_offset: float = 0.0,
        final_align: bool = False,
        final_align_dist_threshold: float = 0.15,
        final_align_yaw_threshold: float = 0.1,
        final_align_timeout: float = 15.0,
        final_align_confirm_duration: float = 1.0,
        state_prefix: str = "",
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        if not waypoint_frames:
            raise ValueError(
                "FollowWaypointsState requires at least one waypoint frame"
            )

        if source_frame is None:
            source_frame = get_base_link()

        self._waypoint_frames = list(waypoint_frames)

        def _tag(name):
            return f"{state_prefix}{name}" if state_prefix else name

        with self:
            for idx, frame in enumerate(self._waypoint_frames):
                is_last = idx == len(self._waypoint_frames) - 1
                path_state_name = _tag(f"PATH_TO_{idx + 1}_{frame.upper()}")

                if is_last and final_align:
                    next_on_success = _tag(f"ALIGN_AT_{idx + 1}_{frame.upper()}")
                elif is_last:
                    next_on_success = "succeeded"
                else:
                    next_frame = self._waypoint_frames[idx + 1]
                    next_on_success = _tag(f"PATH_TO_{idx + 2}_{next_frame.upper()}")

                smach.StateMachine.add(
                    path_state_name,
                    DynamicPathState(
                        plan_target_frame=frame,
                        align_source_frame=source_frame,
                        max_linear_velocity=max_linear_velocity,
                        max_angular_velocity=max_angular_velocity,
                        angle_offset=angle_offset,
                        keep_orientation=keep_orientation,
                    ),
                    transitions={
                        "succeeded": next_on_success,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                if is_last and final_align:
                    smach.StateMachine.add(
                        _tag(f"ALIGN_AT_{idx + 1}_{frame.upper()}"),
                        AlignFrame(
                            source_frame=source_frame,
                            target_frame=frame,
                            angle_offset=angle_offset,
                            dist_threshold=final_align_dist_threshold,
                            yaw_threshold=final_align_yaw_threshold,
                            timeout=final_align_timeout,
                            confirm_duration=final_align_confirm_duration,
                            cancel_on_success=False,
                            keep_orientation=keep_orientation,
                            max_linear_velocity=max_linear_velocity,
                            max_angular_velocity=max_angular_velocity,
                        ),
                        transitions={
                            "succeeded": "succeeded",
                            "preempted": "preempted",
                            "aborted": "aborted",
                        },
                    )

    @classmethod
    def from_prefix(
        cls,
        prefix: str,
        count: int,
        start_index: int = 1,
        **kwargs,
    ) -> "FollowWaypointsState":
        """Builds `<prefix><start_index>` ... `<prefix><start_index + count - 1>`."""
        if count < 1:
            raise ValueError("count must be >= 1")
        frames = [f"{prefix}{i}" for i in range(start_index, start_index + count)]
        return cls(waypoint_frames=frames, **kwargs)

    @classmethod
    def from_rosparam(
        cls,
        param_name: str,
        **kwargs,
    ) -> "FollowWaypointsState":
        frames = rospy.get_param(param_name)
        if not isinstance(frames, (list, tuple)) or not frames:
            raise ValueError(
                f"ROS param '{param_name}' must be a non-empty list of frame names"
            )
        return cls(waypoint_frames=list(frames), **kwargs)


class DynamicPathExecutionState(smach.StateMachine):
    """Waits for a path's reference + waypoint frames to become visible, then follows them.

    Intended for GUI-drawn paths whose reference frame is an object (e.g. octagon_link)
    or a composite frame (e.g. path2_ref): this state stays in WAIT until all the
    required frames are live in TF, so the path can be scheduled at any point in the
    mission and will only execute once its preconditions are met.

    Parameters
    ----------
    path_name:
        Logical name of the path (e.g. "path1"). Used only for state labelling.
    reference_frame:
        Frame that the waypoints are expressed in. Waited on before execution.
    waypoint_frames:
        Ordered list of waypoint TF frame names for this path.
    wait_timeout:
        How long to wait for each required frame before aborting.
    source_frame:
        AUV pose frame (default: `get_base_link()`).
    final_align / final_align_*:
        Forwarded to `FollowWaypointsState` for tight final-pose holding.
    max_linear_velocity / max_angular_velocity / keep_orientation / angle_offset:
        Forwarded to each `DynamicPathState` inside the follower.
    """

    def __init__(
        self,
        path_name: str,
        reference_frame: str,
        waypoint_frames,
        wait_timeout: float = 120.0,
        source_frame: str = None,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        keep_orientation: bool = False,
        angle_offset: float = 0.0,
        final_align: bool = False,
        final_align_dist_threshold: float = 0.15,
        final_align_yaw_threshold: float = 0.1,
        final_align_timeout: float = 15.0,
        final_align_confirm_duration: float = 1.0,
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        if not waypoint_frames:
            raise ValueError(
                "DynamicPathExecutionState requires at least one waypoint frame"
            )
        if not reference_frame:
            raise ValueError("reference_frame must be non-empty")

        if source_frame is None:
            source_frame = get_base_link()

        tag = path_name.upper()
        follow_state_name = f"FOLLOW_{tag}"

        with self:
            smach.StateMachine.add(
                f"WAIT_FOR_{tag}_REF",
                CheckForTransformState(
                    source_frame=source_frame,
                    target_frame=reference_frame,
                    timeout=wait_timeout,
                ),
                transitions={
                    "succeeded": f"WAIT_FOR_{tag}_WP1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            first_wp = waypoint_frames[0]
            next_after_wp1 = follow_state_name
            smach.StateMachine.add(
                f"WAIT_FOR_{tag}_WP1",
                CheckForTransformState(
                    source_frame=source_frame,
                    target_frame=first_wp,
                    timeout=wait_timeout,
                ),
                transitions={
                    "succeeded": next_after_wp1,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                follow_state_name,
                FollowWaypointsState(
                    waypoint_frames=list(waypoint_frames),
                    source_frame=source_frame,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    keep_orientation=keep_orientation,
                    angle_offset=angle_offset,
                    final_align=final_align,
                    final_align_dist_threshold=final_align_dist_threshold,
                    final_align_yaw_threshold=final_align_yaw_threshold,
                    final_align_timeout=final_align_timeout,
                    final_align_confirm_duration=final_align_confirm_duration,
                    state_prefix=f"{tag}_",
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class MultiPathFollowState(smach.StateMachine):
    """Runs a list of `DynamicPathExecutionState`s sequentially.

    `paths` is a list of dicts, each specifying one path. Supported keys:
      name              (required) logical name, e.g. "path1"
      reference_frame   (required) TF frame the path is relative to
      waypoint_frames   (required) ordered list of waypoint TF frame names
      final_align       optional, default False
      ... plus any keyword understood by DynamicPathExecutionState.

    Shared overrides (max_linear_velocity, max_angular_velocity, keep_orientation,
    angle_offset, wait_timeout, source_frame) act as defaults for every path and
    can be overridden per-path.
    """

    def __init__(
        self,
        paths,
        source_frame: str = None,
        wait_timeout: float = 120.0,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
        keep_orientation: bool = False,
        angle_offset: float = 0.0,
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        if not paths:
            raise ValueError("MultiPathFollowState requires at least one path")

        if source_frame is None:
            source_frame = get_base_link()

        shared_defaults = dict(
            source_frame=source_frame,
            wait_timeout=wait_timeout,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            keep_orientation=keep_orientation,
            angle_offset=angle_offset,
        )

        with self:
            for i, spec in enumerate(paths):
                is_last = i == len(paths) - 1
                spec = dict(spec)
                name = spec.pop("name")
                reference_frame = spec.pop("reference_frame")
                waypoint_frames = spec.pop("waypoint_frames")

                kwargs = {**shared_defaults, **spec}

                state_label = f"RUN_{name.upper()}"
                next_on_success = (
                    "succeeded" if is_last else f"RUN_{paths[i + 1]['name'].upper()}"
                )

                smach.StateMachine.add(
                    state_label,
                    DynamicPathExecutionState(
                        path_name=name,
                        reference_frame=reference_frame,
                        waypoint_frames=waypoint_frames,
                        **kwargs,
                    ),
                    transitions={
                        "succeeded": next_on_success,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
