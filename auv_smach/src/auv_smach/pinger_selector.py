"""Pinger-based task ordering.

A single pinger mock (or, in the real arena, the localized pinger) publishes
one TF frame that sits on top of whichever task object it was deployed near.
This module answers exactly one question:

    "Of the candidate tasks, which one is the pinger closest to?"

That answer drives the mission order: the near task runs first, the AUV
follows a GUI-drawn transition path to the far task, then the far task runs.

The pinger position is always read from TF (the mock publishes it there).
The candidate task positions can come from one of two sources, chosen per
candidate via `position_source`:

  * "tf_frame"     — look up a TF frame (e.g. octagon_link, torpedo_target).
                     Matches the real arena, but the frame may not exist
                     until the object is detected.
  * "gazebo_model" — read a Gazebo model pose from /gazebo/model_states
                     (e.g. robosub_octagon). Sim-only, but always available
                     and detection-independent. Handy for dry_run.

The module holds no mission/SMACH logic — only position math.
"""

import math
import threading

import rospy

from auv_smach.tf_utils import get_tf_buffer


class PingerSelectionError(Exception):
    """Raised when the pinger or candidate positions cannot be resolved."""


class _GazeboModelPoses:
    """Lazily subscribes to /gazebo/model_states and caches the latest poses."""

    _instance = None

    def __init__(self):
        from gazebo_msgs.msg import ModelStates  # sim-only dependency

        self._lock = threading.Lock()
        self._positions = {}  # model_name -> (x, y, z)
        self._got_msg = False
        self._sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self._cb,
            queue_size=1,
        )

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _cb(self, msg):
        names = list(msg.name) if msg.name else []
        poses = list(msg.pose) if msg.pose else []
        positions = {}
        for i, name in enumerate(names):
            if i < len(poses):
                p = poses[i].position
                positions[name] = (p.x, p.y, p.z)
        with self._lock:
            self._positions = positions
            self._got_msg = True

    def wait_for_first(self, timeout):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self._lock:
                if self._got_msg:
                    return
            if rospy.Time.now() >= deadline:
                raise PingerSelectionError(
                    "No /gazebo/model_states received within "
                    f"{timeout}s (is Gazebo running?)"
                )
            rate.sleep()

    def get(self, model_name):
        with self._lock:
            if model_name not in self._positions:
                raise PingerSelectionError(
                    f"Gazebo model '{model_name}' not in /gazebo/model_states. "
                    f"Known: {sorted(self._positions)}"
                )
            return self._positions[model_name]


def _pinger_position(pinger_frame, reference_frame, timeout):
    """Pinger position in `reference_frame`, read from TF."""
    buffer = get_tf_buffer()
    try:
        tf = buffer.lookup_transform(
            reference_frame,
            pinger_frame,
            rospy.Time(0),
            rospy.Duration(timeout),
        )
    except Exception as exc:
        raise PingerSelectionError(
            f"TF lookup failed {reference_frame} <- {pinger_frame}: {exc}"
        )
    t = tf.transform.translation
    return (t.x, t.y, t.z)


def _candidate_position(name, cfg, reference_frame, timeout):
    """Resolve one candidate's position via its configured source."""
    source = cfg.get("position_source", "tf_frame")

    if source == "tf_frame":
        frame = cfg.get("decision_frame")
        if not frame:
            raise PingerSelectionError(
                f"Candidate '{name}' uses position_source=tf_frame but has "
                f"no 'decision_frame'."
            )
        buffer = get_tf_buffer()
        try:
            tf = buffer.lookup_transform(
                reference_frame,
                frame,
                rospy.Time(0),
                rospy.Duration(timeout),
            )
        except Exception as exc:
            raise PingerSelectionError(
                f"TF lookup failed {reference_frame} <- {frame} "
                f"(candidate '{name}'): {exc}"
            )
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    if source == "gazebo_model":
        model_name = cfg.get("gazebo_model")
        if not model_name:
            raise PingerSelectionError(
                f"Candidate '{name}' uses position_source=gazebo_model but "
                f"has no 'gazebo_model'."
            )
        gz = _GazeboModelPoses.instance()
        gz.wait_for_first(timeout)
        # NOTE: Gazebo poses are in the Gazebo world frame, and the pinger TF
        # position is in `reference_frame`. We only compare *which* candidate
        # is nearer, and the pinger sits directly on top of one of these same
        # models, so relative ordering is preserved as long as world≈odom in
        # sim (true for the standard sim setup). Good enough for dry_run /
        # ordering validation; physical tests use position_source=tf_frame.
        return gz.get(model_name)

    raise PingerSelectionError(
        f"Candidate '{name}' has unknown position_source='{source}' "
        f"(expected 'tf_frame' or 'gazebo_model')."
    )


def _dist(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def select_nearest_task(
    pinger_frame,
    candidates,
    reference_frame="odom",
    timeout=5.0,
):
    """Return (near_name, far_name, distances) for two candidate tasks.

    Parameters
    ----------
    pinger_frame:
        TF frame the pinger mock publishes (e.g. "pinger_link").
    candidates:
        Mapping of task name -> candidate config dict. Each config selects a
        position source:
          {"position_source": "tf_frame",     "decision_frame": "octagon_link"}
          {"position_source": "gazebo_model", "gazebo_model": "robosub_octagon"}
        If "position_source" is omitted it defaults to "tf_frame".
        Exactly two candidates are expected.
    reference_frame:
        Frame the pinger (and tf_frame candidates) are expressed in.
    timeout:
        Per-lookup wait, in seconds.

    Returns
    -------
    (near_name, far_name, distances) where distances is {name: meters}.
    """
    if len(candidates) != 2:
        raise PingerSelectionError(
            f"Expected exactly 2 candidates, got {len(candidates)}: "
            f"{list(candidates)}"
        )

    pinger_pos = _pinger_position(pinger_frame, reference_frame, timeout)

    distances = {}
    for name, cfg in candidates.items():
        cand_pos = _candidate_position(name, cfg, reference_frame, timeout)
        distances[name] = _dist(pinger_pos, cand_pos)

    near_name = min(distances, key=distances.get)
    far_name = max(distances, key=distances.get)

    rospy.loginfo(
        "[pinger_selector] Pinger '%s' distances: %s -> nearest=%s, farthest=%s",
        pinger_frame,
        ", ".join(f"{n}={d:.3f}m" for n, d in distances.items()),
        near_name,
        far_name,
    )

    return near_name, far_name, distances
