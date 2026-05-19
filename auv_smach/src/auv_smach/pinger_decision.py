"""PingerDecisionState — decides octagon/torpedo order *when its turn comes*.

Unlike a startup-time decision, this is a real SMACH state placed in the
sequence after the earlier tasks (gate, bin, ...). By the time the state
machine reaches it, the AUV has already moved around and the object frames
(octagon_link / torpedo_target) have normally been published. The state
waits for those frames up to a timeout (the AUV may still be settling),
then compares pinger distance and reports which task should run first.

Outcomes:
    octagon_first  — pinger nearer octagon  -> run octagon, then torpedo
    torpedo_first  — pinger nearer torpedo  -> run torpedo, then octagon
    aborted        — required frame never appeared within the timeout
    preempted

It also writes `userdata.pinger_order = (near, far)` for callers that prefer
reading it from userdata instead of branching on the outcome.

Decision frames like octagon_link / torpedo_target are produced by the
per-object target publishers, which only start broadcasting once a SetBool
"enable" service is called True. Normally the octagon/torpedo *task* makes
that call — but the pinger decision runs *before* those tasks, so the frames
would not exist yet. If a candidate config provides `enable_service`, this
state calls it True once before polling, so the target frame is published in
time for the distance comparison (the underlying object must already be in
the map, i.e. the AUV has seen it).
"""

import rospy
import smach

from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.pinger_selector import select_nearest_task, PingerSelectionError


class PingerDecisionState(smach.State):
    def __init__(
        self,
        pinger_frame,
        candidates,
        reference_frame="odom",
        wait_timeout=30.0,
        poll_timeout=2.0,
    ):
        smach.State.__init__(
            self,
            outcomes=[
                "octagon_first",
                "torpedo_first",
                "preempted",
                "aborted",
            ],
            output_keys=["pinger_order"],
        )
        self._pinger_frame = pinger_frame
        self._candidates = candidates
        self._reference_frame = reference_frame
        self._wait_timeout = float(wait_timeout)
        self._poll_timeout = float(poll_timeout)

        names = sorted(candidates.keys())
        if names != ["octagon", "torpedo"]:
            raise ValueError(
                "PingerDecisionState expects exactly candidates "
                f"'octagon' and 'torpedo', got {names}"
            )

    def _enable_target_frames(self):
        """Call each candidate's `enable_service` (SetBool, data=True) once.

        Best-effort: a missing/failing service is logged but not fatal — the
        frame may still be published by another path, and the polling loop
        below will time out and abort cleanly if it never appears.
        """
        for name, cfg in self._candidates.items():
            service_name = cfg.get("enable_service")
            if not service_name:
                continue
            try:
                rospy.wait_for_service(service_name, timeout=self._poll_timeout)
                proxy = rospy.ServiceProxy(service_name, SetBool)
                resp = proxy(SetBoolRequest(data=True))
                rospy.loginfo(
                    "[PingerDecisionState] Enabled '%s' target frame via %s "
                    "(success=%s)",
                    name,
                    service_name,
                    getattr(resp, "success", "?"),
                )
            except Exception as exc:
                rospy.logwarn(
                    "[PingerDecisionState] Could not call enable_service '%s' "
                    "for candidate '%s' (continuing anyway): %s",
                    service_name,
                    name,
                    exc,
                )

    def execute(self, userdata):
        rospy.loginfo(
            "[PingerDecisionState] Deciding order. Waiting up to %.0fs for "
            "candidate positions to become resolvable...",
            self._wait_timeout,
        )

        self._enable_target_frames()

        deadline = rospy.Time.now() + rospy.Duration(self._wait_timeout)
        rate = rospy.Rate(2)
        last_err = None

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                rospy.loginfo("[PingerDecisionState] Preempted")
                return "preempted"

            try:
                near, far, distances = select_nearest_task(
                    pinger_frame=self._pinger_frame,
                    candidates=self._candidates,
                    reference_frame=self._reference_frame,
                    timeout=self._poll_timeout,
                )
                userdata.pinger_order = (near, far)
                rospy.loginfo(
                    "[PingerDecisionState] Decision: %s first "
                    "(distances: %s)",
                    near,
                    ", ".join(f"{n}={d:.2f}m" for n, d in distances.items()),
                )
                return "octagon_first" if near == "octagon" else "torpedo_first"

            except PingerSelectionError as exc:
                last_err = exc
                if rospy.Time.now() >= deadline:
                    rospy.logerr(
                        "[PingerDecisionState] Timed out after %.0fs waiting "
                        "for pinger/candidate positions: %s",
                        self._wait_timeout,
                        exc,
                    )
                    return "aborted"
                rospy.loginfo_throttle(
                    5.0,
                    "[PingerDecisionState] Not resolvable yet, retrying: %s",
                    exc,
                )
                rate.sleep()

        rospy.logerr(
            "[PingerDecisionState] Shutdown while deciding (last error: %s)",
            last_err,
        )
        return "aborted"
