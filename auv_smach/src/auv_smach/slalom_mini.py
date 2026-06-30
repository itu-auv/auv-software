import threading

import rospy
import smach
from auv_msgs.msg import SlalomProp, SlalomTarget
from auv_smach.common import SetDepthState
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest


class RunSlalomMiniControllerState(smach.State):
    def __init__(
        self,
        controller_depth: float = -1.1,
        forward_velocity: float = 0.25,
        yaw_gain: float = 1.0,
        max_yaw_step_rad: float = 0.0,
        target_timeout_s: float = 0.5,
        lateral_strategy: str = "none",
        lateral_gain: float = 0.0,
        max_lateral_velocity: float = 0.15,
        min_runtime_s: float = 5.0,
        target_lost_success_timeout_s: float = 1.5,
        target_acquire_timeout_s: float = 8.0,
        max_runtime_s: float = 25.0,
        min_travel_distance_m: float = 4.0,
        premature_target_lost_abort_timeout_s: float = 3.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.controller_depth = controller_depth
        self.forward_velocity = forward_velocity
        self.yaw_gain = yaw_gain
        self.max_yaw_step_rad = max_yaw_step_rad
        self.target_timeout_s = target_timeout_s
        self.lateral_strategy = lateral_strategy
        self.lateral_gain = lateral_gain
        self.max_lateral_velocity = max_lateral_velocity
        self.min_runtime_s = min_runtime_s
        self.target_lost_success_timeout_s = target_lost_success_timeout_s
        self.target_acquire_timeout_s = target_acquire_timeout_s
        self.max_runtime_s = max_runtime_s
        self.min_travel_distance_m = min_travel_distance_m
        self.premature_target_lost_abort_timeout_s = (
            premature_target_lost_abort_timeout_s
        )

        self.lock = threading.Lock()
        self.last_target_time = None
        self.last_prop_time = None
        self.seen_target_count = 0
        self.seen_prop_count = 0
        self.latest_odom = None
        self.start_odom_xy = None
        self.target_sub = rospy.Subscriber(
            "slalom/target", SlalomTarget, self.target_callback, queue_size=1
        )
        self.prop_sub = rospy.Subscriber(
            "slalom/props", SlalomProp, self.prop_callback, queue_size=10
        )
        self.odom_sub = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, queue_size=1
        )

    def target_callback(self, msg: SlalomTarget):
        with self.lock:
            self.last_target_time = rospy.Time.now()
            self.seen_target_count += 1

    def prop_callback(self, msg: SlalomProp):
        with self.lock:
            self.last_prop_time = rospy.Time.now()
            self.seen_prop_count += 1

    def odom_callback(self, msg: Odometry):
        with self.lock:
            self.latest_odom = msg

    def _reset_target_state(self):
        with self.lock:
            self.last_target_time = None
            self.last_prop_time = None
            self.seen_target_count = 0
            self.seen_prop_count = 0
            self.start_odom_xy = None

    def _get_state(self):
        with self.lock:
            return (
                self.last_target_time,
                self.last_prop_time,
                self.seen_target_count,
                self.seen_prop_count,
                self.latest_odom,
            )

    def _capture_start_odom_if_ready(self):
        with self.lock:
            if self.start_odom_xy is not None or self.latest_odom is None:
                return
            pos = self.latest_odom.pose.pose.position
            self.start_odom_xy = (pos.x, pos.y)

    def _travel_distance(self, odom: Odometry):
        with self.lock:
            if self.start_odom_xy is None:
                return 0.0
            start_x, start_y = self.start_odom_xy
        pos = odom.pose.pose.position
        return ((pos.x - start_x) ** 2 + (pos.y - start_y) ** 2) ** 0.5

    def _set_detection_enabled(self, enabled: bool):
        rospy.wait_for_service("enable_slalom_camera_detections", timeout=5.0)
        set_detection = rospy.ServiceProxy("enable_slalom_camera_detections", SetBool)
        response = set_detection(SetBoolRequest(data=enabled))
        if not response.success:
            raise rospy.ServiceException(response.message)

    def _set_controller_enabled(self, enabled: bool):
        rospy.wait_for_service("slalom_controller/set_enabled", timeout=5.0)
        set_enabled = rospy.ServiceProxy("slalom_controller/set_enabled", SetBool)
        response = set_enabled(SetBoolRequest(data=enabled))
        if not response.success:
            raise rospy.ServiceException(response.message)

    def _configure_controller(self):
        controller_params = {
            "depth": self.controller_depth,
            "forward_velocity": self.forward_velocity,
            "yaw_gain": self.yaw_gain,
            "max_yaw_step_rad": self.max_yaw_step_rad,
            "target_timeout_s": self.target_timeout_s,
            "lateral_strategy": self.lateral_strategy,
            "lateral_gain": self.lateral_gain,
            "max_lateral_velocity": self.max_lateral_velocity,
        }

        for name, value in controller_params.items():
            rospy.set_param(f"slalom_mini_controller/{name}", value)

        rospy.loginfo(
            "[SlalomMini] Controller params: depth=%.2f vx=%.2f yaw_gain=%.2f "
            "max_yaw_step=%.2f target_timeout=%.2f lateral_strategy=%s "
            "lateral_gain=%.2f max_vy=%.2f",
            self.controller_depth,
            self.forward_velocity,
            self.yaw_gain,
            self.max_yaw_step_rad,
            self.target_timeout_s,
            self.lateral_strategy,
            self.lateral_gain,
            self.max_lateral_velocity,
        )

    def _stop_controller(self):
        try:
            rospy.wait_for_service("slalom_controller/stop", timeout=1.0)
            stop_controller = rospy.ServiceProxy("slalom_controller/stop", Trigger)
            stop_controller(TriggerRequest())
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"[SlalomMini] Failed to stop controller cleanly: {e}")

    def _cleanup(self):
        self._stop_controller()
        try:
            self._set_detection_enabled(False)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"[SlalomMini] Failed to disable slalom detection: {e}")

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        self._reset_target_state()

        try:
            self._configure_controller()
            self._set_detection_enabled(True)
            self._set_controller_enabled(True)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"[SlalomMini] Failed to start mini slalom control: {e}")
            self._cleanup()
            return "aborted"

        start_time = rospy.Time.now()
        rate = rospy.Rate(20)

        rospy.loginfo(
            "[SlalomMini] Running mini slalom controller. "
            "min_runtime=%.1fs lost_success=%.1fs acquire_timeout=%.1fs "
            "max_runtime=%.1fs min_travel=%.2fm",
            self.min_runtime_s,
            self.target_lost_success_timeout_s,
            self.target_acquire_timeout_s,
            self.max_runtime_s,
            self.min_travel_distance_m,
        )

        while not rospy.is_shutdown():
            if self.preempt_requested():
                rospy.logwarn("[SlalomMini] Preempt requested.")
                self._cleanup()
                self.service_preempt()
                return "preempted"

            now = rospy.Time.now()
            elapsed = (now - start_time).to_sec()
            self._capture_start_odom_if_ready()
            (
                last_target_time,
                last_prop_time,
                seen_target_count,
                seen_prop_count,
                odom,
            ) = self._get_state()
            traveled = self._travel_distance(odom) if odom is not None else 0.0

            if seen_target_count == 0:
                rospy.loginfo_throttle(
                    1.0, "[SlalomMini] Waiting for first slalom target."
                )
                if elapsed > self.target_acquire_timeout_s:
                    rospy.logerr(
                        "[SlalomMini] No slalom target received within %.1fs.",
                        self.target_acquire_timeout_s,
                    )
                    self._cleanup()
                    return "aborted"
            else:
                target_age = (now - last_target_time).to_sec()
                prop_age = (
                    (now - last_prop_time).to_sec()
                    if last_prop_time is not None
                    else float("inf")
                )
                rospy.loginfo_throttle(
                    1.0,
                    "[SlalomMini] elapsed=%.1fs target_age=%.2fs prop_age=%.2fs "
                    "targets=%d props=%d traveled=%.2fm",
                    elapsed,
                    target_age,
                    prop_age,
                    seen_target_count,
                    seen_prop_count,
                    traveled,
                )

                if (
                    elapsed >= self.min_runtime_s
                    and traveled >= self.min_travel_distance_m
                    and prop_age >= self.target_lost_success_timeout_s
                ):
                    rospy.loginfo(
                        "[SlalomMini] Slalom detections lost after minimum run time and travel distance. Slalom succeeded."
                    )
                    self._cleanup()
                    return "succeeded"

                if (
                    traveled < self.min_travel_distance_m
                    and prop_age >= self.premature_target_lost_abort_timeout_s
                ):
                    rospy.logerr(
                        "[SlalomMini] Slalom detections lost before minimum travel distance "
                        "(%.2fm < %.2fm). Aborting.",
                        traveled,
                        self.min_travel_distance_m,
                    )
                    self._cleanup()
                    return "aborted"

            if self.max_runtime_s > 0.0 and elapsed >= self.max_runtime_s:
                rospy.loginfo(
                    "[SlalomMini] Max runtime reached. Marking slalom as succeeded."
                )
                self._cleanup()
                return "succeeded"

            rate.sleep()

        self._cleanup()
        return "aborted"


class NavigateThroughSlalomMiniState(smach.State):
    def __init__(
        self,
        slalom_depth: float,
        forward_velocity: float = 0.25,
        yaw_gain: float = 1.0,
        max_yaw_step_rad: float = 0.0,
        target_timeout_s: float = 0.5,
        lateral_strategy: str = "none",
        lateral_gain: float = 0.0,
        max_lateral_velocity: float = 0.15,
        min_runtime_s: float = 5.0,
        target_lost_success_timeout_s: float = 1.5,
        target_acquire_timeout_s: float = 8.0,
        max_runtime_s: float = 25.0,
        min_travel_distance_m: float = 4.0,
        premature_target_lost_abort_timeout_s: float = 3.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with self.sm:
            smach.StateMachine.add(
                "SET_SLALOM_DEPTH",
                SetDepthState(depth=slalom_depth),
                transitions={
                    "succeeded": "RUN_SLALOM_MINI_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "RUN_SLALOM_MINI_CONTROLLER",
                RunSlalomMiniControllerState(
                    controller_depth=slalom_depth,
                    forward_velocity=forward_velocity,
                    yaw_gain=yaw_gain,
                    max_yaw_step_rad=max_yaw_step_rad,
                    target_timeout_s=target_timeout_s,
                    lateral_strategy=lateral_strategy,
                    lateral_gain=lateral_gain,
                    max_lateral_velocity=max_lateral_velocity,
                    min_runtime_s=min_runtime_s,
                    target_lost_success_timeout_s=target_lost_success_timeout_s,
                    target_acquire_timeout_s=target_acquire_timeout_s,
                    max_runtime_s=max_runtime_s,
                    min_travel_distance_m=min_travel_distance_m,
                    premature_target_lost_abort_timeout_s=premature_target_lost_abort_timeout_s,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        return self.sm.execute()
