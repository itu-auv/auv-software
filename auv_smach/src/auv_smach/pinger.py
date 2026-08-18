#!/usr/bin/env python3

import smach
import smach_ros
import rospy
import threading
import time
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

from auv_smach.tf_utils import get_base_link
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SearchForPropState,
    DynamicPathState,
    SetDepthState,
    SetAltitudeState,
)


class PublishPingerWaypointState(smach.State):
    def __init__(self, direction="forward"):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.direction = direction
        self.pub = rospy.Publisher(
            "pinger_waypoint_direction", String, queue_size=1, latch=True
        )

    def execute(self, userdata):
        self.pub.publish(String(data=self.direction))
        rospy.sleep(0.5)
        return "succeeded"


class TogglePingerCollection(smach_ros.ServiceState):
    def __init__(self, data):
        smach_ros.ServiceState.__init__(
            self, "toggle_pinger_collection", SetBool, request=SetBoolRequest(data)
        )


class CollectPingerSamplesState(smach.State):
    def __init__(self, duration, depth_abort_threshold=-0.35):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.duration = float(duration)
        self.depth_abort_threshold = float(depth_abort_threshold)
        self._depth = None
        self._depth_lock = threading.Lock()

    def _odometry_callback(self, msg):
        with self._depth_lock:
            self._depth = float(msg.pose.pose.position.z)

    def _toggle(self, enabled):
        try:
            rospy.loginfo(
                "[Pinger] %s collection service",
                "Starting" if enabled else "Stopping",
            )
            rospy.wait_for_service("toggle_pinger_collection", timeout=5.0)
            toggle = rospy.ServiceProxy("toggle_pinger_collection", SetBool)
            response = toggle(SetBoolRequest(data=enabled))
            rospy.loginfo(
                "[Pinger] Collection service response: success=%s message='%s'",
                response.success,
                response.message,
            )
            return bool(response.success)
        except (rospy.ServiceException, rospy.ROSException) as exc:
            rospy.logerr("[Pinger] Failed to toggle collection: %s", exc)
            return False

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        with self._depth_lock:
            self._depth = None
        rospy.loginfo(
            "[Pinger] COLLECTION enter: duration=%.2fs, depth_abort_threshold=%.3fm",
            self.duration,
            self.depth_abort_threshold,
        )
        odometry_sub = rospy.Subscriber("odometry", Odometry, self._odometry_callback)
        if not self._toggle(True):
            rospy.logerr("[Pinger] COLLECTION could not start; returning aborted")
            odometry_sub.unregister()
            return "aborted"

        start_time = time.monotonic()
        outcome = "succeeded"
        finish_reason = "duration reached"
        rate = rospy.Rate(20)
        rospy.loginfo("[Pinger] COLLECTION started; waiting %.2fs for samples", self.duration)
        try:
            while not rospy.is_shutdown():
                if self.preempt_requested():
                    self.service_preempt()
                    outcome = "preempted"
                    finish_reason = "preempt requested"
                    break

                with self._depth_lock:
                    current_depth = self._depth
                if (
                    current_depth is not None
                    and current_depth >= self.depth_abort_threshold
                ):
                    rospy.logwarn(
                        "[Pinger] Collection cancelled: depth %.3fm crossed upper limit %.3fm",
                        current_depth,
                        self.depth_abort_threshold,
                    )
                    outcome = "aborted"
                    finish_reason = "depth upper limit crossed"
                    break

                elapsed = time.monotonic() - start_time
                rospy.loginfo_throttle(
                    1.0,
                    "[Pinger] COLLECTION waiting: elapsed=%.2f/%.2fs depth=%s",
                    elapsed,
                    self.duration,
                    "unknown" if current_depth is None else f"{current_depth:.3f}m",
                )
                if elapsed >= self.duration:
                    finish_reason = "duration reached"
                    break
                rate.sleep()
        finally:
            stopped = self._toggle(False)
            if not stopped:
                outcome = "aborted"
            odometry_sub.unregister()
            rospy.loginfo(
                "[Pinger] COLLECTION exit: outcome=%s reason=%s stop_service=%s",
                outcome,
                finish_reason,
                stopped,
            )
        return outcome


class ResetPingerData(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "clear_pinger_data", Trigger, request=TriggerRequest()
        )


class ComputePingerPosition(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self, "compute_pinger_position", Trigger, request=TriggerRequest()
        )


class PingerSearchState(smach.StateMachine):
    def __init__(
        self,
        direction="forward",
        waypoint_frame="pinger_waypoint",
        wait_for=20.0,
        collection_altitude=1.0,
        depth_abort_threshold=-0.35,
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])

        with self:
            smach.StateMachine.add(
                "PUBLISH_WAYPOINT",
                PublishPingerWaypointState(direction=direction),
                transitions={
                    "succeeded": "AIM_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "AIM_TO_WAYPOINT",
                SearchForPropState(
                    look_at_frame=waypoint_frame,
                    alignment_frame="waypoint_aim",
                    full_rotation=False,
                    source_frame=get_base_link(),
                    rotation_speed=0.4,
                ),
                transitions={
                    "succeeded": "DYNAMIC_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_TO_WAYPOINT",
                DynamicPathState(
                    plan_target_frame=waypoint_frame,
                    max_linear_velocity=0.4,
                    max_angular_velocity=0.3,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_WAYPOINT",
                AlignFrame(
                    source_frame=get_base_link(),
                    target_frame=waypoint_frame,
                    dist_threshold=0.15,
                    yaw_threshold=0.2,
                    timeout=30.0,
                    confirm_duration=10.0,
                    cancel_on_success=True,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "STABLE_ALIGN_TO_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "SET_COLLECTION_ALTITUDE",
                },
            )

            smach.StateMachine.add(
                "STABLE_ALIGN_TO_WAYPOINT",
                AlignFrame(
                    source_frame=get_base_link(),
                    target_frame=waypoint_frame,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    timeout=30.0,
                    confirm_duration=3.0,
                    cancel_on_success=True,
                    keep_orientation=True,
                    max_linear_velocity=0.05,
                    max_angular_velocity=0.05,
                ),
                transitions={
                    "succeeded": "SET_COLLECTION_ALTITUDE",
                    "preempted": "preempted",
                    "aborted": "SET_COLLECTION_ALTITUDE",
                },
            )

            smach.StateMachine.add(
                "SET_COLLECTION_ALTITUDE",
                SetAltitudeState(altitude=collection_altitude),
                transitions={
                    "succeeded": "CANCEL_CONTROL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_CONTROL",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "START_COLLECTION",
                    "preempted": "preempted",
                    "aborted": "START_COLLECTION",
                },
            )

            smach.StateMachine.add(
                "START_COLLECTION",
                CollectPingerSamplesState(
                    duration=wait_for,
                    depth_abort_threshold=depth_abort_threshold,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class PingerTaskState(smach.State):
    def __init__(
        self,
        pinger_frame="pinger_frame",
        waypoint_frame="pinger_waypoint",
        close_frame="pinger_close_approach",
        align_to_pinger=True,
        collection_altitude=1.0,
        depth_abort_threshold=-0.35,
        collection_duration=20.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.pinger_frame = pinger_frame
        self.waypoint_frame = waypoint_frame
        self.close_frame = close_frame
        self.align_to_pinger = align_to_pinger
        self.collection_altitude = collection_altitude
        self.depth_abort_threshold = depth_abort_threshold
        self.collection_duration = collection_duration

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        look_at_pinger_next = (
            "DYNAMIC_PATH_TO_PINGER" if align_to_pinger else "succeeded"
        )

        with self.sm:
            smach.StateMachine.add(
                "RESET_PINGER_DATA",
                ResetPingerData(),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_PINGER_1",
                PingerSearchState(
                    direction="forward",
                    waypoint_frame=self.waypoint_frame,
                    wait_for=self.collection_duration,
                    collection_altitude=self.collection_altitude,
                    depth_abort_threshold=self.depth_abort_threshold,
                ),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_PINGER_2",
                PingerSearchState(
                    direction="right",
                    waypoint_frame=self.waypoint_frame,
                    wait_for=self.collection_duration,
                    collection_altitude=self.collection_altitude,
                    depth_abort_threshold=self.depth_abort_threshold,
                ),
                transitions={
                    "succeeded": "SEARCH_FOR_PINGER_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_PINGER_3",
                PingerSearchState(
                    direction="backward",
                    waypoint_frame=self.waypoint_frame,
                    wait_for=self.collection_duration,
                    collection_altitude=self.collection_altitude,
                    depth_abort_threshold=self.depth_abort_threshold,
                ),
                transitions={
                    "succeeded": "COMPUTE_POSITION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "COMPUTE_POSITION",
                ComputePingerPosition(),
                transitions={
                    "succeeded": "LOOK_AT_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "LOOK_AT_PINGER",
                SearchForPropState(
                    look_at_frame=self.pinger_frame,
                    alignment_frame="look_at_pinger",
                    full_rotation=False,
                    source_frame=get_base_link(),
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": look_at_pinger_next,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            if align_to_pinger:
                smach.StateMachine.add(
                    "DYNAMIC_PATH_TO_PINGER",
                    DynamicPathState(plan_target_frame=self.close_frame),
                    transitions={
                        "succeeded": "ALIGN_TO_PINGER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "ALIGN_TO_PINGER",
                    AlignFrame(
                        source_frame=get_base_link(),
                        target_frame=self.close_frame,
                        keep_orientation=True,
                        dist_threshold=0.05,
                        yaw_threshold=0.05,
                        timeout=30.0,
                        confirm_duration=3.0,
                        cancel_on_success=True,
                        max_linear_velocity=0.05,
                        max_angular_velocity=0.05,
                    ),
                    transitions={
                        "succeeded": "STABLE_ALIGN_TO_PINGER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "STABLE_ALIGN_TO_PINGER",
                    AlignFrame(
                        source_frame=get_base_link(),
                        target_frame=self.close_frame,
                        keep_orientation=True,
                        dist_threshold=0.03,
                        yaw_threshold=0.03,
                        timeout=30.0,
                        confirm_duration=3.0,
                        cancel_on_success=True,
                        max_linear_velocity=0.03,
                        max_angular_velocity=0.03,
                    ),
                    transitions={
                        "succeeded": "IN_ASSAGI",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "IN_ASSAGI",
                    SetDepthState(
                        depth=-5,
                        confirm_duration=100,
                    ),
                    transitions={
                        "succeeded": "succeeded",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

    def execute(self, userdata):
        rospy.loginfo("Starting Pinger Localisation SMACH Task")
        outcome = self.sm.execute()
        if outcome is None:
            return "preempted"
        return outcome
