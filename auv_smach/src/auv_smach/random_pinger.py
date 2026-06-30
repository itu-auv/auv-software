#!/usr/bin/env python3

import math
import threading

import rospy
import smach
import tf2_ros
from std_msgs.msg import Float32

from auv_common_lib.transform import lookup_fresh_transform
from auv_smach.common import CancelAlignControllerState
from auv_smach.initialize import DelayState
from auv_smach.octagon import OctagonTaskState
from auv_smach.tf_utils import get_base_link, get_tf_buffer
from auv_smach.torpedo import TorpedoTaskState


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def angular_distance(angle_a, angle_b):
    return abs(normalize_angle(angle_a - angle_b))


def circular_mean(angles):
    sin_sum = sum(math.sin(angle) for angle in angles)
    cos_sum = sum(math.cos(angle) for angle in angles)
    return math.atan2(sin_sum, cos_sum)


def densest_angle_cluster(angles, max_distance):
    best_cluster = []
    best_spread = float("inf")

    for center in angles:
        cluster = [
            angle for angle in angles if angular_distance(angle, center) <= max_distance
        ]
        cluster_mean = circular_mean(cluster)
        spread = sum(angular_distance(angle, cluster_mean) for angle in cluster)

        if len(cluster) > len(best_cluster) or (
            len(cluster) == len(best_cluster) and spread < best_spread
        ):
            best_cluster = cluster
            best_spread = spread

    return best_cluster


class RandomPingerSelectorState(smach.State):
    def __init__(
        self,
        angle_topic,
        torpedo_frame,
        octagon_frame,
        source_frame=None,
        sample_count=5,
        timeout=10.0,
        fallback_first="torpedo",
        tie_threshold=math.radians(5.0),
        outlier_rejection_threshold=math.radians(15.0),
        min_consensus_ratio=0.6,
        tf_lookup_timeout=2.0,
        tf_freshness_threshold=0.0,
    ):
        smach.State.__init__(
            self,
            outcomes=["torpedo_first", "octagon_first", "preempted", "aborted"],
        )
        self.angle_topic = angle_topic
        self.torpedo_frame = torpedo_frame
        self.octagon_frame = octagon_frame
        self.source_frame = source_frame or get_base_link()
        self.sample_count = max(1, int(sample_count))
        self.timeout = float(timeout)
        self.fallback_first = fallback_first
        self.tie_threshold = float(tie_threshold)
        self.outlier_rejection_threshold = max(0.0, float(outlier_rejection_threshold))
        self.min_consensus_ratio = min(1.0, max(0.0, float(min_consensus_ratio)))
        self.tf_lookup_timeout = rospy.Duration(float(tf_lookup_timeout))
        self.tf_freshness_threshold = rospy.Duration(float(tf_freshness_threshold))

        self.tf_buffer = get_tf_buffer()
        self._samples = []
        self._samples_lock = threading.Lock()
        self._subscriber = rospy.Subscriber(
            self.angle_topic,
            Float32,
            self._angle_callback,
            queue_size=10,
        )

    def _angle_callback(self, msg):
        angle = msg.data
        if not math.isfinite(angle):
            rospy.logwarn_throttle(
                5.0, f"[RandomPingerSelector] Ignoring non-finite angle: {angle}"
            )
            return

        with self._samples_lock:
            self._samples.append(normalize_angle(angle))
            max_buffer_size = max(20, self.sample_count * 2)
            if len(self._samples) > max_buffer_size:
                self._samples = self._samples[-max_buffer_size:]

    def _clear_samples(self):
        with self._samples_lock:
            self._samples = []

    def _get_samples(self):
        with self._samples_lock:
            return list(self._samples)

    def _filter_outlier_samples(self, samples):
        if len(samples) < 3 or self.outlier_rejection_threshold <= 0.0:
            return samples

        cluster = densest_angle_cluster(samples, self.outlier_rejection_threshold)
        min_cluster_size = max(
            1, int(math.ceil(len(samples) * self.min_consensus_ratio))
        )

        if len(cluster) < min_cluster_size:
            rospy.logwarn(
                "[RandomPingerSelector] No stable pinger angle cluster: best "
                "%d/%d sample(s) within %.3f rad, need %d.",
                len(cluster),
                len(samples),
                self.outlier_rejection_threshold,
                min_cluster_size,
            )
            return []

        rejected_count = len(samples) - len(cluster)
        if rejected_count > 0:
            rospy.logwarn(
                "[RandomPingerSelector] Rejected %d/%d pinger outlier sample(s). "
                "Using %d-sample consensus cluster.",
                rejected_count,
                len(samples),
                len(cluster),
            )

        return cluster

    def _fallback_outcome(self, reason):
        if self.fallback_first == "torpedo":
            rospy.logwarn(
                f"[RandomPingerSelector] {reason}. Falling back to torpedo first."
            )
            return "torpedo_first"
        if self.fallback_first == "octagon":
            rospy.logwarn(
                f"[RandomPingerSelector] {reason}. Falling back to octagon first."
            )
            return "octagon_first"

        rospy.logerr(
            f"[RandomPingerSelector] Invalid fallback_first='{self.fallback_first}'."
        )
        return "aborted"

    def _wait_for_samples(self):
        self._clear_samples()
        start_time = rospy.Time.now()
        timeout_duration = rospy.Duration(self.timeout)
        rate = rospy.Rate(20)

        rospy.loginfo(
            f"[RandomPingerSelector] Collecting {self.sample_count} fresh pinger "
            f"sample(s) from '{self.angle_topic}' for up to {self.timeout:.1f}s."
        )

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return None, "preempted"

            samples = self._get_samples()
            if len(samples) >= self.sample_count:
                return samples[-self.sample_count :], None

            if rospy.Time.now() - start_time >= timeout_duration:
                if samples:
                    rospy.logwarn(
                        f"[RandomPingerSelector] Timed out after collecting "
                        f"{len(samples)}/{self.sample_count} pinger samples; using "
                        "available samples."
                    )
                    return samples, None
                return [], None

            rate.sleep()

        return None, "aborted"

    def _angle_to_frame(self, frame):
        if self.tf_freshness_threshold.to_sec() > 0.0:
            transform = lookup_fresh_transform(
                self.tf_buffer,
                self.source_frame,
                frame,
                self.tf_lookup_timeout,
                self.tf_freshness_threshold,
            )
        else:
            transform = self.tf_buffer.lookup_transform(
                self.source_frame,
                frame,
                rospy.Time(0),
                self.tf_lookup_timeout,
            )
        translation = transform.transform.translation
        return math.atan2(translation.y, translation.x)

    def execute(self, userdata):
        samples, outcome = self._wait_for_samples()
        if outcome is not None:
            return outcome

        if not samples:
            return self._fallback_outcome("No pinger samples received")

        filtered_samples = self._filter_outlier_samples(samples)
        if not filtered_samples:
            return self._fallback_outcome(
                "No pinger angle consensus after outlier rejection"
            )

        try:
            pinger_angle = circular_mean(filtered_samples)
            torpedo_angle = self._angle_to_frame(self.torpedo_frame)
            octagon_angle = self._angle_to_frame(self.octagon_frame)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            return self._fallback_outcome(f"TF lookup failed: {exc}")

        torpedo_error = angular_distance(pinger_angle, torpedo_angle)
        octagon_error = angular_distance(pinger_angle, octagon_angle)

        rospy.loginfo(
            "[RandomPingerSelector] pinger=%.3f rad using %d/%d sample(s), "
            "torpedo(%s)=%.3f rad error=%.3f rad, octagon(%s)=%.3f rad "
            "error=%.3f rad",
            pinger_angle,
            len(filtered_samples),
            len(samples),
            self.torpedo_frame,
            torpedo_angle,
            torpedo_error,
            self.octagon_frame,
            octagon_angle,
            octagon_error,
        )

        if abs(torpedo_error - octagon_error) <= self.tie_threshold:
            return self._fallback_outcome(
                f"Pinger comparison is within tie threshold ({self.tie_threshold:.3f} rad)"
            )

        if torpedo_error < octagon_error:
            rospy.loginfo("[RandomPingerSelector] Selected order: torpedo -> octagon")
            return "torpedo_first"

        rospy.loginfo("[RandomPingerSelector] Selected order: octagon -> torpedo")
        return "octagon_first"


class RandomPingerTaskState(smach.State):
    def __init__(
        self,
        torpedo_params,
        octagon_params,
        angle_topic="acoustic/hydrophone/base_angle",
        torpedo_frame="torpedo_map_link",
        octagon_frame="octagon_link",
        stabilization_time=2.0,
        sample_count=5,
        timeout=10.0,
        fallback_first="torpedo",
        tie_threshold=math.radians(5.0),
        outlier_rejection_threshold=math.radians(30.0),
        min_consensus_ratio=0.6,
        tf_lookup_timeout=2.0,
        tf_freshness_threshold=0.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "WAIT_FOR_PINGER_STABILIZATION",
                    "preempted": "preempted",
                    "aborted": "WAIT_FOR_PINGER_STABILIZATION",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_PINGER_STABILIZATION",
                DelayState(delay_time=stabilization_time),
                transitions={
                    "succeeded": "SELECT_PINGER_ORDER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SELECT_PINGER_ORDER",
                RandomPingerSelectorState(
                    angle_topic=angle_topic,
                    torpedo_frame=torpedo_frame,
                    octagon_frame=octagon_frame,
                    sample_count=sample_count,
                    timeout=timeout,
                    fallback_first=fallback_first,
                    tie_threshold=tie_threshold,
                    outlier_rejection_threshold=outlier_rejection_threshold,
                    min_consensus_ratio=min_consensus_ratio,
                    tf_lookup_timeout=tf_lookup_timeout,
                    tf_freshness_threshold=tf_freshness_threshold,
                ),
                transitions={
                    "torpedo_first": "TORPEDO_FIRST",
                    "octagon_first": "OCTAGON_FIRST",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TORPEDO_FIRST",
                TorpedoTaskState(**torpedo_params),
                transitions={
                    "succeeded": "OCTAGON_SECOND",
                    "preempted": "preempted",
                    "aborted": "OCTAGON_SECOND",
                },
            )
            smach.StateMachine.add(
                "OCTAGON_SECOND",
                OctagonTaskState(**octagon_params),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "OCTAGON_FIRST",
                OctagonTaskState(**octagon_params),
                transitions={
                    "succeeded": "TORPEDO_SECOND",
                    "preempted": "preempted",
                    "aborted": "TORPEDO_SECOND",
                },
            )
            smach.StateMachine.add(
                "TORPEDO_SECOND",
                TorpedoTaskState(**torpedo_params),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo("Starting Random Pinger SMACH Task")
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
