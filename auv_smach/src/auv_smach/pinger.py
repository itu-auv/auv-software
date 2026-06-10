#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

from auv_smach.tf_utils import get_base_link
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    DynamicPathState,
)
from auv_smach.initialize import DelayState


class WaitForPingerSamples(smach.State):
    def __init__(self, service_name="has_enough_pinger_samples", poll_rate=2.0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.service_name = service_name
        self.poll_rate = poll_rate
        self.sample_check_srv = rospy.ServiceProxy(self.service_name, Trigger)

    def execute(self, userdata):
        rospy.loginfo("Waiting for enough raw pinger samples.")

        try:
            rospy.wait_for_service(self.service_name, timeout=5.0)
        except rospy.ROSException as e:
            rospy.logerr(f"Service '{self.service_name}' is not available: {e}")
            return "aborted"

        rate = rospy.Rate(self.poll_rate)
        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            try:
                resp = self.sample_check_srv(TriggerRequest())
            except rospy.ServiceException as e:
                rospy.logwarn_throttle(
                    2.0, f"Failed to check pinger sample count: {e}"
                )
                rate.sleep()
                continue

            if resp.success:
                rospy.loginfo(f"Enough raw pinger samples received: {resp.message}")
                return "succeeded"

            rospy.loginfo_throttle(2.0, resp.message)
            rate.sleep()

        return "aborted"


def make_pinger_leg(collect_duration=20.0, waypoint_frame="pinger_waypoint"):
    """
    Creates a StateMachine for one leg of the pinger task:
    1. Call the service to publish the next waypoint frame.
    2. Align to the waypoint.
    3. Cancel active control (stop).
    4. Wait for vehicle stabilization.
    5. Start pinger collection.
    6. Wait until enough raw samples are collected.
    7. Stop pinger collection.
    """
    sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

    with sm:
        # Publish the next waypoint frame.
        smach.StateMachine.add(
            "PUBLISH_WAYPOINT",
            smach_ros.ServiceState(
                "publish_pinger_waypoint", Trigger, request=TriggerRequest()
            ),
            transitions={
                "succeeded": "ALIGN_TO_WAYPOINT",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

        # Align to the waypoint frame
        smach.StateMachine.add(
            "ALIGN_TO_WAYPOINT",
            AlignFrame(
                source_frame=get_base_link(),
                target_frame=waypoint_frame,
                dist_threshold=0.15,
                yaw_threshold=0.2,
                timeout=30.0,
                confirm_duration=5.0,
                cancel_on_success=True,
            ),
            transitions={
                "succeeded": "CANCEL_CONTROL",
                "preempted": "preempted",
                "aborted": "CANCEL_CONTROL",
            },
        )

        # Stop control
        smach.StateMachine.add(
            "CANCEL_CONTROL",
            CancelAlignControllerState(),
            transitions={
                "succeeded": "WAIT_STABILISE",
                "preempted": "preempted",
                "aborted": "WAIT_STABILISE",
            },
        )

        # Wait stabilization
        smach.StateMachine.add(
            "WAIT_STABILISE",
            DelayState(delay_time=2.0),
            transitions={
                "succeeded": "START_COLLECTION",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

        # Start pinger collection
        smach.StateMachine.add(
            "START_COLLECTION",
            smach_ros.ServiceState(
                "toggle_pinger_collection", SetBool, request=SetBoolRequest(data=True)
            ),
            transitions={
                "succeeded": "WAIT_RAW_SAMPLES",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

        # Wait until enough raw samples are collected.
        smach.StateMachine.add(
            "WAIT_RAW_SAMPLES",
            WaitForPingerSamples(),
            transitions={
                "succeeded": "STOP_COLLECTION",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

        # Stop pinger collection
        smach.StateMachine.add(
            "STOP_COLLECTION",
            smach_ros.ServiceState(
                "toggle_pinger_collection", SetBool, request=SetBoolRequest(data=False)
            ),
            transitions={
                "succeeded": "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

    return sm


class PingerTaskState(smach.State):
    """
    Main Pinger Task State:
    1. Resets/clears pinger data.
    2. Runs 3 pinger legs sequentially: forward, left, right.
    3. Triggers calculation of the pinger position.
    4. Aligns to the calculated pinger frame.
    5. Cancels final alignment.
    """

    def __init__(
        self,
        n_legs=3,
        leg_distance=2.0,
        collect_duration=20.0,
        pinger_frame="pinger_frame",
        waypoint_frame="pinger_waypoint",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.n_legs = n_legs
        self.collect_duration = collect_duration
        self.pinger_frame = pinger_frame
        self.waypoint_frame = waypoint_frame

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with self.sm:
            # Clear previous pinger data
            smach.StateMachine.add(
                "RESET_PINGER_DATA",
                smach_ros.ServiceState(
                    "clear_pinger_data", Trigger, request=TriggerRequest()
                ),
                transitions={
                    "succeeded": "LEG_1",
                    "preempted": "preempted",
                    "aborted": "LEG_1",
                },
            )

            # Add legs
            for i in range(1, n_legs + 1):
                leg_name = f"LEG_{i}"
                next_state = f"LEG_{i+1}" if i < n_legs else "COMPUTE_POSITION"
                smach.StateMachine.add(
                    leg_name,
                    make_pinger_leg(
                        collect_duration=self.collect_duration,
                        waypoint_frame=self.waypoint_frame,
                    ),
                    transitions={
                        "succeeded": next_state,
                        "preempted": "preempted",
                        "aborted": next_state,
                    },
                )

            # Compute pinger position
            smach.StateMachine.add(
                "COMPUTE_POSITION",
                smach_ros.ServiceState(
                    "compute_pinger_position", Trigger, request=TriggerRequest()
                ),
                transitions={
                    "succeeded": "DYNAMIC_TO_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                    "DYNAMIC_TO_PINGER",
                    DynamicPathState(
                        plan_target_frame=self.pinger_frame,
                    ),
                    transitions={
                        "succeeded": "ALIGN_TO_PINGER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                # Align to the calculated pinger_frame
            smach.StateMachine.add(
                    "ALIGN_TO_PINGER",
                    AlignFrame(
                        source_frame=get_base_link(),
                        target_frame=self.pinger_frame,
                        dist_threshold=0.3,
                        yaw_threshold=0.2,
                        timeout=30.0,
                        confirm_duration=2.0,
                        cancel_on_success=True,
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
