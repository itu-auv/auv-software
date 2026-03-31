#!/usr/bin/env python3

import smach
import smach_ros
import rospy

from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import DynamicPathState, AlignFrame
from auv_smach.tf_utils import get_base_link


class ToggleStationTrajectoryState(smach_ros.ServiceState):
    def __init__(self, enabled: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_station_trajectory",
            SetBool,
            request=SetBoolRequest(data=enabled),
            outcomes=["succeeded", "preempted", "aborted"],
        )


class WaitForStationSelectionState(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["selected_first_target"],
        )
        self.selection_topic = str(
            rospy.get_param("~pinger_decision/selection_topic", "station_target_selection")
        )
        self.selection_timeout = float(
            rospy.get_param("~pinger_decision/selection_timeout", 10.0)
        )

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        try:
            msg = rospy.wait_for_message(
                self.selection_topic,
                String,
                timeout=self.selection_timeout,
            )
        except rospy.ROSException as exc:
            rospy.logwarn(
                "Timed out waiting for station selection on %s: %s",
                self.selection_topic,
                str(exc),
            )
            return "aborted"

        selection = msg.data.strip()
        if selection not in ["NAVIGATE_TO_TORPEDO_TASK", "NAVIGATE_TO_OCTAGON_TASK"]:
            rospy.logwarn("Invalid station selection received: %s", selection)
            return "aborted"

        userdata.selected_first_target = selection
        rospy.loginfo("Station selection received: %s", selection)
        return "succeeded"


class ExecutePostPingerSequenceState(smach.State):
    def __init__(self, state_factory):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=["selected_first_target"],
        )
        self.state_factory = state_factory
        self.sequence_template = list(
            rospy.get_param(
                "~post_pinger_sequence_template",
                ["FIRST_TARGET", "SECOND_TARGET"],
            )
        )

    @staticmethod
    def _second_target(first_target: str) -> str:
        if first_target == "NAVIGATE_TO_TORPEDO_TASK":
            return "NAVIGATE_TO_OCTAGON_TASK"
        return "NAVIGATE_TO_TORPEDO_TASK"

    def _resolve_sequence(self, first_target: str):
        second_target = self._second_target(first_target)
        resolved = []
        for token in self.sequence_template:
            if token == "FIRST_TARGET":
                resolved.append(first_target)
            elif token == "SECOND_TARGET":
                resolved.append(second_target)
            else:
                resolved.append(token)
        return resolved

    def execute(self, userdata):
        if "SECOND_TARGET" not in self.sequence_template:
            rospy.logerr("post_pinger_sequence_template must include SECOND_TARGET")
            return "aborted"

        first_target = getattr(userdata, "selected_first_target", "").strip()
        if first_target not in ["NAVIGATE_TO_TORPEDO_TASK", "NAVIGATE_TO_OCTAGON_TASK"]:
            rospy.logerr("selected_first_target is invalid: %s", first_target)
            return "aborted"

        run_list = self._resolve_sequence(first_target)
        rospy.loginfo("Resolved post-pinger sequence: %s", run_list)

        for state_name in run_list:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            state_instance = self.state_factory(state_name)
            if state_instance is None:
                rospy.logerr("Unknown state in post-pinger sequence: %s", state_name)
                return "aborted"

            outcome = state_instance.execute({})
            if outcome == "preempted":
                self.service_preempt()
                return "preempted"

            if outcome != "succeeded":
                rospy.logwarn(
                    "State %s returned %s, continuing by policy",
                    state_name,
                    outcome,
                )

        return "succeeded"


class PingerDecisionAndExecutionState(smach.StateMachine):
    def __init__(self, state_factory):
        smach.StateMachine.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["selected_first_target"],
        )

        base_link = get_base_link()
        station_frame = str(
            rospy.get_param("~pinger_decision/station_frame", "station_frame")
        )

        with self:
            smach.StateMachine.add(
                "ENABLE_STATION_TRAJECTORY",
                ToggleStationTrajectoryState(enabled=True),
                transitions={
                    "succeeded": "PATH_TO_STATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "PATH_TO_STATION",
                DynamicPathState(plan_target_frame=station_frame),
                transitions={
                    "succeeded": "ALIGN_TO_STATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_STATION",
                AlignFrame(
                    source_frame=base_link,
                    target_frame=station_frame,
                    dist_threshold=0.2,
                    yaw_threshold=0.2,
                    confirm_duration=1.0,
                    timeout=20.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_SELECTION",
                    "preempted": "preempted",
                    "aborted": "DISABLE_STATION_TRAJECTORY_ON_FAIL",
                },
            )

            smach.StateMachine.add(
                "WAIT_FOR_SELECTION",
                WaitForStationSelectionState(),
                transitions={
                    "succeeded": "DISABLE_STATION_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "DISABLE_STATION_TRAJECTORY_ON_FAIL",
                },
                remapping={"selected_first_target": "selected_first_target"},
            )

            smach.StateMachine.add(
                "DISABLE_STATION_TRAJECTORY",
                ToggleStationTrajectoryState(enabled=False),
                transitions={
                    "succeeded": "EXECUTE_POST_PINGER_SEQUENCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "EXECUTE_POST_PINGER_SEQUENCE",
                ExecutePostPingerSequenceState(state_factory=state_factory),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
                remapping={"selected_first_target": "selected_first_target"},
            )

            smach.StateMachine.add(
                "DISABLE_STATION_TRAJECTORY_ON_FAIL",
                ToggleStationTrajectoryState(enabled=False),
                transitions={
                    "succeeded": "aborted",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
