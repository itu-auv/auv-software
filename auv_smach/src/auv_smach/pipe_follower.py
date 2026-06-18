from .initialize import *
import rospy
import smach
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import SetScanEnabled, SetScanEnabledRequest

from auv_smach.common import (
    AlignFrame,
    SetDepthState,
)


class EnablePipeFramePublisherState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "pipe_frame_publisher/enable",
            Trigger,
            request=TriggerRequest(),
        )


class DisablePipeFramePublisherState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "pipe_frame_publisher/disable",
            Trigger,
            request=TriggerRequest(),
        )


class EnablePipeFollowerLegacyState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "pipe_follower_legacy/enable",
            Trigger,
            request=TriggerRequest(),
        )


class DisablePipeFollowerLegacyState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "pipe_follower_legacy/disable",
            Trigger,
            request=TriggerRequest(),
        )


class EnableArucoState(smach_ros.ServiceState):
    def __init__(self, folder_name):
        smach_ros.ServiceState.__init__(
            self,
            "aruco_detection/set_enabled",
            SetScanEnabled,
            request=SetScanEnabledRequest(True, folder_name),
        )


class DisableArucoState(smach_ros.ServiceState):
    def __init__(self, folder_name):
        smach_ros.ServiceState.__init__(
            self,
            "aruco_detection/set_enabled",
            SetScanEnabled,
            request=SetScanEnabledRequest(False, folder_name),
        )


class TimedPipeFollowerLegacyState(smach.State):
    def __init__(self, duration):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.duration = duration

    def execute(self, userdata):
        rospy.loginfo(
            "[PipeTaskState] Running legacy pipe follower for %.1f seconds",
            self.duration,
        )
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            if (rospy.Time.now() - start_time).to_sec() >= self.duration:
                return "succeeded"

            rate.sleep()

        return "aborted"


class PipeTaskState(smach.State):
    def __init__(self, pipe_map_depth, pipe_target_frame, pipe_method="new"):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.pipe_method = pipe_method
        self.legacy_duration = rospy.get_param("~pipe_legacy_duration", 60.0)

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        if self.pipe_method not in ("new", "legacy"):
            rospy.logwarn(
                "[PipeTaskState] Unknown pipe_method '%s', using 'new'",
                self.pipe_method,
            )
            self.pipe_method = "new"

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_ARUCO",
                EnableArucoState("uykum_var"),
                transitions={
                    "succeeded": (
                        "ENABLE_LEGACY"
                        if self.pipe_method == "legacy"
                        else "ENABLE_PUBLISHER"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ADINI_DEGISTIR",
                SetDepthState(depth=pipe_map_depth, confirm_duration=3.0),
                transitions={
                    "succeeded": (
                        "ENABLE_LEGACY"
                        if self.pipe_method == "legacy"
                        else "ENABLE_PUBLISHER"
                    ),
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            if self.pipe_method == "legacy":
                smach.StateMachine.add(
                    "ENABLE_LEGACY",
                    EnablePipeFollowerLegacyState(),
                    transitions={
                        "succeeded": "FOLLOW_PIPE_LEGACY",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "FOLLOW_PIPE_LEGACY",
                    TimedPipeFollowerLegacyState(duration=self.legacy_duration),
                    transitions={
                        "succeeded": "DISABLE_LEGACY",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "DISABLE_LEGACY",
                    DisablePipeFollowerLegacyState(),
                    transitions={
                        "succeeded": "DISABLE_ARUCO_SUCCEEDED",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
            else:
                smach.StateMachine.add(
                    "ENABLE_PUBLISHER",
                    EnablePipeFramePublisherState(),
                    transitions={
                        "succeeded": "ALIGN_TO_PIPE",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "ALIGN_TO_PIPE",
                    AlignFrame(
                        source_frame="taluy/base_link",
                        target_frame=pipe_target_frame,
                        dist_threshold=0.2,
                        yaw_threshold=0.2,
                        max_linear_velocity=0.2,
                        max_angular_velocity=0.4,
                        confirm_duration=5.0,
                        timeout=120.0,
                        cancel_on_success=False,
                        use_frame_depth=True,
                    ),
                    transitions={
                        "succeeded": "DISABLE_PUBLISHER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

                smach.StateMachine.add(
                    "DISABLE_PUBLISHER",
                    DisablePipeFramePublisherState(),
                    transitions={
                        "succeeded": "DISABLE_ARUCO_FINISH",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    "DISABLE_ARUCO_FINISH",
                    DisableArucoState("uykum_var"),
                    transitions={
                        "succeeded": "succeeded",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

    def execute(self, userdata):
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
