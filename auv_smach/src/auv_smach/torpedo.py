from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
import math

import rospy

from auv_smach.tf_utils import get_base_link
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    CheckForTransformState,
    SetDepthState,
    SearchForPropState,
    DynamicPathState,
    SetDetectionFocusState,
    SetDetectionState,
)
from auv_smach.initialize import DelayState
from auv_smach.acoustic import AcousticTransmitter


TORPEDO_PRIORITY_REALSENSE = "realsense"
TORPEDO_PRIORITY_DA3 = "da3"
TORPEDO_CLOSEST_METHOD_SERVICES = {
    TORPEDO_PRIORITY_REALSENSE: "enable_realsense_publisher",
    TORPEDO_PRIORITY_DA3: "enable_da3_publisher",
}


class TorpedoTargetFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_target_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class TorpedoRealsenseTargetFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_realsense_target_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class DA3PublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "enable_da3_publisher",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ResolveTorpedoClosestFrameState(smach.State):
    def __init__(
        self,
        torpedo_priority: str,
        source_frame: str = "odom",
        wait_time: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.torpedo_priority = torpedo_priority
        self.closest_frame = "torpedo_map_link_closest"
        self.source_frame = source_frame
        self.wait_time = wait_time

    def _ordered_methods(self):
        priority = (self.torpedo_priority or "").lower()
        if priority == TORPEDO_PRIORITY_DA3:
            return [TORPEDO_PRIORITY_DA3, TORPEDO_PRIORITY_REALSENSE]

        if priority != TORPEDO_PRIORITY_REALSENSE:
            rospy.logwarn(
                "Unknown torpedo_priority '%s'. Falling back to '%s'.",
                self.torpedo_priority,
                TORPEDO_PRIORITY_REALSENSE,
            )
        return [TORPEDO_PRIORITY_REALSENSE, TORPEDO_PRIORITY_DA3]

    def _set_bool_service(
        self,
        service_name: str,
        enabled: bool,
        wait_timeout: float = 2.0,
    ) -> bool:
        try:
            rospy.wait_for_service(service_name, timeout=wait_timeout)
            set_enabled = rospy.ServiceProxy(service_name, SetBool)
            response = set_enabled(SetBoolRequest(data=enabled))
            if not response.success:
                rospy.logwarn(
                    "Service %s returned success=False: %s",
                    service_name,
                    response.message,
                )
            return response.success
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("Failed to call %s: %s", service_name, exc)
            return False

    def _set_method_enabled(
        self,
        method: str,
        enabled: bool,
        wait_timeout: float = 2.0,
    ) -> bool:
        return self._set_bool_service(
            TORPEDO_CLOSEST_METHOD_SERVICES[method],
            enabled,
            wait_timeout=wait_timeout,
        )

    def _set_all_methods_enabled(self, enabled: bool):
        for method in TORPEDO_CLOSEST_METHOD_SERVICES:
            self._set_method_enabled(
                method,
                enabled,
                wait_timeout=0.5,
            )

    def _check_for_closest_frame(self, method: str):
        checker = CheckForTransformState(
            source_frame=self.source_frame,
            target_frame=self.closest_frame,
            timeout=self.wait_time,
        )
        checker.preempt_requested = self.preempt_requested
        checker.service_preempt = self.service_preempt

        outcome = checker.execute(None)
        if outcome == "succeeded" and method == TORPEDO_PRIORITY_REALSENSE:
            rospy.loginfo(
                "Realsense transform found. Waiting for 5.0s to let it publish and settle..."
            )
            start_sleep = rospy.Time.now()
            while not rospy.is_shutdown():
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"
                if (rospy.Time.now() - start_sleep).to_sec() >= 5.0:
                    break
                rospy.sleep(0.1)

        return outcome

    def execute(self, userdata):
        methods = self._ordered_methods()
        rospy.loginfo(
            "Torpedo closest frame priority: %s, fallback: %s",
            methods[0],
            methods[1],
        )

        self._set_all_methods_enabled(False)

        chosen_method = None
        target_publisher_enabled = False
        outcome = "aborted"
        try:
            if not self._set_bool_service(
                "set_transform_torpedo_realsense_target_frame", True
            ):
                return "aborted"
            target_publisher_enabled = True

            for method in methods:
                if self.preempt_requested():
                    self.service_preempt()
                    outcome = "preempted"
                    break

                rospy.loginfo("Trying torpedo closest frame source: %s", method)
                if not self._set_method_enabled(method, True):
                    rospy.logwarn(
                        "Could not enable torpedo closest frame source: %s", method
                    )
                    continue

                frame_outcome = self._check_for_closest_frame(method)

                if frame_outcome == "succeeded":
                    rospy.loginfo("Using torpedo closest frame source: %s", method)
                    outcome = "succeeded"
                    chosen_method = method
                    # If we used realsense, disable it now. If da3, keep it active.
                    if method == TORPEDO_PRIORITY_REALSENSE:
                        self._set_method_enabled(method, False)
                    break
                else:
                    self._set_method_enabled(method, False)
                    if frame_outcome == "preempted":
                        outcome = "preempted"
                        break

            return outcome
        finally:
            # Clean up only if we did not succeed with da3
            if not (outcome == "succeeded" and chosen_method == TORPEDO_PRIORITY_DA3):
                self._set_all_methods_enabled(False)
                if target_publisher_enabled:
                    self._set_bool_service(
                        "set_transform_torpedo_realsense_target_frame",
                        False,
                        wait_timeout=1.0,
                    )


class TorpedoFireFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_torpedo_hole_target_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class LaunchTorpedoState(smach_ros.ServiceState):
    def __init__(self, id: int):
        smach_ros.ServiceState.__init__(
            self,
            f"torpedo_{id}/launch",
            Trigger,
            request=TriggerRequest(),
        )


class TorpedoTaskState(smach.State):
    def __init__(
        self,
        torpedo_map_depth,
        torpedo_target_frame,
        torpedo_realsense_target_frame,
        torpedo_fire_frames,
        torpedo_priority: str = TORPEDO_PRIORITY_REALSENSE,
        torpedo_exit_angle: float = 0.0,
        torpedo_search_frame: str = "torpedo_map_link",
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.torpedo_fire_frames = torpedo_fire_frames
        self.torpedo_exit_angle = torpedo_exit_angle
        self.torpedo_search_frame = torpedo_search_frame
        self.base_link = get_base_link()

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA_FOCUS",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "FOCUS_ON_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_TORPEDO",
                SetDetectionFocusState(focus_object="torpedo"),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoTargetFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "SET_TORPEDO_MAP_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_MAP_DEPTH",
                SetDepthState(depth=torpedo_map_depth),
                transitions={
                    "succeeded": "FIND_AND_AIM_TORPEDO_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_TORPEDO_MAP",
                SearchForPropState(
                    look_at_frame=self.torpedo_search_frame,
                    alignment_frame="torpedo_map_travel_start",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.4,
                ),
                transitions={
                    "succeeded": "TRANSMIT_ACOUSTIC_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TRANSMIT_ACOUSTIC_1",
                AcousticTransmitter(acoustic_data=1),
                transitions={
                    "succeeded": "PATH_TO_TORPEDO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_TO_TORPEDO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame=torpedo_target_frame,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_CLOSE_APPROACH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=torpedo_target_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=10.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_TORPEDO_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_TORPEDO_FRAME_PUBLISHER",
                TorpedoTargetFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ROTATE_FOR_REALSENSE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROTATE_FOR_REALSENSE",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=torpedo_target_frame,
                    angle_offset=math.pi,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "RESOLVE_TORPEDO_CLOSEST_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "RESOLVE_TORPEDO_CLOSEST_FRAME",
                ResolveTorpedoClosestFrameState(torpedo_priority=torpedo_priority),
                transitions={
                    "succeeded": "ALIGN_TO_ORIENTED_TORPEDO_MAP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_ORIENTED_TORPEDO_MAP",
                AlignFrame(
                    source_frame=f"{self.base_link}/torpedo_camera_link",
                    target_frame=torpedo_realsense_target_frame,
                    angle_offset=0.0,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=3.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_DA3_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_DA3_PUBLISHER",
                DA3PublisherServiceState(req=False),
                transitions={
                    "succeeded": "DISABLE_REALSENSE_TARGET_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_REALSENSE_TARGET_PUBLISHER",
                TorpedoRealsenseTargetFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "SET_TORPEDO_HOLES_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_HOLES_DETECTION",
                SetDetectionState(camera_name="torpedo", enable=True),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_HOLES_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_HOLES_DETECTION",
                AlignFrame(
                    source_frame=f"{self.base_link}/torpedo_camera_link",
                    target_frame=torpedo_realsense_target_frame,
                    angle_offset=0.0,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=5.0,
                ),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                TorpedoFireFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_FIRE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FIRE_FRAME",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "ALIGN_TO_TORPEDO_FIRE_FRAME_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TORPEDO_FIRE_FRAME_1",
                AlignFrame(
                    source_frame=f"{self.base_link}/torpedo_upper_link",
                    target_frame=self.torpedo_fire_frames[0],
                    angle_offset=0.0,
                    dist_threshold=0.03,
                    yaw_threshold=0.05,
                    confirm_duration=5.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_1",
                LaunchTorpedoState(id=1),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_LAUNCH_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_LAUNCH_1",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "ALIGN_TO_TORPEDO_FIRE_FRAME_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TORPEDO_FIRE_FRAME_2",
                AlignFrame(
                    source_frame=f"{self.base_link}/torpedo_bottom_link",
                    target_frame=self.torpedo_fire_frames[1],
                    angle_offset=0.0,
                    dist_threshold=0.03,
                    yaw_threshold=0.05,
                    confirm_duration=5.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_2",
                LaunchTorpedoState(id=2),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_2_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_2_LAUNCH",
                DelayState(delay_time=3.0),
                transitions={
                    "succeeded": "DISABLE_TORPEDO_HOLES_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_TORPEDO_HOLES_DETECTION",
                SetDetectionState(camera_name="torpedo", enable=False),
                transitions={
                    "succeeded": "ALIGN_TO_TORPEDO_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TORPEDO_EXIT",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=torpedo_realsense_target_frame,
                    angle_offset=self.torpedo_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.0,
                    timeout=10.0,
                    cancel_on_success=False,
                    use_frame_depth=False,
                ),
                transitions={
                    "succeeded": "TRANSMIT_ACOUSTIC_4",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TRANSMIT_ACOUSTIC_4",
                AcousticTransmitter(acoustic_data=4),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
