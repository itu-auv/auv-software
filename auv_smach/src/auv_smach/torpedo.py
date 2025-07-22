from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
import math

from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    DynamicPathState,
    SetDetectionFocusState,
    SetDetectionState,
)
from auv_smach.initialize import DelayState


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
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.torpedo_fire_frames = torpedo_fire_frames

        torpedo_task_params = rospy.get_param("~torpedo_task", {})
        set_torpedo_depth_params = torpedo_task_params.get("set_torpedo_depth", {})
        find_and_aim_torpedo_params = torpedo_task_params.get(
            "find_and_aim_torpedo", {}
        )
        wait_for_aligning_start_params = torpedo_task_params.get(
            "wait_for_aligning_start", {}
        )
        set_torpedo_close_approach_frame_params = torpedo_task_params.get(
            "set_torpedo_close_approach_frame", {}
        )
        wait_for_close_approach_complete_params = torpedo_task_params.get(
            "wait_for_close_approach_complete", {}
        )
        launch_torpedo_1_params = torpedo_task_params.get("launch_torpedo_1", {})
        wait_for_torpedo_launch_params = torpedo_task_params.get(
            "wait_for_torpedo_launch", {}
        )
        launch_torpedo_2_params = torpedo_task_params.get("launch_torpedo_2", {})
        wait_for_torpedo_2_launch_params = torpedo_task_params.get(
            "wait_for_torpedo_2_launch", {}
        )
        set_torpedo_exit_depth_params = torpedo_task_params.get(
            "set_torpedo_exit_depth", {}
        )

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "SET_CAMERA_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_CAMERA_FOCUS",
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
                    "succeeded": "SET_TORPEDO_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_TORPEDO_DEPTH",
                SetDepthState(
                    depth=torpedo_map_depth,
                    sleep_duration=set_torpedo_depth_params.get("sleep_duration", 3.0),
                ),
                transitions={
                    "succeeded": "FIND_AND_AIM_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_TORPEDO",
                SearchForPropState(
                    look_at_frame="torpedo_map_link",
                    alignment_frame="torpedo_map_travel_start",
                    full_rotation=find_and_aim_torpedo_params.get(
                        "full_rotation", False
                    ),
                    set_frame_duration=find_and_aim_torpedo_params.get(
                        "set_frame_duration", 7.0
                    ),
                    source_frame="taluy/base_link",
                    rotation_speed=find_and_aim_torpedo_params.get(
                        "rotation_speed", 0.3
                    ),
                ),
                transitions={
                    "succeeded": "PATH_TO_TORPEDO_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PATH_TO_TORPEDO_TARGET",
                DynamicPathState(
                    plan_target_frame=torpedo_target_frame,
                ),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_TO_TORPEDO_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_TO_TORPEDO_TARGET",
                AlignFrame(
                    source_frame="taluy/base_link",
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
                    source_frame="taluy/base_link",
                    target_frame=torpedo_target_frame,
                    angle_offset=math.pi,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                TorpedoRealsenseTargetFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_FRAME",
                DelayState(delay_time=10.0),
                transitions={
                    "succeeded": "DISABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_TORPEDO_REALSENSE_FRAME_PUBLISHER",
                TorpedoRealsenseTargetFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "TURN_TO_LAUNCH_TORPEDO",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TURN_TO_LAUNCH_TORPEDO",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=torpedo_realsense_target_frame,
                    angle_offset=-math.pi / 2,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=7.0,
                    timeout=30.0,
                    cancel_on_success=False,
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
                    "succeeded": "DISABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_TORPEDO_FIRE_FRAME_PUBLISHER",
                TorpedoFireFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "SET_FIRE_DEPTH_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FIRE_DEPTH_1",
                SetDepthState(
                    depth=0.2,
                    sleep_duration=5.0,
                    frame_id=self.torpedo_fire_frames[0],
                ),
                transitions={
                    "succeeded": "ALIGN_TO_TORPEDO_FIRE_FRAME_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TORPEDO_FIRE_FRAME_1",
                AlignFrame(
                    source_frame="taluy/base_link/torpedo_upper_link",
                    target_frame=self.torpedo_fire_frames[0],
                    angle_offset=-math.pi / 2,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=10.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                ),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_1",
                LaunchTorpedoState(id=launch_torpedo_1_params.get("id", 1)),
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
                    "succeeded": "SET_FIRE_DEPTH_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FIRE_DEPTH_2",
                SetDepthState(
                    depth=0.2,
                    sleep_duration=5.0,
                    frame_id=self.torpedo_fire_frames[1],
                ),
                transitions={
                    "succeeded": "ALIGN_TO_TORPEDO_FIRE_FRAME_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TORPEDO_FIRE_FRAME_2",
                AlignFrame(
                    source_frame="taluy/base_link/torpedo_bottom_link",
                    target_frame=self.torpedo_fire_frames[1],
                    angle_offset=-math.pi / 2,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=10.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                ),
                transitions={
                    "succeeded": "LAUNCH_TORPEDO_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LAUNCH_TORPEDO_2",
                LaunchTorpedoState(id=launch_torpedo_2_params.get("id", 2)),
                transitions={
                    "succeeded": "WAIT_FOR_TORPEDO_2_LAUNCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_TORPEDO_2_LAUNCH",
                DelayState(
                    delay_time=wait_for_torpedo_2_launch_params.get("delay_time", 3.0)
                ),
                transitions={
                    "succeeded": "GET_BACK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "GET_BACK",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=torpedo_realsense_target_frame,
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.0,
                    timeout=10.0,
                    cancel_on_success=False,
                ),
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
