import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    DynamicPathState,
    SearchForPropState,
    SetDepthState,
)
from auv_smach.initialize import DelayState
from auv_smach.red_buoy import RotateAroundBuoyState
from auv_smach.tf_utils import get_base_link


BUOY_FRAME = "buoy"
BUOY_CLOSE_APPROACH_FRAME = "buoy_close_approach"
SURFACE_FRAME = "surface"
BUOY_ROTATION_START_FRAME = "buoy_rotation_start"
BUOY_ROTATION_TARGET_FRAME = "buoy_rotation_target"


class BuoyTrajectoryPublisherState(smach_ros.ServiceState):
    def __init__(self, enable):
        super().__init__(
            "toggle_buoy_trajectory",
            SetBool,
            request=SetBoolRequest(data=enable),
        )


class BuoySurfaceTaskState(smach.State):
    """Approach and circle the buoy, then navigate to and surface in the field."""

    def __init__(
        self,
        mission_depth_m=-0.7,
        rotation_radius_m=3.0,
        rotation_direction="ccw",
        frame_wait_seconds=2.0,
        buoy_frame=BUOY_FRAME,
        close_approach_frame=BUOY_CLOSE_APPROACH_FRAME,
        surface_frame=SURFACE_FRAME,
    ):
        super().__init__(outcomes=["succeeded", "preempted", "aborted"])
        if rotation_direction not in ("cw", "ccw"):
            raise ValueError("rotation_direction must be 'cw' or 'ccw'")
        if rotation_radius_m <= 0.0:
            raise ValueError("rotation_radius_m must be positive")

        self.base_link = get_base_link()
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_DEPTH",
                SetDepthState(depth=mission_depth_m),
                transitions={
                    "succeeded": "ENABLE_BUOY_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BUOY_TRAJECTORY_PUBLISHER",
                BuoyTrajectoryPublisherState(enable=True),
                transitions={
                    "succeeded": "WAIT_FOR_BUOY_CLOSE_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BUOY_CLOSE_APPROACH_FRAME",
                DelayState(delay_time=frame_wait_seconds),
                transitions={
                    "succeeded": "SEARCH_FOR_BUOY_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_BUOY_CLOSE_APPROACH",
                SearchForPropState(
                    look_at_frame=close_approach_frame,
                    alignment_frame="buoy_close_approach_search",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_BUOY_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_BUOY_CLOSE_APPROACH",
                DynamicPathState(plan_target_frame=close_approach_frame),
                transitions={
                    "succeeded": "ALIGN_TO_BUOY_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BUOY_CLOSE_APPROACH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=close_approach_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=15.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ROTATE_AROUND_BUOY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROTATE_AROUND_BUOY",
                RotateAroundBuoyState(
                    radius=rotation_radius_m,
                    direction=rotation_direction,
                    red_buoy_depth=mission_depth_m,
                    buoy_frame=buoy_frame,
                    rotation_start_frame=BUOY_ROTATION_START_FRAME,
                    target_frame=BUOY_ROTATION_TARGET_FRAME,
                ),
                transitions={
                    "succeeded": "SEARCH_FOR_SURFACE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_SURFACE_FRAME",
                SearchForPropState(
                    look_at_frame=surface_frame,
                    alignment_frame="surface_search",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_SURFACE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_SURFACE_FRAME",
                DynamicPathState(plan_target_frame=surface_frame,),
                transitions={
                    "succeeded": "ALIGN_TO_SURFACE_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_SURFACE_FRAME",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=surface_frame,
                    dist_threshold=0.25,
                    yaw_threshold=0.2,
                    confirm_duration=0.5,
                    timeout=20.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "SURFACE_WITHIN_FIELD",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACE_WITHIN_FIELD",
                SetDepthState(depth=0.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def request_preempt(self):
        smach.State.request_preempt(self)
        self.state_machine.request_preempt()

    def execute(self, userdata):
        rospy.loginfo("[BuoySurface] Starting buoy and surface task")
        outcome = self.state_machine.execute()
        return "preempted" if outcome is None else outcome
