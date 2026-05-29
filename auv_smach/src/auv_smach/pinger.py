import smach
import smach_ros
import rospy

from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

from auv_smach.tf_utils import get_base_link
from auv_smach.common import SetDepthState, DynamicPathState
from auv_smach.initialize import DelayState


# Sample frames are broadcast by pinger_trajectory_publisher relative to
# mission_start_link (+/- sample_distance along X/Y).
SAMPLE_FRAMES = [
    "pinger_sample_xp",
    "pinger_sample_xn",
    "pinger_sample_yp",
    "pinger_sample_yn",
]
PINGER_FRAME = "pinger"


class TogglePingerTrajectoryState(smach_ros.ServiceState):
    """Enable/disable the pinger trajectory publisher (broadcasts sample frames)."""

    def __init__(self, enable: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_pinger_trajectory",
            SetBool,
            request=SetBoolRequest(data=enable),
            response_cb=self.response_cb,
        )

    @staticmethod
    def response_cb(userdata, response):
        if not response.success:
            rospy.logwarn("TogglePingerTrajectoryState failed: %s", response.message)
            return "aborted"
        return "succeeded"


class RecordPingerBearingState(smach_ros.ServiceState):
    """Average the recent bearing buffer and store it as a triangulation ray."""

    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "record_pinger_bearing",
            Trigger,
            request=TriggerRequest(),
            response_cb=self.response_cb,
        )

    @staticmethod
    def response_cb(userdata, response):
        if not response.success:
            rospy.logwarn("RecordPingerBearingState failed: %s", response.message)
            return "aborted"
        return "succeeded"


class ComputePingerFrameState(smach_ros.ServiceState):
    """Triangulate the stored rays and broadcast the `pinger` frame."""

    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "compute_pinger_frame",
            Trigger,
            request=TriggerRequest(),
            response_cb=self.response_cb,
        )

    @staticmethod
    def response_cb(userdata, response):
        if not response.success:
            rospy.logwarn("ComputePingerFrameState failed: %s", response.message)
            return "aborted"
        return "succeeded"


class PingerTaskState(smach.State):
    """
    Triangulate an acoustic pinger by sampling its bearing from four locations
    (+/- 2m in X and Y relative to mission_start_link), intersecting the bearing
    rays, then navigating to the resulting `pinger` frame.

    Flow:
      SET_PINGER_DEPTH
      ENABLE_PINGER_TRAJECTORY        -> broadcasts the four sample frames
      for each sample frame:
        NAVIGATE_TO_<frame>           -> DynamicPathState (path planner + follow path)
        DWELL_<frame>                 -> hold still while the bearing buffer fills
        RECORD_<frame>                -> average bearing, store a ray
      COMPUTE_PINGER                  -> least-squares intersection -> `pinger` frame
      NAVIGATE_TO_PINGER
      DISABLE_PINGER_TRAJECTORY
    """

    def __init__(
        self,
        search_depth: float,
        dwell_time: float = 4.0,
        max_linear_velocity: float = None,
        max_angular_velocity: float = None,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_PINGER_DEPTH",
                SetDepthState(depth=search_depth),
                transitions={
                    "succeeded": "ENABLE_PINGER_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_PINGER_TRAJECTORY",
                TogglePingerTrajectoryState(enable=True),
                transitions={
                    "succeeded": f"NAVIGATE_TO_{SAMPLE_FRAMES[0]}",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Build the four sample sub-sequences, chaining to the next one.
            for index, frame in enumerate(SAMPLE_FRAMES):
                is_last = index == len(SAMPLE_FRAMES) - 1
                next_state = (
                    "COMPUTE_PINGER"
                    if is_last
                    else f"NAVIGATE_TO_{SAMPLE_FRAMES[index + 1]}"
                )

                smach.StateMachine.add(
                    f"NAVIGATE_TO_{frame}",
                    DynamicPathState(
                        plan_target_frame=frame,
                        max_linear_velocity=max_linear_velocity,
                        max_angular_velocity=max_angular_velocity,
                    ),
                    transitions={
                        "succeeded": f"DWELL_{frame}",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    f"DWELL_{frame}",
                    DelayState(delay_time=dwell_time),
                    transitions={
                        "succeeded": f"RECORD_{frame}",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    f"RECORD_{frame}",
                    RecordPingerBearingState(),
                    transitions={
                        "succeeded": next_state,
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            smach.StateMachine.add(
                "COMPUTE_PINGER",
                ComputePingerFrameState(),
                transitions={
                    "succeeded": "NAVIGATE_TO_PINGER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_PINGER",
                DynamicPathState(
                    plan_target_frame=PINGER_FRAME,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                ),
                transitions={
                    "succeeded": "DISABLE_PINGER_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_PINGER_TRAJECTORY",
                TogglePingerTrajectoryState(enable=False),
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
