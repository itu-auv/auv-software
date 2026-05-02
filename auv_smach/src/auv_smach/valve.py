from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SetDepthState,
)
from auv_smach.initialize import DelayState


class ValveApproachFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_approach_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ValveContactFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_contact_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ValveTaskState(smach.State):
    """Two-phase valve task: oriented approach → contact.

    The keypoint pipeline (sim_keypoint_node or valve_keypoint_node →
    keypoint_pose_node → object_map_tf_server) publishes `tac/valve` as
    soon as the robot is facing the valve, so no rotational search is
    needed. The valve_trajectory_publisher derives `valve_approach_frame`
    and `valve_contact_frame` from `tac/valve` once their respective
    SetBool gates are enabled.
    """

    def __init__(
        self,
        valve_depth,
        valve_approach_frame: str = "valve_approach_frame",
        valve_contact_frame: str = "valve_contact_frame",
        valve_exit_angle: float = 0.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.valve_exit_angle = valve_exit_angle

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        self.state_machine.set_initial_state(["SET_VALVE_DEPTH"])

        with self.state_machine:
            # 1. Descend to valve depth
            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth),
                transitions={
                    "succeeded": "ENABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # =========================================================
            #  PHASE 1: ORIENTED APPROACH (perpendicular to surface normal)
            # =========================================================

            # 2. Enable approach frame publisher
            smach.StateMachine.add(
                "ENABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 3. Wait for approach frame to be broadcast
            smach.StateMachine.add(
                "WAIT_FOR_APPROACH_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 4. Align to approach frame
            smach.StateMachine.add(
                "ALIGN_TO_APPROACH",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=valve_approach_frame,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 5. Disable approach publisher
            smach.StateMachine.add(
                "DISABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ENABLE_CONTACT_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # =========================================================
            #  PHASE 2: CONTACT (perpendicular to surface normal, contact)
            # =========================================================

            # 6. Enable contact publisher
            smach.StateMachine.add(
                "ENABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_CONTACT_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 7. Wait for contact frame to be broadcast
            smach.StateMachine.add(
                "WAIT_FOR_CONTACT_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_CONTACT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 8. Precise alignment to contact frame
            smach.StateMachine.add(
                "ALIGN_TO_CONTACT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=valve_contact_frame,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=5.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "DISABLE_CONTACT_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 9. Disable contact publisher
            smach.StateMachine.add(
                "DISABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 10. Reset controller
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
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
