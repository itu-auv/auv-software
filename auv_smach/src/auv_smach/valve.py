from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest

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


class ValveCoarseApproachFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_coarse_approach_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


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
    def __init__(
        self,
        valve_depth,
        valve_coarse_approach_frame,
        valve_approach_frame,
        valve_contact_frame,
        valve_exit_angle: float = 0.0,
        use_ground_truth: bool = False,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.valve_exit_angle = valve_exit_angle

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # İlk state: ground truth modunda detection'ı atla
        first_state = "ENABLE_COARSE_APPROACH_PUBLISHER" if use_ground_truth else "ENABLE_FRONT_CAMERA_FOCUS"
        self.state_machine.set_initial_state([first_state])

        with self.state_machine:
            # 1-2: Detection states (ground truth modunda atlanır)
            if not use_ground_truth:
                smach.StateMachine.add(
                    "ENABLE_FRONT_CAMERA_FOCUS",
                    SetDetectionState(camera_name="front", enable=True),
                    transitions={
                        "succeeded": "FOCUS_ON_VALVE",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
                smach.StateMachine.add(
                    "FOCUS_ON_VALVE",
                    SetDetectionFocusState(focus_object="valve"),
                    transitions={
                        "succeeded": "ENABLE_COARSE_APPROACH_PUBLISHER",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )

            # =========================================================
            #  PHASE 1: COARSE APPROACH (oryantasyonsuz, robot→valve)
            # =========================================================

            # 3. Coarse approach frame yayınını başlat
            smach.StateMachine.add(
                "ENABLE_COARSE_APPROACH_PUBLISHER",
                ValveCoarseApproachFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "SET_VALVE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 4. Valve derinliğine in
            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth),
                transitions={
                    "succeeded": "FIND_AND_AIM_VALVE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 5. Valve'i bul ve yönünü ayarla
            smach.StateMachine.add(
                "FIND_AND_AIM_VALVE",
                SearchForPropState(
                    look_at_frame="valve_stand_link",
                    alignment_frame="valve_map_travel_start",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "PATH_TO_COARSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 6. Coarse approach frame'e git
            smach.StateMachine.add(
                "PATH_TO_COARSE_APPROACH",
                DynamicPathState(
                    plan_target_frame=valve_coarse_approach_frame,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_COARSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 7. Coarse approach frame'e hizalan
            smach.StateMachine.add(
                "ALIGN_TO_COARSE_APPROACH",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=valve_coarse_approach_frame,
                    dist_threshold=0.15,
                    yaw_threshold=0.15,
                    confirm_duration=3.0,
                    timeout=30.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_COARSE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 8. Coarse approach publisher'ı kapat
            smach.StateMachine.add(
                "DISABLE_COARSE_APPROACH_PUBLISHER",
                ValveCoarseApproachFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ENABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # =========================================================
            #  PHASE 2: ORIENTED APPROACH (yüzey normaline dik)
            # =========================================================

            # 9. Oriented approach publisher'ı aç
            smach.StateMachine.add(
                "ENABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 10. Approach frame'in oluşması için bekle
            smach.StateMachine.add(
                "WAIT_FOR_APPROACH_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 11. Approach frame'e hizalan (oryantasyonlu)
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
            # 12. Approach publisher'ı kapat
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
            #  PHASE 3: CONTACT (yüzey normaline dik, temas)
            # =========================================================

            # 13. Contact publisher'ı aç
            smach.StateMachine.add(
                "ENABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_CONTACT_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 14. Contact frame'in oluşması için bekle
            smach.StateMachine.add(
                "WAIT_FOR_CONTACT_FRAME",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "ALIGN_TO_CONTACT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 15. Contact frame'e hassas hizalama
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
            # 16. Contact publisher'ı kapat
            smach.StateMachine.add(
                "DISABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "ALIGN_TO_VALVE_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 17. Çıkış açısına dön
            smach.StateMachine.add(
                "ALIGN_TO_VALVE_EXIT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=valve_contact_frame,
                    angle_offset=self.valve_exit_angle,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=0.0,
                    timeout=10.0,
                    cancel_on_success=False,
                    use_frame_depth=False,
                ),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # 18. Controller'ı sıfırla
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
