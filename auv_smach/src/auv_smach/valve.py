#!/usr/bin/env python3
"""
Valve SMACH Task
----------------
AUV'nin vana müdahale görevini yöneten durum makinesi.

Akış:
  1. Kamera detection'ı aç, vanaya odaklan
  2. Valve trajectory publisher'ı etkinleştir
  3. Görev derinliğine in
  4. Vanayı ara (SearchForPropState)
  5. Yaklaşma frame'ine git (DynamicPathState)
  6. Yaklaşma frame'ine hizalan (AlignFrame)
  7. Temas frame'ine git (DynamicPathState)
  8. Temas frame'ine hizalan (AlignFrame - hassas)
  9. Visual servoing ile son hizalama
  10. Vanayı çevir (TODO: mekanik aktuatör entegrasyonu)
  11. Temizlik: publisher'ları kapat

Kullanılan frame'ler:
  - valve_stand_link          → valve_detector.py tarafından yayınlanır
  - valve_approach_frame      → valve_trajectory_publisher.py tarafından yayınlanır
  - valve_contact_frame       → valve_trajectory_publisher.py tarafından yayınlanır
"""

from .initialize import *
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest
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


# =============================================================================
#  SERVİS STATE WRAPPER'LAR
# =============================================================================

class ValveApproachFramePublisherState(smach_ros.ServiceState):
    """valve_trajectory_publisher'daki approach frame yayınını aç/kapat."""
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_approach_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ValveContactFramePublisherState(smach_ros.ServiceState):
    """valve_trajectory_publisher'daki contact frame yayınını aç/kapat."""
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_valve_contact_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


# =============================================================================
#  ANA GÖREV STATE MAKİNESİ
# =============================================================================

class ValveTaskState(smach.State):
    def __init__(
        self,
        valve_depth: float,
        valve_approach_frame: str = "valve_approach_frame",
        valve_contact_frame: str = "valve_contact_frame",
    ):
        """
        Args:
            valve_depth:           Görev derinliği (metre, negatif)
            valve_approach_frame:  Uzak yaklaşma frame'i
            valve_contact_frame:   Temas frame'i
        """
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:

            # ── 1. ÖN KAMERA DETECTION'I AÇ ────────────────────────
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "ENABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # NOT: FOCUS_ON_VALVE kaldırıldı — valve detection ayrı bir HSV-tabanlı
            # node (valve_detector.py) üzerinden çalışıyor, camera_detection_pose_estimator
            # üzerinden geçmiyor. İleride YOLO tabanlı valve detection'a geçilirse
            # burada SetDetectionFocusState(focus_object="valve") tekrar eklenebilir.

            # ── 3. TRAJECTORY PUBLISHER'I ETKİNLEŞTİR ──────────────
            smach.StateMachine.add(
                "ENABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherState(req=True),
                transitions={
                    "succeeded": "SET_VALVE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 4. GÖREV DERİNLİĞİNE İN ────────────────────────────
            smach.StateMachine.add(
                "SET_VALVE_DEPTH",
                SetDepthState(depth=valve_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SEARCH_FOR_VALVE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 5. VANAYI ARA ───────────────────────────────────────
            smach.StateMachine.add(
                "SEARCH_FOR_VALVE",
                SearchForPropState(
                    look_at_frame="valve_stand_link",
                    alignment_frame=valve_approach_frame,
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "PATH_TO_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 6. YAKLAŞMA FRAME'İNE GİT ──────────────────────────
            smach.StateMachine.add(
                "PATH_TO_APPROACH",
                DynamicPathState(
                    plan_target_frame=valve_approach_frame,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 7. YAKLAŞMA FRAME'İNE HİZALAN ──────────────────────
            smach.StateMachine.add(
                "ALIGN_TO_APPROACH",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=valve_approach_frame,
                    dist_threshold=0.15,
                    yaw_threshold=0.1,
                    confirm_duration=3.0,
                    timeout=15.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_CONTACT_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 8. CONTACT FRAME PUBLISHER'I ETKİNLEŞTİR ───────────
            smach.StateMachine.add(
                "ENABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherState(req=True),
                transitions={
                    "succeeded": "PATH_TO_CONTACT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 9. TEMAS FRAME'İNE GİT ─────────────────────────────
            smach.StateMachine.add(
                "PATH_TO_CONTACT",
                DynamicPathState(
                    plan_target_frame=valve_contact_frame,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_CONTACT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 10. TEMAS FRAME'İNE HASSAS HİZALAN ─────────────────
            smach.StateMachine.add(
                "ALIGN_TO_CONTACT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame=valve_contact_frame,
                    dist_threshold=0.05,
                    yaw_threshold=0.05,
                    confirm_duration=3.0,
                    timeout=20.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "WAIT_BEFORE_ENGAGE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ── 11. TEMAS ÖNCESİ BEKLEME ───────────────────────────
            smach.StateMachine.add(
                "WAIT_BEFORE_ENGAGE",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": "CLEANUP",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ─────────────────────────────────────────────────────────
            # TODO: Buraya mekanik aktuatör / gripper state'leri eklenecek
            #   - ENGAGE_GRIPPER → Sapı kavra
            #   - TURN_VALVE     → Motor tork uygula (90° çevir)
            #   - RELEASE        → Sapı bırak
            #   - VERIFY_TURN    → Geri çekilip sapın dönmüş olduğunu doğrula
            # ─────────────────────────────────────────────────────────

            # ── 12. TEMİZLİK ────────────────────────────────────────
            smach.StateMachine.add(
                "CLEANUP",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "DISABLE_APPROACH_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_APPROACH_PUBLISHER",
                ValveApproachFramePublisherState(req=False),
                transitions={
                    "succeeded": "DISABLE_CONTACT_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_CONTACT_PUBLISHER",
                ValveContactFramePublisherState(req=False),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo("=== VALVE TASK STARTED ===")
        outcome = self.state_machine.execute()
        rospy.loginfo(f"=== VALVE TASK FINISHED: {outcome} ===")
        return outcome
