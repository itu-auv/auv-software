#!/usr/bin/env python3
import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SearchForPropState,
    SetAlignControllerTargetState,
    SetDepthState,
    SetDetectionFocusState,
    SetIdRemapState,
)
from auv_smach.initialize import DelayState


class DockingTaskState(smach.State):
    """
    Main state machine for the Subsea Docking Mission.

    Workflow:
    Phase A - Approach (Front Camera with YOLO):
    - SET_DOCKING_FOCUS: Set detection focus to docking board
    - SET_SEARCH_DEPTH: Set depth for optimal front camera visibility
    - SEARCH_FOR_DOCKING: Rotate until docking_board_link TF is found,
      then set alignment frame facing the docking board
    - APPROACH_DOCKING: Align toward docking_board_link until
      docking_station TF (ArUco) appears

    Phase B - ArUco Precision Docking (Bottom Camera):
    - ALIGN_APPROACH: Align puck to approach target (1m above docking station)
    - STOP_ESTIMATOR: Stop the pose estimator (Kalman filter holds pose)
    - ALIGN_FINAL: Descend and dock with precision alignment
    - WAIT_FOR_DOCK: Hold position to demonstrate docking
    - CANCEL_ALIGN: Cleanup
    """

    def __init__(
        self,
        search_depth: float = -0.8,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            # ============================================================
            # PHASE A: Approach (Front Camera)
            # Uses SearchForPropState + AlignFrameUntilTF from common.py
            # ============================================================

            # Set ID remapping for docking model (class 0 -> internal ID 8)
            smach.StateMachine.add(
                "SET_ID_REMAP",
                SetIdRemapState(id_remap_json='{"0": 8}'),
                transitions={
                    "succeeded": "SET_DOCKING_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Set detection focus to docking board
            smach.StateMachine.add(
                "SET_DOCKING_FOCUS",
                SetDetectionFocusState(focus_object="docking"),
                transitions={
                    "succeeded": "SET_SEARCH_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Set depth for searching (can see docking board from distance)
            smach.StateMachine.add(
                "SET_SEARCH_DEPTH",
                SetDepthState(depth=search_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SEARCH_FOR_DOCKING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Rotate until docking board is detected in camera
            smach.StateMachine.add(
                "SEARCH_FOR_DOCKING",
                SearchForPropState(
                    look_at_frame="docking_board_link",
                    alignment_frame="docking_approach_align",
                    full_rotation=False,
                    set_frame_duration=3.0,
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "START_APPROACH_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Start alignment toward docking board
            smach.StateMachine.add(
                "START_APPROACH_ALIGN",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="docking_board_link",
                    max_linear_velocity=0.3,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_BOARD_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Wait for board detection signal (alignment continues in background)
            def board_detected_cb(userdata, msg):
                if msg.data:
                    rospy.loginfo("Board detected via signal!")
                    return False  # Terminate MonitorState
                return True  # Keep monitoring

            smach.StateMachine.add(
                "WAIT_FOR_BOARD_DETECTION",
                smach_ros.MonitorState(
                    "/aruco_board_estimator/board_detected",
                    Bool,
                    board_detected_cb,
                ),
                transitions={
                    "invalid": "CANCEL_APPROACH_ALIGN",
                    "valid": "CANCEL_APPROACH_ALIGN",
                    "preempted": "preempted",
                },
            )

            # Cancel alignment after approach
            smach.StateMachine.add(
                "CANCEL_APPROACH_ALIGN",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "ALIGN_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ============================================================
            # PHASE B: ArUco Precision Docking (Bottom Camera)
            # ============================================================

            # Align to approach position (XY + Z, 1m above board)
            smach.StateMachine.add(
                "ALIGN_APPROACH",
                AlignFrame(
                    source_frame="taluy/base_link/docking_puck_link",
                    target_frame="docking_approach_target",
                    angle_offset=1.5708,
                    dist_threshold=0.03,
                    yaw_threshold=0.04,
                    confirm_duration=1.0,
                    timeout=90.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    align_z=True,
                    z_threshold=0.03,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "STOP_ESTIMATOR",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Stop the pose estimator (Kalman filter will hold the last pose)
            def disable_estimator_cb(userdata):
                try:
                    srv = rospy.ServiceProxy(
                        "/aruco_board_estimator/set_enabled", SetBool
                    )
                    srv.wait_for_service(timeout=5.0)
                    srv(False)
                    rospy.loginfo("Disabled ArUco estimator")
                except rospy.ROSException as e:
                    rospy.logwarn(f"Failed to disable estimator: {e}")
                return "succeeded"

            smach.StateMachine.add(
                "STOP_ESTIMATOR",
                smach.CBState(disable_estimator_cb, outcomes=["succeeded"]),
                transitions={
                    "succeeded": "ALIGN_FINAL",
                },
            )

            # Descend and dock (XY + Z precision)
            smach.StateMachine.add(
                "ALIGN_FINAL",
                AlignFrame(
                    source_frame="taluy/base_link/docking_puck_link",
                    target_frame="docking_puck_target",
                    angle_offset=1.5708,
                    dist_threshold=0.03,
                    yaw_threshold=0.04,
                    confirm_duration=1.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    align_z=True,
                    z_threshold=0.02,
                    max_linear_velocity=0.07,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_DOCK",
                    "preempted": "preempted",
                    "aborted": "WAIT_FOR_DOCK",
                },
            )

            # Wait/Show Docking
            smach.StateMachine.add(
                "WAIT_FOR_DOCK",
                DelayState(delay_time=10.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Cleanup
            smach.StateMachine.add(
                "CANCEL_ALIGN",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "RISE_AFTER_DOCK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            
            smach.StateMachine.add(
                "RISE_AFTER_DOCK",
                SetDepthState(depth=search_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        return self.state_machine.execute(userdata)
