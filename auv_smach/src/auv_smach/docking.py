#!/usr/bin/env python3
import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest
from std_msgs.msg import Bool
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    SearchForPropState,
    SetAlignControllerTargetState,
    SetDepthState,
    SetDetectionFocusState,
    SetModelConfigState,
)
from auv_smach.initialize import DelayState


class WaitForBoardDetectionState(smach.State):
    def __init__(
        self,
        board_detected_topic: str = "/aruco_board_estimator/board_detected",
        rate_hz: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.board_detected_topic = board_detected_topic
        self.rate_hz = rate_hz
        self.board_detected = False

    def _board_detected_cb(self, msg):
        if msg.data:
            self.board_detected = True

    def execute(self, userdata):
        self.board_detected = False
        rate = rospy.Rate(self.rate_hz)

        board_sub = rospy.Subscriber(
            self.board_detected_topic, Bool, self._board_detected_cb
        )

        try:
            while not rospy.is_shutdown():
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                if self.board_detected:
                    rospy.loginfo("Board detected! Transitioning to Phase B.")
                    return "succeeded"

                rate.sleep()

        finally:
            board_sub.unregister()

        return "aborted"


class ToggleDockingApproachFrameState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_docking_approach_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class ToggleDockingTrajectoryState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "toggle_docking_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class DockingTaskState(smach.State):
    def __init__(
        self,
        search_depth: float = -0.3,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_MODEL_CONFIG",
                SetModelConfigState(model_name="tac_docking"),
                transitions={
                    "succeeded": "SET_DOCKING_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_DOCKING_FOCUS",
                SetDetectionFocusState(focus_object="docking"),
                transitions={
                    "succeeded": "SET_SEARCH_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_SEARCH_DEPTH",
                SetDepthState(depth=search_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SEARCH_FOR_STATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SEARCH_FOR_STATION",
                SearchForPropState(
                    look_at_frame="docking_station_link",
                    alignment_frame="docking_approach_align",
                    full_rotation=False,
                    set_frame_duration=3.0,
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "ENABLE_DOCKING_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ENABLE_DOCKING_APPROACH_FRAME",
                ToggleDockingApproachFrameState(req=True),
                transitions={
                    "succeeded": "START_APPROACH_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "START_APPROACH_ALIGN",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="docking_station_yolo",
                    max_linear_velocity=0.3,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_BOARD_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "WAIT_FOR_BOARD_DETECTION",
                WaitForBoardDetectionState(
                    board_detected_topic="/aruco_board_estimator/board_detected",
                    rate_hz=10.0,
                ),
                transitions={
                    "succeeded": "DISABLE_DOCKING_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "DISABLE_DOCKING_APPROACH_FRAME",
                ToggleDockingApproachFrameState(req=False),
                transitions={
                    "succeeded": "CANCEL_APPROACH_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_APPROACH_ALIGN",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "ENABLE_DOCKING_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ENABLE_DOCKING_TRAJECTORY",
                ToggleDockingTrajectoryState(req=True),
                transitions={
                    "succeeded": "ALIGN_ABOVE_STATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_ABOVE_STATION",
                AlignFrame(
                    source_frame="taluy/base_link/docking_puck_link",
                    target_frame="docking_approach_target",
                    angle_offset=1.5708,
                    dist_threshold=0.03,
                    yaw_threshold=0.04,
                    confirm_duration=1.0,
                    timeout=90.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "STOP_ESTIMATOR",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

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
                    "succeeded": "DOCK",
                },
            )

            smach.StateMachine.add(
                "DOCK",
                AlignFrame(
                    source_frame="taluy/base_link/docking_puck_link",
                    target_frame="docking_puck_target",
                    angle_offset=1.5708,
                    dist_threshold=0.03,
                    yaw_threshold=0.04,
                    confirm_duration=1.0,
                    timeout=90.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.2,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "WAIT_FOR_DOCK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "WAIT_FOR_DOCK",
                DelayState(delay_time=30.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CANCEL_ALIGN",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "DISABLE_DOCKING_TRAJECTORY",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "DISABLE_DOCKING_TRAJECTORY",
                ToggleDockingTrajectoryState(req=False),
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
