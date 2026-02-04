#!/usr/bin/env python3
import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest
from auv_smach.common import (
    AlignFrame,
    CancelAlignControllerState,
    DynamicPathWithTransformCheck,
    SearchForPropState,
    SetDepthState,
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
                    look_at_frame="docking_station_yolo",
                    alignment_frame="docking_approach_align",
                    full_rotation=False,
                    set_frame_duration=3.0,
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "APPROACH_UNTIL_ARUCO_DETECTED",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "APPROACH_UNTIL_ARUCO_DETECTED",
                DynamicPathWithTransformCheck(
                    plan_target_frame="docking_station_yolo",
                    transform_source_frame="odom",
                    transform_target_frame="docking_station",
                    max_linear_velocity=0.3,
                    transform_timeout=90.0,
                    keep_orientation=True,
                ),
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
                    confirm_duration=20.0,
                    timeout=40.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.2,
                    use_frame_depth=True,
                ),
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
