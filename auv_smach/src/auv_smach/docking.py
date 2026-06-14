#!/usr/bin/env python3
import rospy
import smach
import smach_ros
from std_srvs.srv import SetBool, SetBoolRequest
import math
from auv_smach.common import (
    AlignFrame,
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
        search_depth: float = -1.0,
        test_mode: bool = False,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # In test mode we assume the docking board is already visible from the
        # start, so we skip the search/approach phase (which would otherwise
        # require driving up to the station) and jump straight to the docking
        # trajectory after only setting the search depth.
        self.test_mode = test_mode

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:

            def init_docking_cameras_cb(userdata):
                try:
                    torpedo_srv = rospy.ServiceProxy(
                        "/aruco_detection_node/torpedo/set_enabled", SetBool
                    )
                    bottom_srv = rospy.ServiceProxy(
                        "/aruco_detection_node/bottom/set_enabled", SetBool
                    )
                    torpedo_srv.wait_for_service(timeout=5.0)
                    bottom_srv.wait_for_service(timeout=5.0)
                    torpedo_srv(False)
                    bottom_srv(True)
                    rospy.loginfo(
                        "Initialized ArUco detection at docking start: torpedo OFF, bottom ON"
                    )
                except rospy.ROSException as e:
                    rospy.logwarn(f"Failed to initialize docking ArUco cameras: {e}")
                return "succeeded"

            smach.StateMachine.add(
                "DISABLE_TORPEDO_ARUCO",
                smach.CBState(init_docking_cameras_cb, outcomes=["succeeded"]),
                transitions={
                    "succeeded": "SET_SEARCH_DEPTH",
                },
            )

            smach.StateMachine.add(
                "SET_SEARCH_DEPTH",
                SetDepthState(depth=search_depth, timeout=10.0),
                transitions={
                    "succeeded": (
                        "ENABLE_DOCKING_TRAJECTORY"
                        if self.test_mode
                        else "SEARCH_FOR_STATION"
                    ),
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
                ),
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
                    target_frame="docking_station",
                    angle_offset=math.pi / 2,
                    dist_threshold=0.05,
                    yaw_threshold=0.04,
                    confirm_duration=1.0,
                    timeout=50.0,
                    cancel_on_success=True,
                    keep_orientation=False,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    use_frame_depth=False,
                ),
                transitions={
                    "succeeded": "STOP_ESTIMATOR",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            def switch_to_torpedo_cam_cb(userdata):
                try:
                    bottom_srv = rospy.ServiceProxy(
                        "/aruco_detection_node/bottom/set_enabled", SetBool
                    )
                    torpedo_srv = rospy.ServiceProxy(
                        "/aruco_detection_node/torpedo/set_enabled", SetBool
                    )
                    bottom_srv.wait_for_service(timeout=5.0)
                    torpedo_srv.wait_for_service(timeout=5.0)
                    bottom_srv(False)
                    torpedo_srv(True)
                    rospy.loginfo("Switched ArUco detection: bottom OFF, torpedo ON")
                except rospy.ROSException as e:
                    rospy.logwarn(f"Failed to switch cameras: {e}")
                return "succeeded"

            smach.StateMachine.add(
                "STOP_ESTIMATOR",
                smach.CBState(switch_to_torpedo_cam_cb, outcomes=["succeeded"]),
                transitions={
                    "succeeded": "ALIGN_PRE_TOUCH",
                },
            )

            smach.StateMachine.add(
                "ALIGN_PRE_TOUCH",
                AlignFrame(
                    source_frame="taluy/base_link/docking_puck_link",
                    target_frame="docking_pre_touch_target",
                    angle_offset=math.pi / 2,
                    dist_threshold=0.03,
                    yaw_threshold=0.04,
                    confirm_duration=3.0,
                    timeout=90.0,
                    cancel_on_success=True,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.2,
                    use_frame_depth=True,
                ),
                transitions={
                    "succeeded": "DOCK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "DOCK",
                AlignFrame(
                    source_frame="taluy/base_link/docking_puck_link",
                    target_frame="docking_puck_target",
                    angle_offset=math.pi / 2,
                    dist_threshold=0.03,
                    yaw_threshold=0.04,
                    confirm_duration=20.0,
                    timeout=40.0,
                    cancel_on_success=True,
                    keep_orientation=True,
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
                SetDepthState(depth=search_depth, timeout=3.0),
                transitions={
                    "succeeded": "RESTART_ESTIMATOR",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            def switch_to_bottom_cam_cb(userdata):
                try:
                    torpedo_srv = rospy.ServiceProxy(
                        "/aruco_detection_node/torpedo/set_enabled", SetBool
                    )
                    bottom_srv = rospy.ServiceProxy(
                        "/aruco_detection_node/bottom/set_enabled", SetBool
                    )
                    torpedo_srv.wait_for_service(timeout=5.0)
                    bottom_srv.wait_for_service(timeout=5.0)
                    torpedo_srv(False)
                    bottom_srv(True)
                    rospy.loginfo("Switched ArUco detection: torpedo OFF, bottom ON")
                except rospy.ROSException as e:
                    rospy.logwarn(f"Failed to switch cameras: {e}")
                return "succeeded"

            smach.StateMachine.add(
                "RESTART_ESTIMATOR",
                smach.CBState(switch_to_bottom_cam_cb, outcomes=["succeeded"]),
                transitions={
                    "succeeded": "succeeded",
                },
            )

        if self.test_mode:
            # Start from SET_SEARCH_DEPTH, skipping DISABLE_TORPEDO_ARUCO so the
            # only state run before the docking trajectory is the search depth.
            rospy.logwarn(
                "[DockingTaskState] test_mode enabled: skipping search/approach, "
                "starting from SET_SEARCH_DEPTH (board assumed already visible)"
            )
            self.state_machine.set_initial_state(["SET_SEARCH_DEPTH"])

    def execute(self, userdata):
        return self.state_machine.execute(userdata)
