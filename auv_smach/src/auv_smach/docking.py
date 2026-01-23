#!/usr/bin/env python3
import math

import numpy as np
import rospy
import smach
import smach_ros
import tf2_ros
import tf.transformations as transformations
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
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
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class PublishApproachFrameState(smach.State):
    """
    Continuously publishes an approach frame at the target's position
    with orientation facing FROM the source (AUV) TO the target.

    This allows the AUV to approach the target while facing it,
    eliminating zig-zag behavior caused by orientation mismatches.

    The state monitors for board detection and returns when detected.
    """

    def __init__(
        self,
        source_frame: str,
        target_frame: str,
        approach_frame: str,
        reference_frame: str = "odom",
        board_detected_topic: str = "/aruco_board_estimator/board_detected",
        rate_hz: float = 10.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.approach_frame = approach_frame
        self.reference_frame = reference_frame
        self.board_detected_topic = board_detected_topic
        self.rate_hz = rate_hz

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        self.board_detected = False

    def _board_detected_cb(self, msg):
        if msg.data:
            self.board_detected = True

    def execute(self, userdata):
        self.board_detected = False
        rate = rospy.Rate(self.rate_hz)

        # Subscribe to board detection
        board_sub = rospy.Subscriber(
            self.board_detected_topic, Bool, self._board_detected_cb
        )

        try:
            while not rospy.is_shutdown():
                # Check for preemption
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                # Check if board detected
                if self.board_detected:
                    rospy.loginfo("Board detected! Transitioning to Phase B.")
                    return "succeeded"

                # Look up transforms
                try:
                    source_transform = self.tf_buffer.lookup_transform(
                        self.reference_frame,
                        self.source_frame,
                        rospy.Time(0),
                        rospy.Duration(0.5),
                    )
                    target_transform = self.tf_buffer.lookup_transform(
                        self.reference_frame,
                        self.target_frame,
                        rospy.Time(0),
                        rospy.Duration(0.5),
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logwarn_throttle(
                        1.0, f"PublishApproachFrameState: TF lookup failed: {e}"
                    )
                    rate.sleep()
                    continue

                # Get positions
                source_pos = np.array(
                    [
                        source_transform.transform.translation.x,
                        source_transform.transform.translation.y,
                    ]
                )
                target_pos = np.array(
                    [
                        target_transform.transform.translation.x,
                        target_transform.transform.translation.y,
                    ]
                )

                # Compute direction from source (AUV) to target (board)
                direction = target_pos - source_pos
                distance = np.linalg.norm(direction)

                if distance > 0.01:  # Avoid division by zero
                    # Compute yaw angle facing toward the target
                    facing_angle = math.atan2(direction[1], direction[0])
                else:
                    # Too close, keep current orientation
                    facing_angle = 0.0

                # Create quaternion from yaw
                quaternion = transformations.quaternion_from_euler(0, 0, facing_angle)

                # Create and publish the approach frame
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.reference_frame
                t.child_frame_id = self.approach_frame
                # Position at target
                t.transform.translation.x = target_transform.transform.translation.x
                t.transform.translation.y = target_transform.transform.translation.y
                t.transform.translation.z = target_transform.transform.translation.z
                # Orientation facing from source to target
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]

                try:
                    req = SetObjectTransformRequest()
                    req.transform = t
                    self.set_object_transform_service(req)
                except rospy.ServiceException as e:
                    rospy.logwarn_throttle(
                        1.0, f"PublishApproachFrameState: Service call failed: {e}"
                    )

                rate.sleep()

        finally:
            board_sub.unregister()

        return "aborted"


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
        search_depth: float = -0.3,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            # ============================================================
            # PHASE A: Approach (Front Camera)
            # ============================================================

            # Set model configuration for docking model
            smach.StateMachine.add(
                "SET_MODEL_CONFIG",
                SetModelConfigState(model_name="tac_docking"),
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

            # Start alignment toward the approach frame (will be published by next state)
            smach.StateMachine.add(
                "START_APPROACH_ALIGN",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link",
                    target_frame="docking_board_approach",
                    max_linear_velocity=0.3,
                    keep_orientation=False,  # Now we can use the frame's orientation
                ),
                transitions={
                    "succeeded": "PUBLISH_APPROACH_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # Continuously publish approach frame and wait for board detection
            smach.StateMachine.add(
                "PUBLISH_APPROACH_FRAME",
                PublishApproachFrameState(
                    source_frame="taluy/base_link",
                    target_frame="docking_board_link",
                    approach_frame="docking_board_approach",
                    reference_frame="odom",
                    board_detected_topic="/aruco_board_estimator/board_detected",
                    rate_hz=10.0,
                ),
                transitions={
                    "succeeded": "CANCEL_APPROACH_ALIGN",
                    "preempted": "preempted",
                    "aborted": "aborted",
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

            # Align to approach position (XY + yaw + depth, 1m above board)
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

            # Align to final docking position (XY + yaw + depth)
            smach.StateMachine.add(
                "ALIGN_FINAL",
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

            # Wait/Show Docking
            smach.StateMachine.add(
                "WAIT_FOR_DOCK",
                DelayState(delay_time=30.0),
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
