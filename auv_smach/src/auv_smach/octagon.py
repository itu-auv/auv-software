from .initialize import *
import smach
import smach_ros
from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SetDetectionFocusState,
    DynamicPathState,
    DynamicPathWithTransformCheck,
    AlignFrame,
    SearchForPropState,
    SetDetectionState,
    AlignAndCreateRotatingFrame,
)
from auv_smach.initialize import DelayState
from auv_smach.acoustic import AcousticTransmitter
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from std_msgs.msg import UInt16
import tf2_ros


class GripperAngleOpenState(smach.State):
    """
    Gerçek gripper için angle topic'ine 2400 değerini yayınlar (açık pozisyon).
    """

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.pub = rospy.Publisher("actuators/gripper/angle", UInt16, queue_size=1)
        self.angle_value = 2400

    def execute(self, userdata) -> str:
        try:
            msg = UInt16()
            msg.data = self.angle_value
            # Birkaç kez yayınla - topic'in alındığından emin olmak için
            for _ in range(3):
                self.pub.publish(msg)
                rospy.sleep(0.1)
            rospy.loginfo(
                f"[GripperAngleOpenState] Published angle: {self.angle_value}"
            )
            return "succeeded"
        except Exception as e:
            rospy.logerr(f"[GripperAngleOpenState] Error: {e}")
            return "aborted"


class GripperAngleCloseState(smach.State):
    """
    Gerçek gripper için angle topic'ine 400 değerini yayınlar (kapalı pozisyon).
    """

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.pub = rospy.Publisher("actuators/gripper/angle", UInt16, queue_size=1)
        self.angle_value = 400

    def execute(self, userdata) -> str:
        try:
            msg = UInt16()
            msg.data = self.angle_value
            # Birkaç kez yayınla - topic'in alındığından emin olmak için
            for _ in range(3):
                self.pub.publish(msg)
                rospy.sleep(0.1)
            rospy.loginfo(
                f"[GripperAngleCloseState] Published angle: {self.angle_value}"
            )
            return "succeeded"
        except Exception as e:
            rospy.logerr(f"[GripperAngleCloseState] Error: {e}")
            return "aborted"


class CheckBottleLinkState(smach.State):
    """
    State to check if bottle_link transform is available.
    Returns 'succeeded' if found, 'aborted' if timeout.
    """

    def __init__(
        self,
        source_frame: str = "odom",
        target_frame: str = "bottle_link",
        timeout: float = 3.0,
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def execute(self, userdata) -> str:
        import rospy

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while (rospy.Time.now() - start_time) < self.timeout:
            if self.preempt_requested():
                return "preempted"

            if self.tf_buffer.can_transform(
                self.source_frame, self.target_frame, rospy.Time(0), rospy.Duration(0.5)
            ):
                rospy.loginfo(
                    f"[CheckBottleLinkState] Transform from '{self.source_frame}' to '{self.target_frame}' found."
                )
                return "succeeded"

            rate.sleep()

        rospy.logwarn(
            f"[CheckBottleLinkState] Timeout: '{self.target_frame}' transform not found after {self.timeout.to_sec()} seconds."
        )
        return "aborted"


class MoveGripperServiceState(smach_ros.ServiceState):
    def __init__(self):
        from std_srvs.srv import Trigger, TriggerRequest

        smach_ros.ServiceState.__init__(
            self,
            "actuators/gripper/close",
            Trigger,
            request=TriggerRequest(),
            response_slots=[],
        )


class OpenGripperServiceState(smach_ros.ServiceState):
    def __init__(self):
        from std_srvs.srv import Trigger, TriggerRequest

        smach_ros.ServiceState.__init__(
            self,
            "actuators/gripper/open",
            Trigger,
            request=TriggerRequest(),
            response_slots=[],
        )


class OctagonFramePublisherServiceState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_octagon_frame",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class OctagonTaskState(smach.State):
    def __init__(self, octagon_depth: float, animal: str):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.griper_mode = True
        self.animal_frame = f"gate_{animal}_link"
        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_INITIAL_DEPTH",
                SetDepthState(depth=-1.2, sleep_duration=4.0),
                transitions={
                    "succeeded": "FOCUS_ON_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOCUS_ON_OCTAGON",
                SetDetectionFocusState(focus_object="octagon"),
                transitions={
                    "succeeded": "FIND_AIM_OCTAGON",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AIM_OCTAGON",
                SearchForPropState(
                    look_at_frame="octagon_link",
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ENABLE_OCTAGON_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_OCTAGON_FRAME_PUBLISHER",
                OctagonFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "WAIT_FOR_OCTAGON_FRAME",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_OCTAGON_FRAME",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "CLOSE_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_DETECTION",
                SetDetectionFocusState(focus_object="none"),
                transitions={
                    "succeeded": "CLOSE_OCTAGON_PUBLİSHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_OCTAGON_PUBLİSHER",
                OctagonFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "SET_OCTAGON_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(depth=octagon_depth, sleep_duration=4.0),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_CLOSE_APPROACH",
                DynamicPathState(
                    plan_target_frame="octagon_closer_link",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_CLOSE_APPROACH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_CLOSE_APPROACH",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_closer_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_BOTTOM_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BOTTOM_DETECTION",
                SetDetectionState(camera_name="bottom", enable=True),
                transitions={
                    "succeeded": "DYNAMIC_PATH_WITH_BOTTLE_CHECK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_WITH_BOTTLE_CHECK",
                DynamicPathWithTransformCheck(
                    plan_target_frame="octagon_link",
                    transform_source_frame="odom",
                    transform_target_frame="octagon_table_link",
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "SEARCH_RIGHT",
                },
            )
            smach.StateMachine.add(
                "MOVE_GRIPPER",
                GripperAngleOpenState(),
                transitions={
                    "succeeded": "ALIGN_TO_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_BOTTLE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_table_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    keep_orientation=True,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "a",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "a",
                AlignFrame(
                    source_frame="taluy/gripper_link",
                    target_frame="bottle_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=60.0,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "m",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "m",
                SetDetectionState(camera_name="bottom", enable=False),
                transitions={
                    "succeeded": "SET_BOTTLE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_BOTTLE_DEPTH",
                SetDepthState(depth=-1.2, sleep_duration=15.0),
                transitions={
                    "succeeded": "SURFACE_WITH_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "CLOSE_GRIPPER",
                GripperAngleCloseState(),
                transitions={
                    "succeeded": "SURFACE_WITH_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # ============== BOTTLE SEARCH SEQUENCE ==============
            # Search Right
            smach.StateMachine.add(
                "SEARCH_RIGHT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_search_right",
                    angle_offset=0.0,
                    dist_threshold=0.15,
                    yaw_threshold=0.15,
                    confirm_duration=2.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CHECK_BOTTLE_AFTER_RIGHT",
                    "preempted": "preempted",
                    "aborted": "CHECK_BOTTLE_AFTER_RIGHT",  # Continue to check even if align fails
                },
            )
            smach.StateMachine.add(
                "CHECK_BOTTLE_AFTER_RIGHT",
                CheckBottleLinkState(
                    source_frame="odom", target_frame="bottle_link", timeout=2.0
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "SEARCH_FORWARD",
                },
            )

            # Search Forward
            smach.StateMachine.add(
                "SEARCH_FORWARD",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_search_forward",
                    angle_offset=0.0,
                    dist_threshold=0.15,
                    yaw_threshold=0.15,
                    confirm_duration=2.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CHECK_BOTTLE_AFTER_FORWARD",
                    "preempted": "preempted",
                    "aborted": "CHECK_BOTTLE_AFTER_FORWARD",
                },
            )
            smach.StateMachine.add(
                "CHECK_BOTTLE_AFTER_FORWARD",
                CheckBottleLinkState(
                    source_frame="odom", target_frame="bottle_link", timeout=2.0
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "SEARCH_LEFT",
                },
            )

            # Search Left
            smach.StateMachine.add(
                "SEARCH_LEFT",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_search_left",
                    angle_offset=0.0,
                    dist_threshold=0.15,
                    yaw_threshold=0.15,
                    confirm_duration=2.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CHECK_BOTTLE_AFTER_LEFT",
                    "preempted": "preempted",
                    "aborted": "CHECK_BOTTLE_AFTER_LEFT",
                },
            )
            smach.StateMachine.add(
                "CHECK_BOTTLE_AFTER_LEFT",
                CheckBottleLinkState(
                    source_frame="odom", target_frame="bottle_link", timeout=2.0
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "SEARCH_BACKWARD",
                },
            )

            # Search Backward
            smach.StateMachine.add(
                "SEARCH_BACKWARD",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="octagon_search_backward",
                    angle_offset=0.0,
                    dist_threshold=0.15,
                    yaw_threshold=0.15,
                    confirm_duration=2.0,
                    timeout=30.0,
                    cancel_on_success=False,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "CHECK_BOTTLE_AFTER_BACKWARD",
                    "preempted": "preempted",
                    "aborted": "CHECK_BOTTLE_AFTER_BACKWARD",
                },
            )
            smach.StateMachine.add(
                "CHECK_BOTTLE_AFTER_BACKWARD",
                CheckBottleLinkState(
                    source_frame="odom", target_frame="bottle_link", timeout=2.0
                ),
                transitions={
                    "succeeded": "ALIGN_TO_BOTTLE",
                    "preempted": "preempted",
                    "aborted": "SEARCH_RIGHT",  # Loop back to start (or could abort)
                },
            )
            # ============== END BOTTLE SEARCH SEQUENCE ==============

            smach.StateMachine.add(
                "SURFACE_WITH_BOTTLE",
                SetDepthState(depth=-0.3, sleep_duration=5.0),  # Close to surface
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACE_TO_ANIMAL_DEPTH",
                SetDepthState(depth=-0.43, sleep_duration=4.0),
                transitions={
                    "succeeded": "SET_DETECTION_FOCUS_TO_ANIMALS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_DETECTION_FOCUS_TO_ANIMALS",
                SetDetectionFocusState(focus_object="gate"),
                transitions={
                    "succeeded": "ROTATE_FOR_ANIMALS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ROTATE_FOR_ANIMALS",
                AlignAndCreateRotatingFrame(
                    source_frame="taluy/base_link",
                    target_frame="animal_search_frame",
                    rotating_frame_name="animal_search_frame",
                ),
                transitions={
                    "succeeded": "b",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "b",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "c",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "c",
                SearchForPropState(
                    look_at_frame=self.animal_frame,
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    set_frame_duration=5.0,
                    source_frame="taluy/base_link",
                    rotation_speed=-0.2,
                ),
                transitions={
                    "succeeded": "SURFACING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACING",
                SetDepthState(depth=-0.05, sleep_duration=7.0),
                transitions={
                    "succeeded": "SET_FINAL_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FINAL_DEPTH",
                SetDepthState(depth=octagon_depth, sleep_duration=4.0),
                transitions={
                    "succeeded": "TRANSMIT_ACOUSTIC_5",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TRANSMIT_ACOUSTIC_5",
                AcousticTransmitter(acoustic_data=5),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
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
        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
