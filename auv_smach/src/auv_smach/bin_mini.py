import smach
import rospy
from auv_smach.tf_utils import get_tf_buffer, get_base_link
from auv_smach.common import (
    SetDepthState,
    SearchForPropState,
    DynamicPathWithTransformAndVisibilityCheck,
    CancelAlignControllerState,
    SetDetectionState,
    SetDetectionFocusBottomState,
    SetDetectionFocusState,
    AlignFrameWithVisibilityCheck,
)
from auv_smach.initialize import DelayState
from std_msgs.msg import Float32


class BallDropperSetAngleState(smach.State):
    """
    for real life gripper).
    """

    def __init__(self, angle_value: int):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.pub = rospy.Publisher(
            "/taluy_mini/actuators/ball_dropper/set_angle", Float32, queue_size=1
        )
        self.angle_value = angle_value

    def execute(self, userdata) -> str:
        try:
            msg = Float32()
            msg.data = float(self.angle_value)
            for _ in range(3):
                self.pub.publish(msg)
                rospy.sleep(0.1)
            rospy.loginfo(
                f"[BallDropperSetAngleState] Published angle: {self.angle_value}"
            )
            return "succeeded"
        except Exception as e:
            rospy.logerr(f"[BallDropperSetAngleState] Error: {e}")
            return "aborted"


class BinTaskMiniState(smach.State):
    def __init__(
        self,
        bin_search_depth: float = -0.5,
        bin_drop_depth: float = -1.0,
        target_frames: list = ["bin_blood_link", "bin_fire_link"],
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = get_tf_buffer()
        self.base_link = get_base_link()
        self.bin_search_depth = bin_search_depth
        self.bin_drop_depth = bin_drop_depth
        self.target_animal = target_frames[0]
        self.bin_look_at_frame = "bin_basket_front_link"
        self.bin_alignment_frame = "bin_search"

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_FRONT_CAMERA",
                SetDetectionState(camera_name="front", enable=True),
                transitions={
                    "succeeded": "SET_FRONT_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_FRONT_FOCUS",
                SetDetectionFocusState(focus_object="bin"),
                transitions={
                    "succeeded": "ENABLE_BOTTOM_CAMERA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BOTTOM_CAMERA",
                SetDetectionState(camera_name="bottom", enable=True),
                transitions={
                    "succeeded": "SET_BOTTOM_FOCUS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_BOTTOM_FOCUS",
                SetDetectionFocusBottomState(focus_object="bin"),
                transitions={
                    "succeeded": "SET_SEARCH_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "SET_SEARCH_DEPTH",
                SetDepthState(
                    depth=-0.5,
                    max_velocity=0.2,
                ),
                transitions={
                    "succeeded": "FIND_AND_AIM_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "FIND_AND_AIM_BIN",
                SearchForPropState(
                    look_at_frame=self.bin_look_at_frame,
                    alignment_frame=self.bin_alignment_frame,
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=-0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_TO_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_FRAME_TO_BIN",
                DynamicPathWithTransformAndVisibilityCheck(
                    plan_target_frame=self.bin_look_at_frame,
                    transform_source_frame=self.base_link,
                    transform_target_frame=self.target_animal,
                    align_source_frame=self.base_link,
                    prop_name=self.bin_look_at_frame,
                    lost_timeout=1000.0,
                    transform_timeout=1000.0,
                    camera_name="bottom",
                    max_linear_velocity=0.15,
                    max_linear_velocity_y=0.025,
                    max_angular_velocity=1,
                ),
                transitions={
                    "succeeded": "ALIGN_PRECISELY_TO_BIN",
                    "target_lost": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_PRECISELY_TO_BIN",
                AlignFrameWithVisibilityCheck(
                    source_frame=self.base_link,
                    target_frame=self.target_animal,
                    prop_name=self.target_animal,
                    lost_timeout=100.0,
                    confirm_duration=10.0,
                    timeout=30.0,
                    cancel_on_success=True,
                    keep_orientation=True,
                    camera_name="bottom",
                    max_linear_velocity=0.05,
                ),
                transitions={
                    "succeeded": "DROP_BALL_1",
                    "target_lost": "CANCEL_ALIGN_CONTROLLER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_1",
                BallDropperSetAngleState(angle_value=55.0),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_1",
                DelayState(delay_time=5.0),
                transitions={
                    "succeeded": "DROP_BALL_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_2",
                BallDropperSetAngleState(angle_value=110.0),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_2",
                DelayState(delay_time=5.0),
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
        rospy.logdebug("[DropMarkerInBinMiniState] Starting state machine execution.")

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
