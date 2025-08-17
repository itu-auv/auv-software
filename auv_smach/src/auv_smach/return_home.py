from .initialize import *
import smach
import smach_ros
import rospy
import tf2_ros
from auv_navigation.path_planning.path_planners import PathPlanners
from geometry_msgs.msg import PoseStamped
from auv_smach.common import (
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
    AlignFrame,
    DynamicPathState,
)
from auv_smach.acoustic import AcousticTransmitter


class NavigateReturnThroughGateState(smach.State):
    def __init__(self, station_frame: str = "bin_whole_link"):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.gate_look_at_frame = "gate_middle_part"
        self.station_frame = station_frame

        # Initialize the state machine container
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            # smach.StateMachine.add(
            #     "SET_RETURN_DEPTH",
            #     SetDepthState(
            #         depth=-1.2,
            #         sleep_duration=rospy.get_param("~set_depth_sleep_duration", 3.0),
            #     ),
            #     transitions={
            #         "succeeded": "LOOK_AT_STATION",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )
            smach.StateMachine.add(
                "LOOK_AT_STATION",
                SearchForPropState(
                    look_at_frame=self.station_frame,
                    alignment_frame="look_at_station",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_STATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_STATION",
                DynamicPathState(
                    plan_target_frame=self.station_frame,
                    angle_offset=1.57,
                ),
                transitions={
                    "succeeded": "LOOK_AT_GATE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "LOOK_AT_GATE",
                SearchForPropState(
                    look_at_frame=self.gate_look_at_frame,
                    alignment_frame="look_at_gate",
                    full_rotation=False,
                    set_frame_duration=7.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "DYNAMIC_PATH_TO_GATE_RETURN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_TO_GATE_RETURN",
                DynamicPathState(
                    plan_target_frame=self.gate_look_at_frame,
                    angle_offset=3.14,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_ENTRANCE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_ENTRANCE",
                AlignFrame(
                    source_frame="taluy/base_link",
                    target_frame="gate_entrance",
                    angle_offset=3.14,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=2.0,
                    timeout=10.0,
                    cancel_on_success=False,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "TRANSMIT_ACOUSTIC_7",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "TRANSMIT_ACOUSTIC_7",
                AcousticTransmitter(acoustic_data=7),
                transitions={
                    "succeeded": "FINISHED_ROBOSUB",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FINISHED_ROBOSUB",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.logdebug(
            "[NavigateReturnThroughGateState] Starting state machine execution."
        )

        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"
        return outcome
