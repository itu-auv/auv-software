from .initialize import *
import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
import tf2_ros

from auv_smach.common import (
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SearchForPropState,
)

from auv_smach.initialize import DelayState

from auv_smach.common import (
    DropBallState,
    ExecutePlannedPathsState,
    CancelAlignControllerState,
    PlanPathToSingleFrameState,
)

from auv_navigation.path_planning.path_planners import PathPlanners


class CheckForDropAreaState(smach.State):
    def __init__(self, source_frame: str = "odom", timeout: float = 2.0):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["found_frame"],
        )
        self.source_frame = source_frame
        self.timeout = rospy.Duration(timeout)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_frames = ["bin/blue_link", "bin/red_link"]

    def execute(self, userdata) -> str:
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while (rospy.Time.now() - start_time) < self.timeout:
            if self.preempt_requested():
                return "preempted"

            # Check for both blue and red bin frames
            for frame in self.target_frames:
                if self.tf_buffer.can_transform(
                    self.source_frame, frame, rospy.Time(0), self.timeout
                ):
                    rospy.loginfo(
                        f"[CheckForDropAreaState] Transform from '{self.source_frame}' to '{frame}' found."
                    )
                    userdata.found_frame = frame
                    return "succeeded"

            rate.sleep()

        rospy.logwarn(
            f"[CheckForDropAreaState] Timeout: No drop area transforms found after {self.timeout.to_sec()} seconds."
        )
        return "aborted"


class SetAlignToFoundState(smach.State):
    def __init__(self, source_frame: str):
        super().__init__(
            outcomes=["succeeded", "preempted", "aborted"], input_keys=["found_frame"]
        )
        self.source_frame = source_frame

    def execute(self, userdata):
        if self.preempt_requested():
            return "preempted"

        if "found_frame" not in userdata or not userdata.found_frame:
            rospy.logerr("[SetAlignToFoundState] No found_frame in userdata")
            return "aborted"

        rospy.loginfo(
            f"[SetAlignToFoundState] Setting align target to {userdata.found_frame}"
        )

        align_state = SetAlignControllerTargetState(
            source_frame=self.source_frame,
            target_frame=userdata.found_frame,
            keep_orientation=True,
        )
        return align_state.execute(userdata)


class BinSecondTrialAttemptState(smach.StateMachine):
    def __init__(self, tf_buffer):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )
        self.tf_buffer = tf_buffer

        with self:
            smach.StateMachine.add(
                "PLAN_PATH_TO_BIN_SECOND_TRIAL",
                PlanPathToSingleFrameState(
                    tf_buffer=self.tf_buffer,
                    target_frame="bin_second_trial",
                    source_frame="taluy/base_link",
                ),
                transitions={
                    "succeeded": "EXECUTE_BIN_PATH_TO_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_BIN_PATH_TO_SECOND_TRIAL",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_FOUND_TO_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_FOUND_TO_SECOND_TRIAL",
                CheckForDropAreaState(source_frame="odom", timeout=2.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "ENABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                },
            )
            smach.StateMachine.add(
                "ENABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                BinTransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "CANCEL_ALIGN_FOR_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_FOR_SEARCH",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "FIND_AND_AIM_BIN_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_BIN_SECOND_TRIAL",
                SearchForPropState(
                    look_at_frame="bin_whole_link",
                    alignment_frame="bin_search",
                    full_rotation=True,
                    set_frame_duration=4.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "PLAN_BIN_PATHS_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_BIN_PATHS_SECOND_TRIAL",
                PlanBinPathState(self.tf_buffer),
                transitions={
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BIN_FRAME_PUBLISHER_SECOND_TRIAL",
                BinTransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_SECOND_TRIAL",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="dynamic_target"
                ),
                transitions={
                    "succeeded": "EXECUTE_BIN_PATH_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_BIN_PATH_SECOND_TRIAL",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_FOUND_SECOND_TRIAL",
                    "preempted": "CANCEL_ALIGN_CONTROLLER_SECOND_TRIAL",
                    "aborted": "CANCEL_ALIGN_CONTROLLER_SECOND_TRIAL",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_FOUND_SECOND_TRIAL",
                CheckForDropAreaState(source_frame="odom", timeout=2.0),
                transitions={
                    "succeeded": "CANCEL_ALIGN_CONTROLLER_SECOND_TRIAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_ALIGN_CONTROLLER_SECOND_TRIAL",
                CancelAlignControllerState(),
                transitions={
                    "succeeded": "aborted",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class BinTransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_bin_frames",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class PlanBinPathState(smach.State):
    def __init__(self, tf_buffer):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["planned_paths"],
        )
        self.tf_buffer = tf_buffer

    def execute(self, userdata) -> str:
        try:
            if self.preempt_requested():
                rospy.logwarn("[PlanTorpedoPathState] Preempt requested")
                return "preempted"

            path_planners = PathPlanners(
                self.tf_buffer
            )  # instance of PathPlanners with tf_buffer
            paths = path_planners.path_for_bin()

            if paths is None:
                return "aborted"

            userdata.planned_paths = paths
            return "succeeded"

        except Exception as e:
            rospy.logerr("[PlanTorpedoPathState] Error: %s", str(e))
            return "aborted"


class BinTaskState(smach.State):
    def __init__(self, bin_task_depth):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        with self.state_machine:
            smach.StateMachine.add(
                "ENABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "SET_BIN_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BIN_DEPTH",
                SetDepthState(depth=bin_task_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "FIND_AND_AIM_BIN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FIND_AND_AIM_BIN",
                SearchForPropState(
                    look_at_frame="bin_whole_link",
                    alignment_frame="bin_search",
                    full_rotation=False,
                    set_frame_duration=4.0,
                    source_frame="taluy/base_link",
                    rotation_speed=0.3,
                ),
                transitions={
                    "succeeded": "PLAN_BIN_PATHS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "PLAN_BIN_PATHS",
                PlanBinPathState(self.tf_buffer),
                transitions={
                    "succeeded": "DISABLE_BIN_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_BIN_FRAME_PUBLISHER",
                BinTransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET_TO_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET_TO_PATH",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="dynamic_target"
                ),
                transitions={
                    "succeeded": "EXECUTE_BIN_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "EXECUTE_BIN_PATH",
                ExecutePlannedPathsState(),
                transitions={
                    "succeeded": "CHECK_DROP_AREA_FOUND",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "BIN_SECOND_TRIAL_ATTEMPT",
                },
            )
            smach.StateMachine.add(
                "CHECK_DROP_AREA_FOUND",
                CheckForDropAreaState(source_frame="odom", timeout=2.0),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "BIN_SECOND_TRIAL_ATTEMPT",
                },
            )
            smach.StateMachine.add(
                "BIN_SECOND_TRIAL_ATTEMPT",
                BinSecondTrialAttemptState(self.tf_buffer),
                transitions={
                    "succeeded": "SET_ALIGN_TO_FOUND_DROP_AREA",
                    "preempted": "CANCEL_ALIGN_CONTROLLER",
                    "aborted": "DROP_BALL_1",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_TO_FOUND_DROP_AREA",
                SetAlignToFoundState(source_frame="taluy/base_link/ball_dropper_link"),
                transitions={
                    "succeeded": "WAIT_FOR_ALIGNING_DROP_AREA",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_ALIGNING_DROP_AREA",
                DelayState(delay_time=15.0),
                transitions={
                    "succeeded": "DROP_BALL_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DROP_BALL_1",
                DropBallState(),
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
                DropBallState(),
                transitions={
                    "succeeded": "WAIT_FOR_BALL_DROP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "WAIT_FOR_BALL_DROP_2",
                DelayState(delay_time=3.0),
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
