from auv_smach.tf_utils import get_base_link
from .initialize import *
import smach
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_smach.common import (
    CancelAlignControllerState,
    SearchForPropState,
    SetDepthState,
    SetDetectionState,
    AlignFrame,
)
from auv_smach.initialize import DelayState
import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from auv_msgs.msg import MiniSlalomAction, MiniSlalomGoal
from auv_msgs.srv import (
    SetDetectionFocus,
    SetDetectionFocusRequest,
)


class PublishSearchPointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/publish_search_points",
            Trigger,
            request=TriggerRequest(),
        )


class StartPointSearchState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/start_point_search",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class StopPointSearchState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/stop_point_search",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class PublishWaypointsState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "slalom/publish_waypoints",
            SetBool,
            request=SetBoolRequest(data=True),
        )


class NavigateThroughSlalomState(smach.State):
    def __init__(
        self,
        slalom_depth: float,
        slalom_direction: str = "left",
        slalom_exit_angle: float = 0.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.base_link = get_base_link()
        self.slalom_depth = slalom_depth
        self.slalom_direction = slalom_direction

        self.sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

        with self.sm:
            smach.StateMachine.add(
                "SET_SLALOM_DEPTH",
                SetDepthState(depth=self.slalom_depth),
                transitions={
                    "succeeded": "ENABLE_SLALOM_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_SLALOM_DETECTION",
                SetDetectionState(camera_name="slalom", enable=True),
                transitions={
                    "succeeded": "ROTATE_TO_FIND_RED_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ROTATE_TO_FIND_RED_PIPE",
                SearchForPropState(
                    look_at_frame="slalom_red_pipe_link",
                    alignment_frame="sus",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ROTATE_TO_FIND_WHITE_PIPE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ROTATE_TO_FIND_WHITE_PIPE",
                SearchForPropState(
                    look_at_frame="slalom_white_pipe_link",
                    alignment_frame="sus",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "PUBLISH_SEARCH_POINTS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "PUBLISH_SEARCH_POINTS",
                PublishSearchPointsState(),
                transitions={
                    "succeeded": "START_POINT_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "START_POINT_SEARCH",
                StartPointSearchState(),
                transitions={
                    "succeeded": "ALIGN_SEQUENCE_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            first_side = "right" if self.slalom_direction == "left" else "left"
            second_side = "left" if self.slalom_direction == "left" else "right"

            smach.StateMachine.add(
                "ALIGN_SEQUENCE_START",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_search_start",
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_FIRST_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_FIRST_SEARCH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=f"slalom_search_{first_side}",
                    confirm_duration=2.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_SECOND_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_TO_SECOND_SEARCH",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame=f"slalom_search_{second_side}",
                    confirm_duration=2.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "STOP_POINT_SEARCH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "STOP_POINT_SEARCH",
                StopPointSearchState(),
                transitions={
                    "succeeded": "PUBLISH_WAYPOINTS",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "PUBLISH_WAYPOINTS",
                PublishWaypointsState(),
                transitions={
                    "succeeded": "WAIT_FOR_TF",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # just in case
            smach.StateMachine.add(
                "WAIT_FOR_TF",
                DelayState(delay_time=1.0),
                transitions={
                    "succeeded": "ALIGN_WP_0",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_WP_0",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_wp_0",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                ),
                transitions={
                    "succeeded": "LOOK_WP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "LOOK_WP_1",
                SearchForPropState(
                    look_at_frame="slalom_wp_1",
                    alignment_frame=f"sus",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_WP_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_WP_1",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_wp_1",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    keep_orientation=True,
                ),
                transitions={
                    "succeeded": "LOOK_WP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "LOOK_WP_2",
                SearchForPropState(
                    look_at_frame="slalom_wp_2",
                    alignment_frame=f"sus",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_WP_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            smach.StateMachine.add(
                "ALIGN_WP_2",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="slalom_wp_2",
                    confirm_duration=1.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    keep_orientation=True,
                ),
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
        return self.sm.execute()


class NavigateThroughMiniSlalomState(smach.State):
    """RoboSub 2026 image-space slalom task for Taluy Mini."""

    def __init__(
        self,
        slalom_depth: float,
        slalom_direction: str = "left",
        target_gate_count: int = 3,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.slalom_depth = slalom_depth
        self.slalom_direction = slalom_direction
        self.target_gate_count = target_gate_count
        self.active_pub = rospy.Publisher("slalom/active", Bool, queue_size=1)
        self.visual_wrench_pub = rospy.Publisher(
            "slalom/visual_wrench", WrenchStamped, queue_size=1
        )

    @staticmethod
    def _set_front_detection(enabled):
        rospy.wait_for_service("vision/enable_front_camera_detections", timeout=5.0)
        service = rospy.ServiceProxy("vision/enable_front_camera_detections", SetBool)
        response = service(SetBoolRequest(data=enabled))
        if not response.success:
            raise rospy.ServiceException(response.message)

    @staticmethod
    def _set_focus(focus):
        rospy.wait_for_service("vision/set_front_camera_focus", timeout=5.0)
        service = rospy.ServiceProxy("vision/set_front_camera_focus", SetDetectionFocus)
        response = service(SetDetectionFocusRequest(focus_object=focus))
        if not response.success:
            raise rospy.ServiceException(response.message)

    def _cleanup(self):
        self.active_pub.publish(False)
        self.visual_wrench_pub.publish(WrenchStamped())
        try:
            self._set_focus("none")
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("[MiniSlalom] Cleanup focus failed: %s", exc)

    def execute(self, userdata):
        depth_outcome = SetDepthState(depth=self.slalom_depth).execute(userdata)
        if depth_outcome != "succeeded":
            return depth_outcome

        client = actionlib.SimpleActionClient("slalom/run", MiniSlalomAction)
        try:
            self._set_front_detection(True)
            self._set_focus("slalom")
            if not client.wait_for_server(rospy.Duration(8.0)):
                rospy.logerr("[MiniSlalom] slalom/run action server unavailable")
                return "aborted"

            goal = MiniSlalomGoal(
                direction=self.slalom_direction,
                target_gate_count=self.target_gate_count,
            )
            client.send_goal(goal)
            while not rospy.is_shutdown():
                if self.preempt_requested():
                    client.cancel_goal()
                    client.wait_for_result(rospy.Duration(1.0))
                    self.service_preempt()
                    return "preempted"
                if client.wait_for_result(rospy.Duration(0.1)):
                    result = client.get_result()
                    if client.get_state() == GoalStatus.SUCCEEDED:
                        return "succeeded" if result and result.success else "aborted"
                    return "aborted"
            return "preempted"
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr("[MiniSlalom] Setup or execution failed: %s", exc)
            client.cancel_all_goals()
            return "aborted"
        finally:
            self._cleanup()
