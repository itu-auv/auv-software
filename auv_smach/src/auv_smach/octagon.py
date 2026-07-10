from .initialize import *
import smach
import smach_ros
from auv_smach.tf_utils import get_base_link, get_tf_buffer
from auv_common_lib.transform import lookup_fresh_transform
from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
    SetDetectionFocusState,
    SetDetectionFocusBottomState,
    DynamicPathState,
    DynamicPathWithTransformCheck,
    CheckForTransformState,
    AlignFrame,
    SearchForPropState,
    SetDetectionState,
    AlignAndCreateRotatingFrame,
    GravityZEnable,
)
from auv_smach.initialize import DelayState
from auv_smach.acoustic import AcousticTransmitter
from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from std_msgs.msg import String, UInt16
import tf2_ros


class GripperAngleOpenState(smach.State):
    """
    for real life gripper).
    """

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.pub = rospy.Publisher("actuators/gripper1/set_angle", UInt16, queue_size=1)
        self.angle_value = 2100

    def execute(self, userdata) -> str:
        try:
            msg = UInt16()
            msg.data = self.angle_value
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
    for real life gripper close).
    """

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.pub = rospy.Publisher("actuators/gripper1/set_angle", UInt16, queue_size=1)
        self.angle_value = 1100

    def execute(self, userdata) -> str:
        try:
            msg = UInt16()
            msg.data = self.angle_value
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
        self.tf_buffer = get_tf_buffer()

    def execute(self, userdata) -> str:
        import rospy

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while (rospy.Time.now() - start_time) < self.timeout:
            if self.preempt_requested():
                return "preempted"

            try:
                lookup_fresh_transform(
                    self.tf_buffer,
                    self.source_frame,
                    self.target_frame,
                    rospy.Duration(rospy.get_param("~tf_lookup_timeout", 0.2)),
                    rospy.Duration(rospy.get_param("~tf_freshness_threshold", 0.4)),
                )
                rospy.loginfo(
                    f"[CheckBottleLinkState] Transform from '{self.source_frame}' to '{self.target_frame}' found."
                )
                return "succeeded"
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                pass

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


class PickAndDropSequence(smach.StateMachine):
    def __init__(self, target_object: str, target_basket: str):
        smach.StateMachine.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        base_link = get_base_link()
        keep_object_orientation = target_object in {"pill_link", "nutbolt_link"}
        rospy.loginfo(
            "[PickAndDropSequence] target_object=%s, target_basket=%s",
            target_object,
            target_basket,
        )

        with self:
            smach.StateMachine.add(
                "DEPTH_BEFORE_ALIGNING_OBJECT",
                SetDepthState(
                    depth=-0.6,
                ),
                transitions={
                    "succeeded": "ALIGN_TARGET_OBJECT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TARGET_OBJECT",
                AlignFrame(
                    source_frame="taluy/gripper_link_p2",
                    target_frame=target_object,
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    keep_orientation=keep_object_orientation,
                    closest_yaw=not keep_object_orientation,
                    confirm_duration=4.0,
                    timeout=30.0,
                    max_linear_velocity=0.2,
                    max_angular_velocity=0.2,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ALIGN_TARGET_OBJECT_slow",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TARGET_OBJECT_slow",
                AlignFrame(
                    source_frame="taluy/gripper_link_p2",
                    target_frame=target_object,
                    dist_threshold=0.05,
                    yaw_threshold=0.1,
                    keep_orientation=keep_object_orientation,
                    closest_yaw=not keep_object_orientation,
                    confirm_duration=0.1,
                    timeout=3.0,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "CLOSE_GRAVITY_Z",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_GRAVITY_Z",
                GravityZEnable(enable=False),
                transitions={
                    "succeeded": "DEPTH_TO_COLLECT_OBJECT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DEPTH_TO_COLLECT_OBJECT",
                SetDepthState(
                    depth=-1.1,
                    max_velocity=0.07,
                    depth_threshold=0.05,
                    confirm_duration=2.0,
                    timeout=15.0,
                ),
                transitions={
                    "succeeded": "CLOSE_GRIPPER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CLOSE_GRIPPER",
                GripperAngleCloseState(),
                transitions={
                    "succeeded": "DEPTH_TO_DEFAULT_AFTER_PICKING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DEPTH_TO_DEFAULT_AFTER_PICKING",
                SetDepthState(depth=-0.5, max_velocity=0.25, confirm_duration=1.0),
                transitions={
                    "succeeded": "OPEN_GRAVITY_Z",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "OPEN_GRAVITY_Z",
                GravityZEnable(enable=True),
                transitions={
                    "succeeded": "ALIGN_TO_MIDDLE_BASKET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_MIDDLE_BASKET",
                AlignFrame(
                    source_frame=base_link,
                    target_frame="octagon_table_segment_link",
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    closest_yaw=False,
                    keep_orientation=True,
                    confirm_duration=1.0,
                    timeout=15.0,
                    max_linear_velocity=0.25,
                    max_angular_velocity=0.25,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "SURFACE_WITH_OBJECT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACE_WITH_OBJECT",
                SetDepthState(
                    depth=0,
                    timeout=10.0,
                    max_velocity=0.4,
                    depth_threshold=0.05,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "DEPTH_TO_DEFAULT_AFTER_SURFACING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DEPTH_TO_DEFAULT_AFTER_SURFACING",
                SetDepthState(depth=-0.5, max_velocity=0.3, confirm_duration=1.0),
                transitions={
                    "succeeded": "ALIGN_TARGET_BASKET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TARGET_BASKET",
                AlignFrame(
                    source_frame="taluy/gripper_link_p2",
                    target_frame=target_basket,
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    closest_yaw=False,
                    confirm_duration=4.0,
                    keep_orientation=True,
                    timeout=20.0,
                    max_linear_velocity=0.25,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ALIGN_TARGET_BASKET_slow",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TARGET_BASKET_slow",
                AlignFrame(
                    source_frame="taluy/gripper_link_p2",
                    target_frame=target_basket,
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    closest_yaw=False,
                    confirm_duration=0.1,
                    keep_orientation=True,
                    timeout=20.0,
                    max_linear_velocity=0.1,
                    max_angular_velocity=0.1,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DEPTH_TO_DROP_OBJECT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DEPTH_TO_DROP_OBJECT",
                SetDepthState(
                    depth=-0.95,
                    max_velocity=0.1,
                    confirm_duration=1.0,
                    depth_threshold=0.07,
                ),
                transitions={
                    "succeeded": "OPEN_GRIPPER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "OPEN_GRIPPER",
                GripperAngleOpenState(),
                transitions={
                    "succeeded": "DEPTH_TO_DEFAULT_AFTER_DROPPING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DEPTH_TO_DEFAULT_AFTER_DROPPING",
                SetDepthState(depth=-0.5, max_velocity=0.25, confirm_duration=1.0),
                transitions={
                    "succeeded": "ALIGN_TO_MIDDLE_BASKET_AFTER_DROPPING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_MIDDLE_BASKET_AFTER_DROPPING",
                AlignFrame(
                    source_frame=base_link,
                    target_frame="octagon_table_segment_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    closest_yaw=False,
                    keep_orientation=True,
                    confirm_duration=3.0,
                    timeout=20.0,
                    max_linear_velocity=0.25,
                    max_angular_velocity=0.25,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class PickRemainingOctagonTargetsState(smach.State):
    def __init__(
        self,
        target_baskets: dict,
        object_list_topic: str = "octagon/object_list",
        wait_timeout: float = 2.0,
        max_attempts: int = 2,
        set_role_search_rotation_count=None,
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
        )
        self.target_baskets = target_baskets
        self.wait_timeout = rospy.Duration(wait_timeout)
        self.max_attempts = max(0, max_attempts)
        self.total_targets = len(target_baskets)
        self.set_role_search_rotation_count = set_role_search_rotation_count
        self.latest_objects = []
        self.latest_remaining_count = 1
        self.last_update_time = rospy.Time(0)
        self.active_sequence = None
        self.object_list_sub = rospy.Subscriber(
            object_list_topic, String, self._object_list_cb, queue_size=1
        )

    def _object_list_cb(self, msg):
        selected_objects = []
        seen_objects = set()

        for object_name in msg.data.split(","):
            object_name = object_name.strip()
            if not object_name:
                continue
            if object_name not in self.target_baskets or object_name in seen_objects:
                continue

            selected_objects.append(object_name)
            seen_objects.add(object_name)

        self.latest_objects = selected_objects
        self.latest_remaining_count = len(selected_objects)
        self.last_update_time = rospy.Time.now()

    def request_preempt(self):
        smach.State.request_preempt(self)
        if self.active_sequence is not None:
            self.active_sequence.request_preempt()

    def _wait_for_fresh_targets(self):
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while rospy.Time.now() - start_time < self.wait_timeout:
            if self.preempt_requested():
                return None

            if self.last_update_time >= start_time:
                break

            rate.sleep()

        return [
            (object_name, self.target_baskets[object_name])
            for object_name in self.latest_objects
        ]

    def execute(self, userdata) -> str:
        previous_target_object = None
        attempted_objects = set()

        for attempt in range(self.max_attempts):
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            targets = self._wait_for_fresh_targets()
            if targets is None:
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"
                break

            if not targets:
                break

            target = None
            for candidate in targets:
                if candidate[0] not in attempted_objects:
                    target = candidate
                    break

            if target is None:
                target_index = 0
                if previous_target_object is not None:
                    target_names = [target_object for target_object, _ in targets]
                    if previous_target_object in target_names:
                        target_index = (
                            target_names.index(previous_target_object) + 1
                        ) % len(targets)
                target = targets[target_index]

            target_object, target_basket = target
            rospy.loginfo(
                "[PickRemainingOctagonTargetsState] attempt=%d/%d, target=%s",
                attempt + 1,
                self.max_attempts,
                target_object,
            )
            previous_target_object = target_object
            attempted_objects.add(target_object)

            self.active_sequence = PickAndDropSequence(target_object, target_basket)
            try:
                outcome = self.active_sequence.execute()
            finally:
                self.active_sequence = None

            if outcome is None or outcome == "preempted":
                return "preempted"

        rotation_count = max(0, self.total_targets - self.latest_remaining_count)
        if self.set_role_search_rotation_count is not None:
            self.set_role_search_rotation_count(rotation_count)
        rospy.loginfo(
            "[PickRemainingOctagonTargetsState] remaining=%d, role_search_rotations=%d",
            self.latest_remaining_count,
            rotation_count,
        )
        return "succeeded"


class OctagonTaskState(smach.State):
    def __init__(
        self,
        octagon_depth: float,
        octagon_role_frame: str = None,
        octagon_search_frame: str = None,
        start_from_table: bool = False,
        octagon_target_role_frame: str = None,
        remaining_targets_wait_timeout: float = 2.0,
        remaining_targets_max_attempts: int = 2,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.griper_mode = True
        self.base_link = get_base_link()
        self.octagon_target_role_frame = octagon_target_role_frame or octagon_role_frame
        self.octagon_search_frame = octagon_search_frame
        if self.octagon_target_role_frame is None:
            raise ValueError("octagon_target_role_frame must be provided")
        if self.octagon_search_frame is None:
            raise ValueError("octagon_search_frame must be provided")
        after_bottom_focus = (
            "MOVE_GRIPPER" if start_from_table else "DYNAMIC_PATH_WITH_BOTTLE_CHECK"
        )
        pick_and_drop_targets = [
            ("pill_link", "basket_redcross_segment_link"),
            ("nutbolt_link", "basket_warning_segment_link"),
            ("electric_link", "basket_warning_segment_link"),
            ("bandaid_link", "basket_redcross_segment_link"),
        ]
        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        pick_and_drop_target_baskets = dict(pick_and_drop_targets)
        role_search_rotation = AlignAndCreateRotatingFrame(
            # source_frame=self.base_link,
            source_frame="taluy/base_link/dvl_mount_link",
            rotating_frame_name="octagon_target_role_search_frame",
            rotation_period=15.0,
            rotation_count=3,
        )
        role_search_rotation_count = {"value": 3}

        def set_role_search_rotation_count(rotation_count):
            role_search_rotation_count["value"] = rotation_count
            role_search_rotation._states["CREATE_ROTATING_FRAME"].rotation_count = max(
                1, rotation_count
            )

        def get_role_search_rotation_outcome(userdata):
            return "rotate" if role_search_rotation_count["value"] > 0 else "skip"

        fallback_align_targets = [
            "pill_link",
            "nutbolt_link",
            "electric_link",
            "bandaid_link",
            "basket_redcross_segment_link",
            "basket_warning_segment_link",
        ]

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_INITIAL_DEPTH",
                SetDepthState(depth=-0.6),
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
                    look_at_frame=self.octagon_search_frame,
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=0.4,
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
                    "succeeded": "SET_OCTAGON_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(depth=octagon_depth),
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
                    source_frame=self.base_link,
                    target_frame="octagon_closer_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_OCTAGON_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_OCTAGON_FRAME_PUBLISHER",
                OctagonFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "SET_BATUHAN_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BATUHAN_DEPTH",
                SetDepthState(depth=-0.6),
                transitions={
                    "succeeded": "ENABLE_SEGMENT_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_SEGMENT_DETECTION",
                SetDetectionState(camera_name="segment", enable=True),
                transitions={
                    "succeeded": "yolo_bekle",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "yolo_bekle",
                DelayState(delay_time=2.0),
                transitions={
                    "succeeded": after_bottom_focus,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_WITH_BOTTLE_CHECK",
                DynamicPathWithTransformCheck(
                    plan_target_frame="octagon_further_link",
                    transform_source_frame="odom",
                    transform_target_frame="octagon_table_segment_link",
                    max_linear_velocity=0.2,
                ),
                transitions={
                    "succeeded": "MOVE_GRIPPER",
                    "preempted": "preempted",
                    "aborted": "CHECK_OCTAGON_SEGMENT_FALLBACK_1",
                },
            )
            for index, target_frame in enumerate(fallback_align_targets):
                check_state_name = f"CHECK_OCTAGON_SEGMENT_FALLBACK_{index + 1}"
                align_state_name = f"ALIGN_TO_OCTAGON_SEGMENT_FALLBACK_{index + 1}"
                next_state = (
                    f"CHECK_OCTAGON_SEGMENT_FALLBACK_{index + 2}"
                    if index + 1 < len(fallback_align_targets)
                    else "MOVE_GRIPPER"
                )
                smach.StateMachine.add(
                    check_state_name,
                    CheckForTransformState(
                        source_frame="odom",
                        target_frame=target_frame,
                        timeout=0.5,
                        check_rate_hz=20,
                    ),
                    transitions={
                        "succeeded": align_state_name,
                        "preempted": "preempted",
                        "aborted": next_state,
                    },
                )
                smach.StateMachine.add(
                    align_state_name,
                    AlignFrame(
                        source_frame=self.base_link,
                        target_frame=target_frame,
                        dist_threshold=0.1,
                        yaw_threshold=0.1,
                        confirm_duration=2.0,
                        timeout=15.0,
                        keep_orientation=True,
                        max_linear_velocity=0.2,
                        max_angular_velocity=0.1,
                        cancel_on_success=False,
                    ),
                    transitions={
                        "succeeded": "MOVE_GRIPPER",
                        "preempted": "preempted",
                        "aborted": next_state,
                    },
                )
            smach.StateMachine.add(
                "MOVE_GRIPPER",
                GripperAngleOpenState(),
                transitions={
                    "succeeded": "ALIGN_TO_TABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TABLE",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="octagon_table_segment_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=4.0,
                    timeout=30.0,
                    keep_orientation=True,
                    max_linear_velocity=0.3,
                    max_angular_velocity=0.3,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "ENABLE_OCTAGON_FRAME_PUBLISHER_ON_TABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_OCTAGON_FRAME_PUBLISHER_ON_TABLE",
                OctagonFramePublisherServiceState(req=True),
                transitions={
                    "succeeded": "PICK_AND_DROP_SEQUENCE_1",
                    "preempted": "preempted",
                    "aborted": "PICK_AND_DROP_SEQUENCE_1",
                },
            )
            smach.StateMachine.add(
                "PICK_AND_DROP_SEQUENCE_1",
                PickAndDropSequence(*pick_and_drop_targets[0]),
                transitions={
                    "succeeded": "PICK_AND_DROP_SEQUENCE_2",
                    "preempted": "preempted",
                    "aborted": "PICK_AND_DROP_SEQUENCE_2",
                },
            )
            smach.StateMachine.add(
                "PICK_AND_DROP_SEQUENCE_2",
                PickAndDropSequence(*pick_and_drop_targets[1]),
                transitions={
                    "succeeded": "PICK_AND_DROP_SEQUENCE_3",
                    "preempted": "preempted",
                    "aborted": "PICK_AND_DROP_SEQUENCE_3",
                },
            )
            smach.StateMachine.add(
                "PICK_AND_DROP_SEQUENCE_3",
                PickAndDropSequence(*pick_and_drop_targets[2]),
                transitions={
                    "succeeded": "PICK_AND_DROP_SEQUENCE_4",
                    "preempted": "preempted",
                    "aborted": "PICK_AND_DROP_SEQUENCE_4",
                },
            )
            smach.StateMachine.add(
                "PICK_AND_DROP_SEQUENCE_4",
                PickAndDropSequence(*pick_and_drop_targets[3]),
                transitions={
                    "succeeded": "PICK_REMAINING_OCTAGON_TARGETS",
                    "preempted": "preempted",
                    "aborted": "PICK_REMAINING_OCTAGON_TARGETS",
                },
            )
            smach.StateMachine.add(
                "PICK_REMAINING_OCTAGON_TARGETS",
                PickRemainingOctagonTargetsState(
                    pick_and_drop_target_baskets,
                    wait_timeout=remaining_targets_wait_timeout,
                    max_attempts=remaining_targets_max_attempts,
                    set_role_search_rotation_count=set_role_search_rotation_count,
                ),
                transitions={
                    "succeeded": "OCTAGON_FACING_DEPTH",
                    "preempted": "preempted",
                    "aborted": "OCTAGON_FACING_DEPTH",
                },
            )
            smach.StateMachine.add(
                "OCTAGON_FACING_DEPTH",
                SetDepthState(
                    depth=-0.4,
                    depth_threshold=0.05,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "CHECK_ROLE_SEARCH_ROTATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CHECK_ROLE_SEARCH_ROTATION",
                smach.CBState(
                    get_role_search_rotation_outcome,
                    outcomes=["rotate", "skip"],
                ),
                transitions={
                    "rotate": "ROTATE_THREE_TURNS",
                    "skip": "SEARCH_FOR_ROLE_TARGET",
                },
            )
            smach.StateMachine.add(
                "ROTATE_THREE_TURNS",
                role_search_rotation,
                transitions={
                    "succeeded": "ALIGN_TO_TABLE_BEFORE_ROLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TABLE_BEFORE_ROLE",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="octagon_table_segment_link",
                    dist_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=15.0,
                    keep_orientation=True,
                    max_linear_velocity=0.3,
                    max_angular_velocity=0.3,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "SEARCH_FOR_ROLE_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SEARCH_FOR_ROLE_TARGET",
                SearchForPropState(
                    look_at_frame=self.octagon_target_role_frame,
                    alignment_frame="octagon_search_frame",
                    full_rotation=False,
                    source_frame=self.base_link,
                    rotation_speed=-0.2,
                ),
                transitions={
                    "succeeded": "FINAL_SURFACE",
                    "preempted": "preempted",
                    "aborted": "FINAL_SURFACE",
                },
            )
            smach.StateMachine.add(
                "FINAL_SURFACE",
                SetDepthState(
                    depth=-0.1,
                    timeout=10.0,
                    depth_threshold=0.05,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "DISABLE_SEGMENT_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_SEGMENT_DETECTION",
                SetDetectionState(camera_name="segment", enable=False),
                transitions={
                    "succeeded": "FINISHED_OCTAGON_TASK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FINISHED_OCTAGON_TASK",
                SetDepthState(depth=-0.5, confirm_duration=1.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "succeeded",
                },
            )

        if start_from_table:
            self.state_machine.set_initial_state(["ENABLE_SEGMENT_DETECTION"])

    def execute(self, userdata):
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome


class OctagonSurfaceState(smach.State):
    def __init__(
        self,
        octagon_depth: float,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_link = get_base_link()

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        # Open the container for adding states
        with self.state_machine:
            smach.StateMachine.add(
                "SET_OCTAGON_INITIAL_DEPTH",
                SetDepthState(depth=-1.2),
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
                    source_frame=self.base_link,
                    rotation_speed=0.4,
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
                    "succeeded": "SET_OCTAGON_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_OCTAGON_DEPTH",
                SetDepthState(depth=octagon_depth),
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
                    source_frame=self.base_link,
                    target_frame="octagon_closer_link",
                    angle_offset=0.0,
                    dist_threshold=0.1,
                    yaw_threshold=0.1,
                    confirm_duration=1.0,
                    timeout=60.0,
                    cancel_on_success=False,
                ),
                transitions={
                    "succeeded": "DISABLE_OCTAGON_FRAME_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_OCTAGON_FRAME_PUBLISHER",
                OctagonFramePublisherServiceState(req=False),
                transitions={
                    "succeeded": "SET_BATUHAN_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_BATUHAN_DEPTH",
                SetDepthState(depth=-0.6),
                transitions={
                    "succeeded": "ENABLE_SEGMENT_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_SEGMENT_DETECTION",
                SetDetectionState(camera_name="segment", enable=True),
                transitions={
                    "succeeded": "DYNAMIC_PATH_WITH_BOTTLE_CHECK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DYNAMIC_PATH_WITH_BOTTLE_CHECK",
                DynamicPathWithTransformCheck(
                    plan_target_frame="octagon_further_link",
                    transform_source_frame="odom",
                    transform_target_frame="octagon_table_segment_link",
                    max_linear_velocity=0.2,
                ),
                transitions={
                    "succeeded": "ALIGN_TO_TABLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_TO_TABLE",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="octagon_table_segment_link",
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
                    "succeeded": "OCTAGON_FACING_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "OCTAGON_FACING_DEPTH",
                SetDepthState(
                    depth=-0.4,
                    depth_threshold=0.05,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "FINAL_SURFACE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FINAL_SURFACE",
                SetDepthState(
                    depth=-0.05,
                    max_velocity=0.2,
                    depth_threshold=0.05,
                    confirm_duration=2.0,
                ),
                transitions={
                    "succeeded": "FINISHED_OCTAGON_TASK",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FINISHED_OCTAGON_TASK",
                SetDepthState(depth=-0.5, max_velocity=0.2, confirm_duration=1.0),
                transitions={
                    "succeeded": "DISABLE_SEGMENT_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_SEGMENT_DETECTION",
                SetDetectionState(camera_name="segment", enable=False),
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
