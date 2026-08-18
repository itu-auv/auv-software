import copy

import rospy
import smach
from std_srvs.srv import Trigger, TriggerRequest

from auv_smach.common import (
    AlignFrame,
    DynamicPathState,
    SearchForPropState,
    SetDepthState,
)
from auv_smach.tf_utils import get_base_link


CONFIG_PARAM = "surface_waypoint_mission/config"
IMAGE_CAPTURE_SERVICE = "image_saver/capture"
IMAGE_FOLDER_PARAM = "image_saver/waypoint_folder"
PHOTOS_PER_WAYPOINT = 1
PHOTO_INTERVAL_SECONDS = 0.4
WAYPOINT_COUNT = 3


class CaptureWaypointImageState(smach.State):
    """Requests exactly one image from image_saver."""

    def __init__(self, waypoint_index):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.waypoint_index = waypoint_index

    def execute(self, _userdata):
        waypoint_name = f"waypoint_{self.waypoint_index}"
        try:
            rospy.wait_for_service(IMAGE_CAPTURE_SERVICE, timeout=5.0)
            capture_image = rospy.ServiceProxy(IMAGE_CAPTURE_SERVICE, Trigger)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr("[SurfaceWaypoints] Image service failed: %s", exc)
            return "aborted"

        rospy.set_param(IMAGE_FOLDER_PARAM, waypoint_name)
        for photo_number in range(1, PHOTOS_PER_WAYPOINT + 1):
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            try:
                response = capture_image(TriggerRequest())
            except rospy.ServiceException as exc:
                rospy.logerr("[SurfaceWaypoints] Image service failed: %s", exc)
                return "aborted"
            if not response.success:
                rospy.logerr(
                    "[SurfaceWaypoints] Photo %d failed at %s: %s",
                    photo_number,
                    waypoint_name,
                    response.message,
                )
                return "aborted"
            if photo_number < PHOTOS_PER_WAYPOINT:
                rospy.sleep(PHOTO_INTERVAL_SECONDS)

        rospy.loginfo(
            "[SurfaceWaypoints] %d photos captured at %s",
            PHOTOS_PER_WAYPOINT,
            waypoint_name,
        )
        return "succeeded"


class VisitSurfaceWaypointState(smach.StateMachine):
    """Look at, reach, surface at, and leave one waypoint."""

    def __init__(self, source_frame, waypoint):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )

        frame_id = waypoint["frame_id"]
        camera_enabled = waypoint["camera_enabled"]
        after_surface = "CAPTURE_IMAGE" if camera_enabled else "DIVE_FOR_NEXT_WAYPOINT"

        with self:
            smach.StateMachine.add(
                "LOOK_AT_WAYPOINT",
                SearchForPropState(
                    look_at_frame=frame_id,
                    alignment_frame=f"{frame_id}_look_at",
                    source_frame=source_frame,
                    full_rotation=False,
                ),
                transitions={
                    "succeeded": "FOLLOW_PATH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "FOLLOW_PATH",
                DynamicPathState(
                    plan_target_frame=frame_id,
                    align_source_frame=source_frame,
                    keep_orientation=False,
                ),
                transitions={
                    "succeeded": "ALIGN_AT_WAYPOINT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_AT_WAYPOINT",
                AlignFrame(
                    source_frame=source_frame,
                    target_frame=frame_id,
                    dist_threshold=0.25,
                    yaw_threshold=0.2,
                    timeout=20.0,
                    confirm_duration=0.5,
                    cancel_on_success=True,
                    keep_orientation=False,
                    use_frame_depth=False,
                ),
                transitions={
                    "succeeded": "SURFACE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SURFACE",
                SetDepthState(depth=0.0),
                transitions={
                    "succeeded": after_surface,
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            if camera_enabled:
                smach.StateMachine.add(
                    "CAPTURE_IMAGE",
                    CaptureWaypointImageState(waypoint["index"]),
                    transitions={
                        "succeeded": "DIVE_FOR_NEXT_WAYPOINT",
                        "preempted": "preempted",
                        "aborted": "aborted",
                    },
                )
            smach.StateMachine.add(
                "DIVE_FOR_NEXT_WAYPOINT",
                SetDepthState(depth=-1.0),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


class NavigateToSurfaceWaypointsState(smach.State):
    """Visits the three GUI waypoints in their configured order."""

    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = get_base_link()
        self.active_state = None

    def request_preempt(self):
        smach.State.request_preempt(self)
        if self.active_state is not None:
            self.active_state.request_preempt()

    def execute(self, _userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        self.active_state = SetDepthState(depth=-1.0)
        initial_depth_outcome = self.active_state.execute(None)
        self.active_state = None
        if initial_depth_outcome != "succeeded":
            if initial_depth_outcome == "preempted" or self.preempt_requested():
                self.service_preempt()
                return "preempted"
            return "aborted"

        waypoints = self._wait_for_waypoints()
        if waypoints is None:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            return "aborted"

        rospy.loginfo(
            "[SurfaceWaypoints] Visit order: %s",
            " -> ".join(f"WP{waypoint['index']}" for waypoint in waypoints),
        )

        for waypoint in waypoints:
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            rospy.loginfo(
                "[SurfaceWaypoints] Visiting WP%d, camera=%s",
                waypoint["index"],
                "ON" if waypoint["camera_enabled"] else "OFF",
            )
            self.active_state = VisitSurfaceWaypointState(
                self.source_frame,
                waypoint,
            )
            outcome = self.active_state.execute()
            self.active_state = None

            if outcome == "preempted" or self.preempt_requested():
                self.service_preempt()
                return "preempted"
            if outcome != "succeeded":
                return "aborted"

        return "succeeded"

    def _wait_for_waypoints(self):
        start_time = rospy.Time.now()
        rate = rospy.Rate(5.0)

        while not rospy.is_shutdown():
            if self.preempt_requested():
                return None

            config = rospy.get_param(CONFIG_PARAM, None)
            if config is not None:
                try:
                    return self._validate_waypoints(config)
                except (TypeError, ValueError) as exc:
                    rospy.logerr(
                        "[SurfaceWaypoints] Invalid GUI configuration: %s", exc
                    )
                    return None

            if (rospy.Time.now() - start_time).to_sec() >= 300.0:
                rospy.logerr("[SurfaceWaypoints] Timed out waiting for GUI input")
                return None

            rospy.logwarn_throttle(
                10.0,
                "[SurfaceWaypoints] Waiting for three waypoint entries from GUI",
            )
            rate.sleep()

        return None

    @staticmethod
    def _validate_waypoints(config):
        waypoints = config.get("waypoints") if isinstance(config, dict) else None
        if not isinstance(waypoints, list) or len(waypoints) != WAYPOINT_COUNT:
            raise ValueError("exactly three waypoints are required")

        frames = set()
        normalized = []
        for index, waypoint in enumerate(waypoints, start=1):
            frame_id = str(waypoint.get("frame_id", "")).strip("/")
            if not frame_id or frame_id in frames:
                raise ValueError("waypoint frame names must be unique")
            frames.add(frame_id)

            item = copy.deepcopy(waypoint)
            item["index"] = index
            item["frame_id"] = frame_id
            item["camera_enabled"] = bool(item.get("camera_enabled", False))
            normalized.append(item)
        return normalized
