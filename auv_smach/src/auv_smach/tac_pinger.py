from .initialize import SetStartFrameState
import threading

import numpy as np
import rospy
import smach
import tf2_ros
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker

from auv_smach.common import AlignFrame
from auv_smach.tf_utils import get_base_link, get_tf_buffer


class CapturePingerMeasurementState(smach.State):
    def __init__(
        self,
        measurements,
        marker_topic: str = "/taluy/acoustic/hydrophone/marker",
        odom_frame: str = "odom",
        timeout: float = 5.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.measurements = measurements
        self.marker_topic = marker_topic
        self.odom_frame = odom_frame
        self.timeout = timeout
        self.base_link = get_base_link()
        self.tf_buffer = get_tf_buffer()

        self._lock = threading.Lock()
        self._latest_marker = None
        self._latest_marker_received_at = rospy.Time(0)
        self._marker_sub = rospy.Subscriber(
            self.marker_topic, Marker, self._marker_callback
        )

    def _marker_callback(self, msg):
        with self._lock:
            self._latest_marker = msg
            self._latest_marker_received_at = rospy.Time.now()

    @staticmethod
    def _rotation_matrix(transform):
        rotation = transform.transform.rotation
        return quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])[
            :3, :3
        ]

    def _lookup_robot_position(self):
        transform = self.tf_buffer.lookup_transform(
            self.odom_frame,
            self.base_link,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        translation = transform.transform.translation
        return np.array([translation.x, translation.y], dtype=float)

    def _marker_direction_in_odom(self, marker):
        if len(marker.points) < 2:
            rospy.logwarn(
                "[CapturePingerMeasurementState] Marker has fewer than 2 points"
            )
            return None

        start = marker.points[0]
        end = marker.points[1]
        direction_marker = np.array(
            [end.x - start.x, end.y - start.y, end.z - start.z],
            dtype=float,
        )

        if np.linalg.norm(direction_marker[:2]) < 1e-9:
            rospy.logwarn("[CapturePingerMeasurementState] Marker direction is zero")
            return None

        transform = self.tf_buffer.lookup_transform(
            self.odom_frame,
            marker.header.frame_id,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        direction_odom = self._rotation_matrix(transform).dot(direction_marker)
        direction_2d = direction_odom[:2]
        direction_norm = np.linalg.norm(direction_2d)

        if direction_norm < 1e-9:
            rospy.logwarn("[CapturePingerMeasurementState] Odom direction is zero")
            return None

        return direction_2d / direction_norm

    def execute(self, userdata):
        start_time = rospy.Time.now()
        deadline = start_time + rospy.Duration(self.timeout)
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            if rospy.Time.now() > deadline:
                rospy.logwarn(
                    "[CapturePingerMeasurementState] Timeout waiting for marker on %s",
                    self.marker_topic,
                )
                return "aborted"

            with self._lock:
                marker = self._latest_marker
                marker_received_at = self._latest_marker_received_at

            if marker is None or marker_received_at < start_time:
                rate.sleep()
                continue

            try:
                robot_position = self._lookup_robot_position()
                direction = self._marker_direction_in_odom(marker)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    1.0, "[CapturePingerMeasurementState] TF lookup failed: %s", e
                )
                rate.sleep()
                continue

            if direction is None:
                return "aborted"

            angle = np.arctan2(direction[1], direction[0])
            self.measurements.append(
                {
                    "position": robot_position,
                    "direction": direction,
                    "angle": angle,
                }
            )
            rospy.loginfo(
                "[CapturePingerMeasurementState] Measurement %d: "
                "robot=(%.3f, %.3f), angle=%.2f deg",
                len(self.measurements),
                robot_position[0],
                robot_position[1],
                np.degrees(angle),
            )
            return "succeeded"

        return "aborted"


class EstimatePingerPositionState(smach.State):
    def __init__(
        self,
        measurements,
        output_frame: str = "tac_pinger_link",
        odom_frame: str = "odom",
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["pinger_position"],
        )

        self.measurements = measurements
        self.output_frame = output_frame
        self.odom_frame = odom_frame

    @staticmethod
    def _estimate_intersection(measurements):
        matrix = np.zeros((2, 2), dtype=float)
        vector = np.zeros(2, dtype=float)

        for measurement in measurements:
            position = measurement["position"]
            direction = measurement["direction"]
            normal_projection = np.eye(2) - np.outer(direction, direction)
            matrix += normal_projection
            vector += normal_projection.dot(position)

        if abs(np.linalg.det(matrix)) < 1e-9:
            return None

        return np.linalg.solve(matrix, vector)

    def _publish_pinger_frame(self, pinger_position):
        rospy.wait_for_service("set_object_transform", timeout=2.0)
        set_object_transform = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )

        request = SetObjectTransformRequest()
        request.transform.header.frame_id = self.odom_frame
        request.transform.child_frame_id = self.output_frame
        request.transform.transform.translation.x = float(pinger_position[0])
        request.transform.transform.translation.y = float(pinger_position[1])
        request.transform.transform.translation.z = 0.0
        request.transform.transform.rotation.w = 1.0

        response = set_object_transform(request)
        if not response.success:
            rospy.logwarn(
                "[EstimatePingerPositionState] Failed to publish pinger frame: %s",
                response.message,
            )
            return False

        return True

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        if len(self.measurements) < 3:
            rospy.logwarn(
                "[EstimatePingerPositionState] Need 3 measurements, got %d",
                len(self.measurements),
            )
            return "aborted"

        pinger_position = self._estimate_intersection(self.measurements[-3:])
        if pinger_position is None:
            rospy.logwarn("[EstimatePingerPositionState] Bearing lines are degenerate")
            return "aborted"

        userdata.pinger_position = pinger_position
        rospy.loginfo(
            "[EstimatePingerPositionState] Estimated pinger position in %s: "
            "(%.3f, %.3f)",
            self.odom_frame,
            pinger_position[0],
            pinger_position[1],
        )

        try:
            if not self._publish_pinger_frame(pinger_position):
                return "aborted"
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(
                "[EstimatePingerPositionState] set_object_transform failed: %s", e
            )
            return "aborted"

        return "succeeded"


class TacPingerState(smach.State):
    def __init__(
        self,
        marker_topic: str = "/taluy/acoustic/hydrophone/marker",
        odom_frame: str = "odom",
        output_frame: str = "tac_pinger_link",
        capture_timeout: float = 5.0,
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["pinger_position"],
        )

        self.base_link = get_base_link()
        self.measurements = []
        self.marker_topic = marker_topic
        self.odom_frame = odom_frame
        self.output_frame = output_frame
        self.capture_timeout = capture_timeout

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"],
            output_keys=["pinger_position"],
        )

        with self.state_machine:
            smach.StateMachine.add(
                "SET_START_FRAME_MIDDLE",
                SetStartFrameState(
                    frame_name="pinger_listener_middle",
                    x=0.0,
                    y=0.0,
                    z=0.0,
                ),
                transitions={
                    "succeeded": "SET_START_FRAME_LEFT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_START_FRAME_LEFT",
                SetStartFrameState(
                    frame_name="pinger_listener_left",
                    x=0.0,
                    y=2.0,
                    z=0.0,
                ),
                transitions={
                    "succeeded": "SET_START_FRAME_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_START_FRAME_RIGHT",
                SetStartFrameState(
                    frame_name="pinger_listener_right",
                    x=0.0,
                    y=-2.0,
                    z=0.0,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_MIDDLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_MIDDLE",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="pinger_listener_middle",
                    cancel_on_success=True,
                ),
                transitions={
                    "succeeded": "CAPTURE_PINGER_MIDDLE",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CAPTURE_PINGER_MIDDLE",
                CapturePingerMeasurementState(
                    self.measurements,
                    marker_topic=self.marker_topic,
                    odom_frame=self.odom_frame,
                    timeout=self.capture_timeout,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_LEFT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_LEFT",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="pinger_listener_left",
                    cancel_on_success=True,
                ),
                transitions={
                    "succeeded": "CAPTURE_PINGER_LEFT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CAPTURE_PINGER_LEFT",
                CapturePingerMeasurementState(
                    self.measurements,
                    marker_topic=self.marker_topic,
                    odom_frame=self.odom_frame,
                    timeout=self.capture_timeout,
                ),
                transitions={
                    "succeeded": "ALIGN_FRAME_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ALIGN_FRAME_RIGHT",
                AlignFrame(
                    source_frame=self.base_link,
                    target_frame="pinger_listener_right",
                    cancel_on_success=True,
                ),
                transitions={
                    "succeeded": "CAPTURE_PINGER_RIGHT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CAPTURE_PINGER_RIGHT",
                CapturePingerMeasurementState(
                    self.measurements,
                    marker_topic=self.marker_topic,
                    odom_frame=self.odom_frame,
                    timeout=self.capture_timeout,
                ),
                transitions={
                    "succeeded": "ESTIMATE_PINGER_POSITION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ESTIMATE_PINGER_POSITION",
                EstimatePingerPositionState(
                    self.measurements,
                    output_frame=self.output_frame,
                    odom_frame=self.odom_frame,
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        self.measurements.clear()
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        if outcome == "succeeded":
            userdata.pinger_position = self.state_machine.userdata.pinger_position

        return outcome
