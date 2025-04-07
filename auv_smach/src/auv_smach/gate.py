from .initialize import *
import smach
import smach_ros
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from robot_localization.srv import SetPose, SetPoseRequest, SetPoseResponse
from auv_msgs.srv import (
    SetObjectTransform,
    SetObjectTransformRequest,
    SetObjectTransformResponse,
    AlignFrameController,
    AlignFrameControllerRequest,
    AlignFrameControllerResponse,
)
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_ros
import numpy as np
import tf.transformations as transformations
import tf2_geometry_msgs

from auv_smach.common import (
    NavigateToFrameState,
    SetAlignControllerTargetState,
    CancelAlignControllerState,
    SetDepthState,
)

from auv_smach.initialize import DelayState

class SearchForGateState(smach.State):
    def __init__(self, base_frame="taluy/base_link", target_frame="gate_search", check_frame="gate_blue_arrow_link", rotation_rate=0.2):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.base_frame = base_frame
        self.target_frame = target_frame
        self.check_frame = check_frame
        self.rotation_rate = rotation_rate 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.set_object_transform_service = rospy.ServiceProxy("set_object_transform", SetObjectTransform)
        self.rate = rospy.Rate(10)  

    def execute(self, userdata):
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"

            elapsed = rospy.Time.now().to_sec() - start_time
            current_angle = -self.rotation_rate * elapsed

            try:
                odom_to_base = self.tf_buffer.lookup_transform("odom", self.base_frame, rospy.Time(0), rospy.Duration(1.0))
            except Exception as e:
                rospy.logwarn("TF lookup failed: {}".format(e))
                return "aborted"

            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"
            t.child_frame_id = self.target_frame
            t.transform.translation.x = odom_to_base.transform.translation.x
            t.transform.translation.y = odom_to_base.transform.translation.y
            t.transform.translation.z = odom_to_base.transform.translation.z

            quaternion = transformations.quaternion_from_euler(0, 0, current_angle)
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]

            req = SetObjectTransformRequest()
            req.transform = t
            try:
                res = self.set_object_transform_service(req)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call failed: {}".format(e))
                return "aborted"
            if not res.success:
                rospy.logwarn("SetObjectTransform failed: {}".format(res.message))

            try:
                self.tf_buffer.lookup_transform("odom", self.check_frame, rospy.Time(0), rospy.Duration(0.5))
                rospy.loginfo("Check frame '{}' found, stopping rotation.".format(self.check_frame))
                return "succeeded"
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass

            if abs(current_angle) >= 2 * np.pi:
                return "succeeded"

            self.rate.sleep()


class TransformServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "set_transform_gate_trajectory",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class NavigateThroughGateState(smach.State):
    def __init__(self, gate_depth: float):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )
        with self.state_machine:
            smach.StateMachine.add(
                "SET_GATE_SEARCH",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="gate_search"
                ),
                transitions={"succeeded": "SEARCH_FOR_GATE", "preempted": "preempted", "aborted": "aborted"},
            )
            smach.StateMachine.add(
                "SEARCH_FOR_GATE",
                SearchForGateState(base_frame="taluy/base_link", target_frame="gate_search", check_frame="gate_blue_arrow_link", rotation_rate=0.2),
                transitions={"succeeded": "ENABLE_GATE_TRAJECTORY_PUBLISHER", "preempted": "preempted", "aborted": "aborted"},
            )
            smach.StateMachine.add(
                "ENABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=True),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=gate_depth, sleep_duration=3.0),
                transitions={
                    "succeeded": "SET_ALIGN_CONTROLLER_TARGET",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_ALIGN_CONTROLLER_TARGET",
                SetAlignControllerTargetState(
                    source_frame="taluy/base_link", target_frame="gate_target"
                ),
                transitions={
                    "succeeded": "DISABLE_GATE_TRAJECTORY_PUBLISHER",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "DISABLE_GATE_TRAJECTORY_PUBLISHER",
                TransformServiceEnableState(req=False),
                transitions={
                    "succeeded": "NAVIGATE_TO_GATE_START",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_GATE_START",
                NavigateToFrameState(
                    "taluy/base_link", "gate_enterance", "gate_target"
                ),
                transitions={
                    "succeeded": "NAVIGATE_TO_GATE_EXIT",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "NAVIGATE_TO_GATE_EXIT",
                NavigateToFrameState(
                    "gate_enterance", "gate_exit", "gate_target", n_turns=-1
                ),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            # smach.StateMachine.add(
            #     "CANCEL_ALIGN_CONTROLLER",
            #     CancelAlignControllerState(),
            #     transitions={
            #         "succeeded": "succeeded",
            #         "preempted": "preempted",
            #         "aborted": "aborted",
            #     },
            # )

    def execute(self, userdata):
        outcome = self.state_machine.execute()

        if outcome is None:
            return "preempted"

        return outcome
