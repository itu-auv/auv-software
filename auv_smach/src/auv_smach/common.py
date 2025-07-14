import smach
import smach_ros
import rospy
import threading
import numpy as np
import tf2_ros
import tf.transformations as transformations
import math
import angles

from std_srvs.srv import Trigger, TriggerRequest, SetBool, SetBoolRequest
from auv_msgs.srv import AlignFrameController, AlignFrameControllerRequest
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped

from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetObjectTransform,
    SetObjectTransformRequest,
    SetObjectTransformResponse,
)

from auv_msgs.srv import SetDepth, SetDepthRequest
from auv_msgs.srv import VisualServoing, VisualServoingRequest

from auv_navigation.follow_path_action import follow_path_client

from auv_navigation.path_planning.path_planners import PathPlanners

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
    quaternion_from_euler,
    euler_from_quaternion,
)


def transform_to_matrix(transform):
    trans = translation_matrix(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
    )
    rot = quaternion_matrix(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )
    return np.dot(trans, rot)


def multiply_transforms(transform1, transform2):
    matrix1 = transform_to_matrix(transform1)
    matrix2 = transform_to_matrix(transform2)
    return np.dot(matrix1, matrix2)


def matrix_to_transform(matrix):
    trans = translation_from_matrix(matrix)
    rot = quaternion_from_matrix(matrix)

    transform = TransformStamped()
    transform.transform.translation.x = trans[0]
    transform.transform.translation.y = trans[1]
    transform.transform.translation.z = trans[2]
    transform.transform.rotation.x = rot[0]
    transform.transform.rotation.y = rot[1]
    transform.transform.rotation.z = rot[2]
    transform.transform.rotation.w = rot[3]

    return transform


def concatenate_transforms(transform1, transform2):
    combined_matrix = multiply_transforms(transform1.transform, transform2.transform)
    return matrix_to_transform(combined_matrix)


# ------------------- STATES -------------------


class VisualServoingCentering(smach_ros.ServiceState):
    def __init__(self, target_prop):
        request = VisualServoingRequest()
        request.target_prop = target_prop
        super(VisualServoingCentering, self).__init__(
            "visual_servoing/start",
            VisualServoing,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class VisualServoingNavigation(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingNavigation, self).__init__(
            "visual_servoing/navigate",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )


class SetDepthState(smach_ros.ServiceState):
    """
    Calls /taluy/set_depth with the requested depth.
    continuously publishes True to /taluy/enable topic
    whilst the state is running.

    Outcomes:
        - succeeded: The service call returned success.
        - preempted: The state was preempted.
        - aborted: The service call failed.
    """

    def __init__(self, depth: float, sleep_duration: float = 5.0, frame_id: str = ""):
        set_depth_request = SetDepthRequest()
        set_depth_request.target_depth = depth
        set_depth_request.frame_id = frame_id
        self.sleep_duration = sleep_duration

        super(SetDepthState, self).__init__(
            "set_depth",
            SetDepth,
            request=set_depth_request,
            outcomes=["succeeded", "preempted", "aborted"],
        )

        self._stop_publishing = threading.Event()

        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

    def _publish_enable_loop(self):
        publish_rate = rospy.get_param("~enable_rate", 20)
        rate = rospy.Rate(publish_rate)
        while not rospy.is_shutdown() and not self._stop_publishing.is_set():
            self.enable_pub.publish(Bool(True))
            rate.sleep()

    def execute(self, userdata):
        # if there's an immediate preempt
        if self.preempt_requested():
            rospy.logwarn("[SetDepthState] Preempt requested before execution.")
            self.service_preempt()
            return "preempted"

        # Call the service
        result = super(SetDepthState, self).execute(userdata)

        # Clear the stop flag
        self._stop_publishing.clear()
        # start publishing in the background thread
        pub_thread = threading.Thread(target=self._publish_enable_loop)
        pub_thread.start()
        # Wait for the specified sleep duration
        if self.sleep_duration > 0:
            rospy.sleep(self.sleep_duration)

        # signal the publishing thread to stop
        self._stop_publishing.set()
        pub_thread.join()

        return result


class ClearObjectMapState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "clear_object_transforms",
            Trigger,
            request=TriggerRequest(),
        )


class SetDetectionState(smach_ros.ServiceState):
    """
    Calls the service to enable or disable camera detections.
    """

    def __init__(self, camera_name: str, enable: bool):
        if camera_name not in ["front", "bottom"]:
            raise ValueError("camera_name must be 'front' or 'bottom'")

        service_name = f"enable_{camera_name}_camera_detections"
        request = SetBoolRequest(data=enable)

        super(SetDetectionState, self).__init__(
            service_name,
            SetBool,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )
