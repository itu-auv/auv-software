import smach
import smach_ros
import rospy
from std_srvs.srv import Trigger, TriggerRequest
from auv_msgs.srv import (
    AlignFrameController,
    AlignFrameControllerRequest,
)
from geometry_msgs.msg import TransformStamped
import numpy as np
from auv_msgs.srv import SetDepth, SetDepthRequest
from auv_navigation import follow_path_action_client 

from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
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


class SetDepthState(smach_ros.ServiceState):
    def __init__(self, depth: float, sleep_duration=0.0):
        set_depth_request = SetDepthRequest()
        set_depth_request.target_depth = depth
        self.sleep_duration = sleep_duration

        smach_ros.ServiceState.__init__(
            self,
            "/taluy/set_depth",
            SetDepth,
            request=set_depth_request,
        )

    def execute(self, ud):
        return_data = super().execute(ud)

        if self.sleep_duration > 0:
            rospy.sleep(self.sleep_duration)

        return return_data


class LaunchTorpedoState(smach_ros.ServiceState):
    def __init__(self, id: int):
        smach_ros.ServiceState.__init__(
            self,
            f"/taluy/actuators/torpedo_{id}/launch",
            Trigger,
            request=TriggerRequest(),
        )


class DropBallState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "/taluy/actuators/ball_dropper/drop",
            Trigger,
            request=TriggerRequest(),
        )


class CancelAlignControllerState(smach_ros.ServiceState):
    def __init__(self):
        smach_ros.ServiceState.__init__(
            self,
            "/taluy/control/align_frame/cancel",
            Trigger,
            request=TriggerRequest(),
        )


class SetAlignControllerTargetState(smach_ros.ServiceState):
    def __init__(self, source_frame: str, target_frame: str):
        align_request = AlignFrameControllerRequest()
        align_request.source_frame = source_frame
        align_request.target_frame = target_frame
        align_request.angle_offset = 0.0

        smach_ros.ServiceState.__init__(
            self,
            "/taluy/control/align_frame/start",
            AlignFrameController,
            request=align_request,
        )


class NavigateToFrameState(smach.State):
    def __init__(self, source_frame, target_frame, n_turns=0):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.n_turns = n_turns
        self._client = None  
        

    def execute(self, userdata):
        
        # Lazy initialization of client
        if self._client is None:
            self._client = follow_path_action_client.FollowPathActionClient()
        try:
            if self.preempt_requested():
                rospy.logwarn("[NavigateToFrameState] Preempt requested")
                return "preempted"
            
            success = self._client.navigate_to_frame(
                self.source_frame, 
                self.target_frame,
                n_turns=self.n_turns,
            )
            return "succeeded" if success else "aborted"
                
        except Exception as e:
            rospy.logwarn("[NavigateToFrameState] Error: %s", str(e))
            return "aborted"
        
    def request_preempt(self):
        """Handle preemption request."""
        smach.State.request_preempt(self)
        if self._client:
            self._client.cancel_current_goal()
