#!/usr/bin/env python3

import rospy
import actionlib
from nav_msgs.msg import Path
from auv_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult
from auv_msgs.srv import AlignFrameController
from std_srvs.srv import Trigger, TriggerRequest


class PathFollowActionServer:
    def __init__(self):
        rospy.init_node("path_follow_action_server")

        self.server = actionlib.SimpleActionServer(
            "follow_path", FollowPathAction, self.execute_callback, auto_start=False
        )
        self.server.start()

        self.frame_align_service = rospy.ServiceProxy(
            "frame_alignment_controller", AlignFrameController
        )

        self.cancel_frame_align_service = rospy.ServiceProxy("cancel_control", Trigger)

        self.feedback = FollowPathFeedback()
        self.result = FollowPathResult()

        # lookahead distance
        self.carrot_distance = 1.0  
        self.rate = rospy.Rate(10) 

    def execute_callback(self, goal):
        rospy.loginfo("Received a new path to follow")
        path: Path = goal.path
        if not path.poses:
            rospy.logerr("Path is empty. Aborting.")
            self.result.success = False
            self.result.message = "Path is empty"
            self.server.set_aborted(self.result)
            return

        total_poses = len(path.poses)
        current_index = 0

        while current_index < total_poses and not rospy.is_shutdown():
            # preemption check
            if self.server.is_preempt_requested():
                rospy.loginfo("Path following preempted")

                self.call_service_if_available(
                    self.cancel_frame_align_service,
                    "Frame alignment control canceled",
                    "Failed to cancel frame alignment control",
                )
                self.server.set_preempted()
                return

            # progress feedback
            self.feedback.percentage_completed = (current_index / total_poses) * 100
            self.server.publish_feedback(self.feedback)

            # use align_frame_controller to align the frame with the target pose
            target_pose = path.poses[current_index]

            # TODO: this should probably not be a service,
            # TODO: maybe a topic or just a class we run to calculate the transform and cmd_vel
            # TODO: or we can integrate this logic into the align_frame_controller we already have isntead of another node
            self.call_service_if_available(
                self.frame_align_service,
                "Frame alignment control started",
                "Failed to start frame alignment control",
            )

            # check if we are close enough to the target
            if self.reached_pose(target_pose):
                current_index += 1

            self.rate.sleep()

        # task complete
        self.result.success = True
        self.result.message = "Successfully followed the path"
        rospy.loginfo(self.result.message)
        self.server.set_succeeded(self.result)

    def reached_pose(self, target_pose, threshold=0.1):
        # placeholder
        return (
            abs(target_pose.pose.position.x) < threshold
            and abs(target_pose.pose.position.y) < threshold
        )

    # copied from auv_teleop/joy_manager.py
    def call_service_if_available(self, service, success_message, failure_message):
        try:
            service.wait_for_service(timeout=1)
            response = service(TriggerRequest())
            if response.success:
                rospy.loginfo(success_message)
            else:
                rospy.logwarn(failure_message)
        except rospy.exceptions.ROSException:
            rospy.logwarn(f"Service {service.resolved_name} is not available")


if __name__ == "__main__":
    try:
        PathFollowActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
