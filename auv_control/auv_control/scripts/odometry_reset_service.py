#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from geometry_msgs.msg import PoseWithCovarianceStamped
from robot_localization.srv import SetPose, SetPoseRequest


def handle_reset_request(req):
    """
    Service handler that calls set_pose and reset_heading services.
    """
    rospy.loginfo("Odometry reset request received.")
    try:
        # Wait for services to be available
        rospy.wait_for_service("set_pose", timeout=1.0)
        rospy.wait_for_service("reset_heading", timeout=1.0)

        # Create service proxies
        set_pose_client = rospy.ServiceProxy("set_pose", SetPose)
        reset_heading_client = rospy.ServiceProxy("reset_heading", Trigger)

        # Call /taluy/set_pose
        rospy.loginfo("Calling set_pose service.")
        set_pose_req = SetPoseRequest()
        set_pose_req.pose = PoseWithCovarianceStamped()
        set_pose_req.pose.header.stamp = rospy.Time.now()
        set_pose_req.pose.header.frame_id = "odom"
        set_pose_client(set_pose_req)
        rospy.loginfo("Called set_pose service.")

        # Call /taluy/reset_heading
        rospy.loginfo("Calling reset_heading service.")
        reset_heading_response = reset_heading_client(TriggerRequest())
        rospy.loginfo("Called reset_heading service.")

        if reset_heading_response.success:
            return TriggerResponse(success=True, message="Odometry reset successfully.")
        else:
            return TriggerResponse(
                success=False,
                message=f"Failed to reset heading: {reset_heading_response.message}",
            )

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return TriggerResponse(success=False, message=f"Service call failed: {e}")
    except rospy.ROSException as e:
        rospy.logerr(f"Could not connect to services: {e}")
        return TriggerResponse(
            success=False, message=f"Could not connect to services: {e}"
        )
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
        return TriggerResponse(
            success=False, message=f"An unexpected error occurred: {e}"
        )


def odometry_reset_service():
    """
    Initializes the ROS node and service.
    """
    rospy.init_node("odometry_reset_service_node")
    s = rospy.Service("odometry_reset", Trigger, handle_reset_request)
    rospy.loginfo("Odometry reset service ready.")
    rospy.spin()


if __name__ == "__main__":
    try:
        odometry_reset_service()
    except rospy.ROSInterruptException:
        pass
