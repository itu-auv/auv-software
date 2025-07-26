#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetDepthResponse,
    SetHeading,
    SetHeadingResponse,
)
from robot_localization.srv import SetPose, SetPoseRequest
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from auv_common_lib.control.enable_state import ControlEnableHandler


class ReferencePosePublisherNode:
    def __init__(self):
        self.heading_control_mode = rospy.get_param("~heading_control_mode", "cmd_vel")

        # Initialize subscribers
        self.odometry_sub = rospy.Subscriber(
            "odometry", Odometry, self.odometry_callback, tcp_nodelay=True
        )
        self.set_depth_service = rospy.Service(
            "set_depth", SetDepth, self.target_depth_handler
        )
        if self.heading_control_mode == "cmd_vel":
            self.cmd_vel_sub = rospy.Subscriber(
                "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
            )
        elif self.heading_control_mode == "set_heading":
            self.set_heading_service = rospy.Service(
                "set_heading", SetHeading, self.target_heading_handler
            )
        self.reset_odometry_service = rospy.Service(
            "reset_odometry", Trigger, self.reset_odometry_handler
        )
        self.set_pose_client = rospy.ServiceProxy("set_pose", SetPose)

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("cmd_pose", PoseStamped, queue_size=10)

        self.control_enable_handler = ControlEnableHandler(1.0)

        # Initialize internal state
        self.target_depth = -0.4
        self.target_heading = 0.0
        self.last_cmd_time = rospy.Time.now()
        self.target_frame_id = ""
        self.is_resetting = False

        self.set_pose_req = SetPoseRequest()
        self.set_pose_req.pose = PoseWithCovarianceStamped()
        self.set_pose_req.pose.header.stamp = rospy.Time.now()
        self.set_pose_req.pose.header.frame_id = "odom"

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.command_timeout = rospy.get_param("~command_timeout", 0.1)

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        self.target_depth = req.target_depth
        self.target_frame_id = req.frame_id
        return SetDepthResponse(
            success=True,
            message=f"Target depth set to {self.target_depth} in frame {self.target_frame_id}",
        )

    def target_heading_handler(self, req: SetHeading):
        self.target_heading = req.heading
        return SetHeadingResponse(
            success=True,
            message=f"Target heading set to {self.target_heading}",
        )

    def reset_odometry_handler(self, req):
        if self.is_resetting:
            return TriggerResponse(
                success=False, message="Odometry reset already in progress."
            )

        self.is_resetting = True
        rospy.logdebug("Starting odometry reset.")

        try:
            self.set_pose_client(self.set_pose_req)
            rospy.logdebug("Called set_pose service.")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"Service call failed: {e}")
            self.is_resetting = False
            return TriggerResponse(success=False, message=f"Service call failed: {e}")

        rospy.logdebug("Waiting for heading to settle.")
        rospy.sleep(2.0)

        self.is_resetting = False
        rospy.logdebug("Odometry reset finished.")
        return TriggerResponse(success=True, message="Odometry reset successfully.")

    def odometry_callback(self, msg):
        if self.control_enable_handler.is_enabled() and not self.is_resetting:
            return

        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        _, _, self.target_heading = euler_from_quaternion(quaternion)

    def cmd_vel_callback(self, msg):
        if (not self.control_enable_handler.is_enabled()) or self.is_resetting:
            return

        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
        dt = min(dt, self.command_timeout)
        self.target_depth += msg.linear.z * dt
        self.target_heading += msg.angular.z * dt
        self.last_cmd_time = rospy.Time.now()

    def control_loop(self):
        # Create and publish the cmd_pose message
        cmd_pose_stamped = PoseStamped()

        cmd_pose_stamped.pose.position.z = self.target_depth
        cmd_pose_stamped.header.frame_id = self.target_frame_id
        quaternion = quaternion_from_euler(0.0, 0.0, self.target_heading)
        cmd_pose_stamped.pose.orientation.x = quaternion[0]
        cmd_pose_stamped.pose.orientation.y = quaternion[1]
        cmd_pose_stamped.pose.orientation.z = quaternion[2]
        cmd_pose_stamped.pose.orientation.w = quaternion[3]

        self.cmd_pose_pub.publish(cmd_pose_stamped)

    def run(self):
        rate = rospy.Rate(self.update_rate)

        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("reference_pose_publisher_node")
        reference_pose_publisher_node = ReferencePosePublisherNode()
        reference_pose_publisher_node.run()
    except rospy.ROSInterruptException:
        pass
