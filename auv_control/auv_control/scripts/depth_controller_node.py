#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse
import time


class DepthControllerNode:
    def __init__(self):
        rospy.init_node("depth_controller_node")

        # Initialize subscribers
        self.odometry_sub = rospy.Subscriber(
            "/taluy/odometry", Odometry, self.odometry_callback
        )
        self.set_depth_service = rospy.Service(
            "/taluy/set_depth", SetDepth, self.target_depth_handler
        )

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("/taluy/cmd_pose", Pose, queue_size=10)

        # Initialize internal state
        self.target_depth = 0.0
        self.current_depth = 0.0

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.tau = rospy.get_param("~tau", 2.0)
        self.dt = 1.0 / self.update_rate
        self.alpha = self.dt / (self.tau + self.dt)

        rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.control_loop)

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        self.target_depth = req.target_depth
        return SetDepthResponse()

    def odometry_callback(self, msg):
        self.current_depth = msg.pose.pose.position.z

    def control_loop(self, event):
        # Create and publish the cmd_pose message
        cmd_pose = Pose()
        cmd_pose.position.z = self.target_depth
        cmd_pose.orientation.w = 1.0
        self.cmd_pose_pub.publish(cmd_pose)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        depth_controller_node = DepthControllerNode()
        depth_controller_node.run()
    except rospy.ROSInterruptException:
        pass
