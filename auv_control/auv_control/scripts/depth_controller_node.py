#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse
import time
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from auv_common_lib.control.enable_state import ControlEnableHandler
from angles import normalize_angle


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
        self.cmd_vel_sub = rospy.Subscriber(
            "/taluy/cmd_vel", Twist, self.cmd_vel_callback
        )

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("/taluy/cmd_pose", Pose, queue_size=10)

        # use ControlEnableHandler
        self.control_enable_handler = ControlEnableHandler(1.0)

        # Initialize internal state
        self.target_depth = 0.0
        self.current_depth = 0.0
        self.target_heading = 0.0
        self.last_cmd_time = rospy.Time.now()

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

        if self.control_enable_handler.is_enabled():
            return

        quaterion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        _, _, self.target_heading = euler_from_quaternion(quaterion)

    def cmd_vel_callback(self, msg):
        if not self.control_enable_handler.is_enabled():
            return

        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
        dt = min(dt, 0.1)

        self.target_heading += msg.angular.z * dt
        self.last_cmd_time = rospy.Time.now()

    def control_loop(self, event):
        # Create and publish the cmd_pose message
        cmd_pose = Pose()
        cmd_pose.position.z = self.target_depth

        # self.target_heading = normalize_angle(self.target_heading)
        quaternion = quaternion_from_euler(0.0, 0.0, self.target_heading)

        cmd_pose.orientation.x = quaternion[0]
        cmd_pose.orientation.y = quaternion[1]
        cmd_pose.orientation.z = quaternion[2]
        cmd_pose.orientation.w = quaternion[3]

        self.cmd_pose_pub.publish(cmd_pose)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        depth_controller_node = DepthControllerNode()
        depth_controller_node.run()
    except rospy.ROSInterruptException:
        pass
