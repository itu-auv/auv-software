#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from auv_common_lib.control.enable_state import ControlEnableHandler


class ReferencePosePublisherNode:
    def __init__(self):
        # Initialize subscribers
        self.odometry_sub = rospy.Subscriber(
            "odometry", Odometry, self.odometry_callback, tcp_nodelay=True
        )
        self.set_depth_service = rospy.Service(
            "set_depth", SetDepth, self.target_depth_handler
        )
        self.cmd_vel_sub = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
        )

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("cmd_pose", Pose, queue_size=10)

        self.control_enable_handler = ControlEnableHandler(1.0)

        # Internal state
        self.target_depth = -0.4
        self.target_heading = 0.0
        self.current_depth = 0.0  # updated from odometry
        self.last_cmd_time = rospy.Time.now()

        # Parameters
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10)) # rate of loop for control and alignment check
        self.command_timeout = rospy.get_param("~command_timeout", 0.1)
        self.alignment_threshold = rospy.get_param("~alignment_threshold", 0.1)
        self.alignment_timeout = rospy.get_param("~alignment_timeout", 10.0)
        
    def is_aligned(self, target_depth: float) -> bool:
        """
        Check whether the current depth is within the allowed error range of the target.
        """
        error = abs(target_depth - self.current_depth)
        rospy.loginfo("is_aligned: target_depth=%.2f, current_depth=%.2f, error=%.2f (allowed: %.2f)",
                       target_depth, self.current_depth, error, self.alignment_threshold) #!delete
        return error < self.alignment_threshold

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        """
        Set the new target depth immediately and update the command pose.
        Then wait for up to self.alignment_timeout for the vehicle to align with the target depth.
        If alignment is achieved within the period, return success.
        Otherwise, return failure.
        """
        # Set the target depth immediately
        self.target_depth = req.target_depth

        # Define the timeout and check frequency
        timeout_duration = self.alignment_timeout  
        start_time = rospy.Time.now()

        # Wait until either alignment is achieved or the timeout is reached
        while (rospy.Time.now() - start_time).to_sec() < self.alignment_timeout:
            if self.is_aligned(req.target_depth):
                return SetDepthResponse(
                    success=True,
                    message=f"Target depth set to {req.target_depth:.2f} and aligned. "
                            f"Current depth: {self.current_depth:.2f}"
                )
            self.loop_rate.sleep()  # sleep to maintain the loop rate

        # alignment was not achieved within period
        return SetDepthResponse(
            success=False,
            message=f"Target depth set to {req.target_depth:.2f} but failed to align within {timeout_duration} seconds. "
                    f"Current depth: {self.current_depth:.2f}"
        )

    def odometry_callback(self, msg):
        # Always update current depth from odometry
        self.current_depth = msg.pose.pose.position.z

        if self.control_enable_handler.is_enabled():
            return

        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        _, _, self.target_heading = euler_from_quaternion(quaternion)

    def cmd_vel_callback(self, msg):
        if not self.control_enable_handler.is_enabled():
            return

        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
        dt = min(dt, self.command_timeout)
        self.target_depth += msg.linear.z * dt
        self.target_heading += msg.angular.z * dt
        self.last_cmd_time = rospy.Time.now()

    def control_loop(self):
        # Create and publish the cmd_pose message with the desired depth and heading
        cmd_pose = Pose()
        cmd_pose.position.z = self.target_depth  # Set to the target depth

        quaternion = quaternion_from_euler(0.0, 0.0, self.target_heading)
        cmd_pose.orientation.x = quaternion[0]
        cmd_pose.orientation.y = quaternion[1]
        cmd_pose.orientation.z = quaternion[2]
        cmd_pose.orientation.w = quaternion[3]

        self.cmd_pose_pub.publish(cmd_pose)

    def run(self):
        while not rospy.is_shutdown():
            self.control_loop()
            self.loop_rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("reference_pose_publisher_node")
        node = ReferencePosePublisherNode()
        node.run()
    except rospy.ROSInterruptException:
        pass