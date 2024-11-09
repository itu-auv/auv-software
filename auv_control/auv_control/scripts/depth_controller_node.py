#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse



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
        self.cmd_pose_pub = rospy.Publisher(
            "/taluy/cmd_pose_stamped", PoseStamped, queue_size=10)

        # Initialize internal state
        self.target_depth = 0.0
        self.current_depth = 0.0
        self.target_frame_id = ""  

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.tau = rospy.get_param("~tau", 2.0)
        self.dt = 1.0 / self.update_rate
        self.alpha = self.dt / (self.tau + self.dt)
        # Set up a timer to call the control_loop method at a rate defined by update_rate
        rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.control_loop)

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
    
        self.target_frame_id = req.frame_id  # If the target frame_id is not given, simply empty the frame_id.
        # We get the target frame_id as empty string from the service request if not given anything. 
        self.target_depth = req.target_depth
        
        return SetDepthResponse(success=True, message="Depth set successfully")


    def odometry_callback(self, msg):
        self.current_depth = msg.pose.pose.position.z
        rospy.logdebug(f"Current depth updated: {self.current_depth}")
        
    def control_loop(self, event):
        # Create and publish the cmd_pose message        
        cmd_pose_stamped = PoseStamped()
        cmd_pose_stamped.header.frame_id = self.target_frame_id
        cmd_pose_stamped.pose.position.z = self.target_depth
        cmd_pose_stamped.pose.orientation.w = 1.0
        self.cmd_pose_pub.publish(cmd_pose_stamped)
        rospy.logdebug(f"Published cmd_pose with target depth: {self.target_depth} and frame_id: {self.target_frame_id} to /taluy/cmd_pose")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        depth_controller_node = DepthControllerNode()
        depth_controller_node.run()
    except rospy.ROSInterruptException:
        pass