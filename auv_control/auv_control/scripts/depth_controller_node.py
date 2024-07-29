#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import time

class DepthControllerNode:
    def __init__(self):
        rospy.init_node('depth_controller_node', anonymous=True)
        
        # Initialize subscribers
        self.enable_sub = rospy.Subscriber('/taluy/enable', Bool, self.enable_callback)
        self.target_depth_sub = rospy.Subscriber('/taluy/target_depth', Float32, self.target_depth_callback)
        self.odometry_sub = rospy.Subscriber('/taluy/odometry', Odometry, self.odometry_callback)
        
        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher('/taluy/cmd_pose', Pose, queue_size=10)
        
        # Initialize internal state
        self.enabled = False
        self.target_depth = 0.0
        self.current_depth = 0.0
        self.last_enable_time = time.time()
        
        # Parameters
        self.update_rate = rospy.get_param('~update_rate', 10) # Hz
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.1) # Adjust this for slower or faster transitions
        
        rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.control_loop)
        
    def enable_callback(self, msg):
        self.enabled = msg.data
        self.last_enable_time = time.time()

    def target_depth_callback(self, msg):
        self.target_depth = msg.data

    def odometry_callback(self, msg):
        self.current_depth = msg.pose.pose.position.z

    def control_loop(self, event):
        if self.enabled and (time.time() - self.last_enable_time < 1.0):
            # Calculate the smoothed target depth
            new_depth = self.current_depth + self.smoothing_factor * (self.target_depth - self.current_depth)
        else:
            # Timeout, hold the current depth
            new_depth = self.current_depth
            self.target_depth = self.current_depth

        # Create and publish the cmd_pose message
        cmd_pose = Pose()
        cmd_pose.position.z = new_depth
        cmd_pose.orientation.w = 1.0
        self.cmd_pose_pub.publish(cmd_pose)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        depth_controller_node = DepthControllerNode()
        depth_controller_node.run()
    except rospy.ROSInterruptException:
        pass
