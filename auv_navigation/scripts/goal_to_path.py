#!/usr/bin/env python3.8

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class GoalPathPublisher:
    def __init__(self):
        rospy.init_node('goal_path_publisher', anonymous=True)
        self.goal_sub = rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_callback)
        self.path_pub = rospy.Publisher('target_path', Path, queue_size=10)

    def goal_callback(self, msg):
        # Create a Path message with a single PoseStamped
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "odom"  # Assuming the frame_id is "odom", change as needed
        
        msg.header.frame_id = "odom"
        # Add the received PoseStamped as the only point in the path
        path.poses.append(msg)

        # Publish the path
        self.path_pub.publish(path)
        rospy.loginfo("Published path with single goal point.")

if __name__ == '__main__':
    try:
        GoalPathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
