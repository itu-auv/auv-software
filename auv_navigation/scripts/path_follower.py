#!/usr/bin/env python3.8

import rospy
import math
import tf
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String


class PathFollower:
    def __init__(self):
        rospy.init_node('path_follower', anonymous=True)

        # Subscribers
        self.path_sub = rospy.Subscriber('target_path', Path, self.path_callback)
        self.odom_sub = rospy.Subscriber('odometry', Odometry, self.odom_callback)
        self.mode_sub = rospy.Subscriber('follow_mode', String, self.mode_callback)
        self.cancel_sub = rospy.Subscriber('cancel_path', String, self.cancel_callback)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # Internal state
        self.path = []
        self.current_index = 0
        self.current_pose = None
        self.follow_mode = 'aim'  # Default mode: 'aim' or 'align'
        self.cancelled = False
        self.hold_last_position = False

        # Parameters
        self.goal_threshold = 0.5  # Distance to waypoint to consider it reached
        self.kp_linear = 0.3  # Linear speed proportional gain
        self.kp_angular = 0.8  # Angular speed proportional gain
        self.kd_linear = 0.1  # Linear speed derivative gain
        self.kd_angular = 0.3  # Angular speed derivative gain

        self.previous_error_x = 0
        self.previous_error_y = 0
        self.previous_error_yaw = 0
        self.previous_time = rospy.Time.now()

        self.rate = rospy.Rate(10)

        self.control_loop()

    def path_callback(self, msg):
        if self.cancelled:
            return  # Ignore new paths if cancelled

        self.path = msg.poses
        self.current_index = 0
        self.hold_last_position = False
        self.cancelled = False
        rospy.loginfo(f"Received new path with {len(self.path)} points.")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def mode_callback(self, msg):
        if msg.data in ['aim', 'align']:
            self.follow_mode = msg.data
            rospy.loginfo(f"Set follow mode to: {self.follow_mode}")
        else:
            rospy.logwarn(f"Invalid follow mode: {msg.data}. Use 'aim' or 'align'.")

    def cancel_callback(self, msg):
        self.cancelled = True
        self.hold_last_position = False
        self.stop()
        rospy.loginfo("Path following cancelled.")

    def control_loop(self):
        while not rospy.is_shutdown():
            if self.path and self.current_pose and not self.cancelled:
                self.navigate_to_waypoint()
            elif self.hold_last_position and self.current_pose:
                self.hold_position()
            self.rate.sleep()

    def navigate_to_waypoint(self):
        if self.current_index >= len(self.path):
            rospy.loginfo("Reached end of path. Holding position.")
            self.hold_last_position = True
            return

        target_pose = self.path[self.current_index].pose
        distance = self.calculate_distance(self.current_pose.position, target_pose.position)

        if distance < self.goal_threshold:
            if len(self.path) == 1:
                rospy.loginfo("Single-point path. Holding position.")
                self.hold_last_position = True
            else:
                self.current_index += 1
                rospy.loginfo(f"Reached waypoint {self.current_index}. Moving to next.")
            return

        # Transform target position to robot's body frame
        transformed_position = self.transform_to_body_frame(self.current_pose, target_pose.position)

        # Calculate control commands using PD controllers
        cmd_vel = self.calculate_cmd_vel(transformed_position)

        self.cmd_vel_pub.publish(cmd_vel)

    def hold_position(self):
        # Continuously control the robot to maintain its last pose
        target_pose = self.path[-1].pose

        # Transform target position to robot's body frame
        transformed_position = self.transform_to_body_frame(self.current_pose, target_pose.position)

        # Calculate control commands using PD controllers
        cmd_vel = self.calculate_cmd_vel(transformed_position)

        self.cmd_vel_pub.publish(cmd_vel)

    def transform_to_body_frame(self, current_pose, target_position):
        # Calculate the difference between the target and current position
        dx = target_position.x - current_pose.position.x
        dy = target_position.y - current_pose.position.y

        # Get current yaw
        current_yaw = self.get_yaw_from_quaternion(current_pose.orientation)

        # Transform to the robot's body frame
        transformed_x = math.cos(current_yaw) * dx + math.sin(current_yaw) * dy
        transformed_y = -math.sin(current_yaw) * dx + math.cos(current_yaw) * dy

        return transformed_x, transformed_y

    def calculate_cmd_vel(self, transformed_position):
        current_time = rospy.Time.now()
        dt = (current_time - self.previous_time).to_sec()

        error_x = transformed_position[0]
        error_y = transformed_position[1]
        error_yaw = math.atan2(error_y, error_x)  # Yaw error

        # PD control for x and y axis
        cmd_vel = Twist()
        cmd_vel.linear.x = self.kp_linear * error_x + self.kd_linear * (error_x - self.previous_error_x) / dt
        cmd_vel.linear.y = self.kp_linear * error_y + self.kd_linear * (error_y - self.previous_error_y) / dt

        # PD control for angular (yaw) axis
        cmd_vel.angular.z = self.kp_angular * self.normalize_angle(error_yaw) + \
                            self.kd_angular * (self.normalize_angle(error_yaw) - self.previous_error_yaw) / dt

        # Limit linear and angular speeds
        cmd_vel.linear.x = max(min(cmd_vel.linear.x, 0.3), -0.3)
        cmd_vel.linear.y = max(min(cmd_vel.linear.y, 0.3), -0.3)
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, 0.5), -0.5)

        # Save previous errors and time
        self.previous_error_x = error_x
        self.previous_error_y = error_y
        self.previous_error_yaw = self.normalize_angle(error_yaw)
        self.previous_time = current_time

        return cmd_vel

    def calculate_distance(self, current_position, target_position):
        return math.sqrt((target_position.x - current_position.x)**2 +
                         (target_position.y - current_position.y)**2)

    def get_yaw_from_quaternion(self, orientation):
        # Convert quaternion to Euler angles
        siny_cosp = 2 * (orientation.w * orientation.z +
                         orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y +
                             orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def stop(self):
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)


if __name__ == '__main__':
    try:
        PathFollower()
    except rospy.ROSInterruptException:
        pass
