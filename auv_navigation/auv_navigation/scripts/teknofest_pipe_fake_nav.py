#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import tf2_ros
import math
import numpy as np
import threading
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class MoveForwardState(smach.State):
    """State to move forward for a specified distance"""

    def __init__(self, distance, linear_velocity=0.2):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.distance = distance
        self.linear_velocity = linear_velocity
        self.cmd_vel_pub = rospy.Publisher("/taluy/cmd_vel", Twist, queue_size=1)
        self.enable_pub = rospy.Publisher("/taluy/enable", Bool, queue_size=1)
        self.odom_sub = None
        self.initial_position = None
        self.current_position = None
        self.rate = rospy.Rate(20)  # 20 Hz

    def odom_callback(self, msg):
        """Callback to track vehicle position"""
        self.current_position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )

    def execute(self, userdata):
        rospy.loginfo(f"MoveForwardState: Moving forward {self.distance} meters")

        # Subscribe to odometry
        self.odom_sub = rospy.Subscriber("odometry", Odometry, self.odom_callback)

        # Wait for initial position
        while self.current_position is None and not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            self.rate.sleep()

        self.initial_position = self.current_position.copy()

        # Create twist message for forward movement
        twist = Twist()
        twist.linear.x = self.linear_velocity

        # Move forward until distance is reached
        while not rospy.is_shutdown():
            if self.preempt_requested():
                # Stop the vehicle
                twist.linear.x = 0.0
                self.cmd_vel_pub.publish(twist)
                self.service_preempt()
                return "preempted"

            # Calculate distance traveled
            if self.current_position is not None:
                distance_traveled = np.linalg.norm(
                    self.current_position - self.initial_position
                )

                if distance_traveled >= self.distance:
                    # Stop the vehicle
                    twist.linear.x = 0.0
                    self.cmd_vel_pub.publish(twist)
                    rospy.loginfo(
                        f"MoveForwardState: Completed {self.distance}m movement"
                    )
                    break

            # Publish movement command and enable signal
            self.cmd_vel_pub.publish(twist)
            self.enable_pub.publish(Bool(True))
            self.rate.sleep()

        # Clean up
        if self.odom_sub:
            self.odom_sub.unregister()

        return "succeeded"


class TurnState(smach.State):
    """State to turn left or right by a specified angle"""

    def __init__(self, angle_degrees, angular_velocity=0.1):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.target_angle = math.radians(angle_degrees)  # Convert to radians
        self.angular_velocity = (
            angular_velocity if angle_degrees > 0 else -angular_velocity
        )
        self.cmd_vel_pub = rospy.Publisher("/taluy/cmd_vel", Twist, queue_size=1)
        self.enable_pub = rospy.Publisher("/taluy/enable", Bool, queue_size=1)
        self.odom_sub = None
        self.initial_yaw = None
        self.current_yaw = None
        self.total_rotation = 0.0
        self.previous_yaw = None
        self.rate = rospy.Rate(20)  # 20 Hz

    def odom_callback(self, msg):
        """Callback to track vehicle orientation"""
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.current_yaw = yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def execute(self, userdata):
        direction = "left" if self.target_angle > 0 else "right"
        rospy.loginfo(
            f"TurnState: Turning {direction} {math.degrees(abs(self.target_angle))} degrees"
        )

        # Subscribe to odometry
        self.odom_sub = rospy.Subscriber("odometry", Odometry, self.odom_callback)

        # Wait for initial orientation
        while self.current_yaw is None and not rospy.is_shutdown():
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            self.rate.sleep()

        self.initial_yaw = self.current_yaw
        self.previous_yaw = self.current_yaw
        self.total_rotation = 0.0

        # Create twist message for rotation
        twist = Twist()
        twist.angular.z = self.angular_velocity

        # Rotate until target angle is reached
        while not rospy.is_shutdown():
            if self.preempt_requested():
                # Stop the vehicle
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                self.service_preempt()
                return "preempted"

            # Calculate total rotation
            if self.current_yaw is not None and self.previous_yaw is not None:
                angle_diff = self.normalize_angle(self.current_yaw - self.previous_yaw)
                self.total_rotation += angle_diff
                self.previous_yaw = self.current_yaw

                if abs(self.total_rotation) >= abs(self.target_angle):
                    # Stop the vehicle
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    rospy.loginfo(
                        f"TurnState: Completed {math.degrees(abs(self.target_angle))}° turn"
                    )
                    break

            # Publish rotation command and enable signal
            self.cmd_vel_pub.publish(twist)
            self.enable_pub.publish(Bool(True))
            self.rate.sleep()

        # Clean up
        if self.odom_sub:
            self.odom_sub.unregister()

        return "succeeded"


class PipelineNavigationStateMachine(smach.StateMachine):
    """State machine for navigating the pipeline path"""

    def __init__(self):
        smach.StateMachine.__init__(
            self, outcomes=["succeeded", "preempted", "aborted"]
        )

        with self:
            # 1. Move forward 6 meters
            smach.StateMachine.add(
                "MOVE_FORWARD_6M",
                MoveForwardState(distance=6.0, linear_velocity=0.2),
                transitions={
                    "succeeded": "TURN_LEFT_90_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 2. Turn left 90 degrees
            smach.StateMachine.add(
                "TURN_LEFT_90_1",
                TurnState(angle_degrees=90, angular_velocity=0.1),
                transitions={
                    "succeeded": "MOVE_FORWARD_2_5M",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 3. Move forward 2.5 meters
            smach.StateMachine.add(
                "MOVE_FORWARD_2_5M",
                MoveForwardState(distance=2.5, linear_velocity=0.2),
                transitions={
                    "succeeded": "TURN_LEFT_90_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 4. Turn left 90 degrees
            smach.StateMachine.add(
                "TURN_LEFT_90_2",
                TurnState(angle_degrees=90, angular_velocity=0.1),
                transitions={
                    "succeeded": "MOVE_FORWARD_2M",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 5. Move forward 2 meters
            smach.StateMachine.add(
                "MOVE_FORWARD_2M",
                MoveForwardState(distance=2.0, linear_velocity=0.2),
                transitions={
                    "succeeded": "TURN_LEFT_90_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 6. Turn left 90 degrees
            smach.StateMachine.add(
                "TURN_LEFT_90_3",
                TurnState(angle_degrees=90, angular_velocity=0.1),
                transitions={
                    "succeeded": "MOVE_FORWARD_1_5M_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 7. Move forward 1.5 meters
            smach.StateMachine.add(
                "MOVE_FORWARD_1_5M_1",
                MoveForwardState(distance=1.5, linear_velocity=0.1),
                transitions={
                    "succeeded": "TURN_RIGHT_90_1",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 8. Turn right 90 degrees
            smach.StateMachine.add(
                "TURN_RIGHT_90_1",
                TurnState(angle_degrees=-90, angular_velocity=0.1),
                transitions={
                    "succeeded": "MOVE_FORWARD_1_5M_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 9. Move forward 1.5 meters
            smach.StateMachine.add(
                "MOVE_FORWARD_1_5M_2",
                MoveForwardState(distance=1.5, linear_velocity=0.2),
                transitions={
                    "succeeded": "TURN_RIGHT_90_2",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 10. Turn right 90 degrees
            smach.StateMachine.add(
                "TURN_RIGHT_90_2",
                TurnState(angle_degrees=-90, angular_velocity=0.1),
                transitions={
                    "succeeded": "MOVE_FORWARD_1M",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 11. Move forward 1 meter
            smach.StateMachine.add(
                "MOVE_FORWARD_1M",
                MoveForwardState(distance=1.0, linear_velocity=0.2),
                transitions={
                    "succeeded": "TURN_LEFT_90_4",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 12. Turn left 90 degrees
            smach.StateMachine.add(
                "TURN_LEFT_90_4",
                TurnState(angle_degrees=90, angular_velocity=0.1),
                transitions={
                    "succeeded": "MOVE_FORWARD_2M_FINAL",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

            # 13. Move forward 2 meters (final segment)
            smach.StateMachine.add(
                "MOVE_FORWARD_2M_FINAL",
                MoveForwardState(distance=2.0, linear_velocity=0.2),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )


def main():
    rospy.init_node("teknofest_pipe_fake_nav")

    # Create the state machine
    sm = PipelineNavigationStateMachine()

    # Create and start the introspection server (for visualization)
    sis = smach_ros.IntrospectionServer("pipeline_navigation", sm, "/SM_ROOT")
    sis.start()

    rospy.loginfo("Starting pipeline navigation...")

    try:
        # Execute the state machine
        outcome = sm.execute()
        rospy.loginfo(f"Pipeline navigation completed with outcome: {outcome}")

    except rospy.ROSInterruptException:
        rospy.loginfo("Pipeline navigation interrupted")
    finally:
        # Stop the introspection server
        sis.stop()


if __name__ == "__main__":
    main()
