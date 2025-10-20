#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import threading
import time
import math


class GripperController:
    def __init__(self):
        rospy.init_node("gripper_controller")

        # Get parameters
        self.namespace = rospy.get_param("~namespace", "taluy")

        # Joint names
        self.left_joint = f"{self.namespace}/finger_left_joint"
        self.right_joint = f"{self.namespace}/finger_right_joint"
        self.wrist_joint = f"{self.namespace}/wrist"

        # Publishers for joint commands
        self.left_pub = rospy.Publisher(
            f"/{self.namespace}/finger_left_joint_position_controller/command",
            Float64,
            queue_size=1,
        )
        self.right_pub = rospy.Publisher(
            f"/{self.namespace}/finger_right_joint_position_controller/command",
            Float64,
            queue_size=1,
        )
        self.wrist_pub = rospy.Publisher(
            f"/{self.namespace}/wrist_position_controller/command",
            Float64,
            queue_size=1,
        )

        # Position limits (VERY conservative for stability)
        self.open_position = (
            0.08  # Small opening angle (about 4.5 degrees) - within 0.1 limit
        )
        self.closed_position = 0.0
        self.wrist_rotation = math.pi / 8  # Very small wrist rotation (22.5 degrees)

        # Current joint states
        self.current_joints = {}
        self.joint_sub = rospy.Subscriber(
            f"/{self.namespace}/joint_states", JointState, self.joint_state_callback
        )

        # Services
        self.open_service = rospy.Service(
            f"/{self.namespace}/gripper/open", Trigger, self.open_gripper
        )
        self.close_service = rospy.Service(
            f"/{self.namespace}/gripper/close", Trigger, self.close_gripper
        )
        self.rotate_wrist_service = rospy.Service(
            f"/{self.namespace}/gripper/rotate_wrist", Trigger, self.rotate_wrist
        )

        rospy.loginfo(f"Gripper Controller initialized for namespace: {self.namespace}")
        rospy.loginfo("Available services:")
        rospy.loginfo(f"  - /{self.namespace}/gripper/open")
        rospy.loginfo(f"  - /{self.namespace}/gripper/close")
        rospy.loginfo(f"  - /{self.namespace}/gripper/rotate_wrist")

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if name in [self.left_joint, self.right_joint, self.wrist_joint]:
                self.current_joints[name] = msg.position[i]

    def smooth_move_to_position(self, target_left, target_right, duration=2.0):
        """Smoothly move gripper fingers to target positions over specified duration"""
        # Get current positions (default to 0 if not available)
        current_left = self.current_joints.get(self.left_joint, 0.0)
        current_right = self.current_joints.get(self.right_joint, 0.0)

        # Calculate steps for smooth motion
        steps = int(duration * 20)  # 20 Hz control rate
        dt = duration / steps

        for i in range(steps + 1):
            # Linear interpolation
            progress = float(i) / steps
            left_pos = current_left + progress * (target_left - current_left)
            right_pos = current_right + progress * (target_right - current_right)

            # Publish positions
            self.left_pub.publish(Float64(left_pos))
            self.right_pub.publish(Float64(right_pos))

            # Sleep for smooth motion
            rospy.sleep(dt)

    def open_gripper(self, req):
        """Open the gripper smoothly"""
        try:
            rospy.loginfo("Opening gripper smoothly...")

            # Use smooth motion to open gripper over 5 seconds (very slow for stability)
            thread = threading.Thread(
                target=self.smooth_move_to_position,
                args=(self.open_position, self.open_position, 5.0),
            )
            thread.daemon = True
            thread.start()

            return TriggerResponse(success=True, message="Gripper opening smoothly...")
        except Exception as e:
            rospy.logerr(f"Failed to open gripper: {e}")
            return TriggerResponse(
                success=False, message=f"Failed to open gripper: {e}"
            )

    def close_gripper(self, req):
        """Close the gripper smoothly"""
        try:
            rospy.loginfo("Closing gripper smoothly...")

            # Use smooth motion to close gripper over 3 seconds (slower for stability)
            thread = threading.Thread(
                target=self.smooth_move_to_position,
                args=(self.closed_position, self.closed_position, 3.0),
            )
            thread.daemon = True
            thread.start()

            return TriggerResponse(success=True, message="Gripper closing smoothly...")
        except Exception as e:
            rospy.logerr(f"Failed to close gripper: {e}")
            return TriggerResponse(
                success=False, message=f"Failed to close gripper: {e}"
            )

    def rotate_wrist(self, req):
        """Rotate the wrist 90 degrees"""
        try:
            rospy.loginfo("Rotating wrist...")

            # Toggle wrist rotation between 0 and 90 degrees
            if abs(self.wrist_rotation) < 0.1:
                self.wrist_rotation = 1.57  # 90 degrees
            else:
                self.wrist_rotation = 0.0  # 0 degrees

            self.wrist_pub.publish(Float64(self.wrist_rotation))

            return TriggerResponse(
                success=True,
                message=f"Wrist rotated to {self.wrist_rotation:.2f} radians",
            )
        except Exception as e:
            rospy.logerr(f"Failed to rotate wrist: {e}")
            return TriggerResponse(
                success=False, message=f"Failed to rotate wrist: {e}"
            )

    def run(self):
        """Main loop"""
        rospy.loginfo("Gripper Controller is running...")
        rospy.spin()


if __name__ == "__main__":
    try:
        controller = GripperController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
