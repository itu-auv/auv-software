#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import Twist
from auv_msgs.srv import AlignFrameController, AlignFrameControllerResponse
import auv_common_lib.control.enable_state as enable_state
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import angles


class FrameAligner:
    def __init__(self):
        rospy.init_node("frame_aligner_node", anonymous=True)
        self.listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.active = False
        self.source_frame = ""
        self.target_frame = ""
        self.angle_offset = 0.0

        # Initialize the enable signal handler with a timeout duration
        self.rate = rospy.get_param("~rate", 0.1)

        enable_timeout = 2.0 / self.rate  # 2 control dt period

        self.enable_handler = enable_state.ControlEnableHandler(enable_timeout)

        # Service for setting frames and starting alignment
        rospy.Service(
            "frame_alignment_controller",
            AlignFrameController,
            self.handle_align_request,
        )

        # Service for canceling control
        rospy.Service("cancel_control", Trigger, self.handle_cancel_request)

    def handle_align_request(self, req):
        if not self.enable_handler.is_enabled():
            message = "Control enable signal not active. Cannot start alignment."
            rospy.logerr(message)
            return AlignFrameControllerResponse(success=False, message=message)

        trans, rot = self.get_transform(
            self.source_frame, self.target_frame, self.angle_offset
        )

        if trans is None or rot is None:
            rospy.logerr("Failed to get transform. Cannot start alignment.")
            return AlignFrameControllerResponse(
                success=False, message="Failed to get transform"
            )

        self.source_frame = req.source_frame
        self.target_frame = req.target_frame
        self.angle_offset = req.angle_offset
        self.active = True
        rospy.loginfo(
            f"Aligning {self.source_frame} to {self.target_frame} with angle offset {self.angle_offset}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req):
        self.active = False
        rospy.loginfo("Control canceled")
        return TriggerResponse(success=True, message="Control deactivated")

    def get_transform(
        self, source_frame, target_frame, angle_offset, time=rospy.Time(0)
    ):
        try:
            # Get the current transform
            trans, rot = self.listener.lookupTransform(target_frame, source_frame, time)
            return trans, rot

            # Apply the angle offset to the rotation
            roll, pitch, yaw = euler_from_quaternion(rot)
            yaw -= angle_offset  # Adjust yaw with angle offset
            yaw = angles.normalize_angle(yaw)
            new_rot = quaternion_from_euler(roll, pitch, yaw)

            return trans, new_rot
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logerr(
                f"Cannot lookup transform from {source_frame} to {target_frame}"
            )
            return None, None

    def constrain(self, value, max_value):
        if value > max_value:
            return max_value
        if value < -max_value:
            return -max_value
        return value

    def compute_cmd_vel(self, trans, rot):
        kp = 0.55
        angle_kp = 0.45
        max_linear_velocity = 0.3
        max_angular_velocity = 0.4

        twist = Twist()
        # Set linear velocities based on translation differences
        twist.linear.x = self.constrain(trans[0] * kp, max_linear_velocity)
        twist.linear.y = self.constrain(trans[1] * kp, max_linear_velocity)

        # Convert quaternion to Euler angles and set angular velocity
        _, _, yaw = euler_from_quaternion(rot)
        twist.angular.z = self.constrain(yaw * angle_kp, max_angular_velocity)

        return twist

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if not self.active or not self.enable_handler.is_enabled():
                self.active = False
                rate.sleep()
                continue

            trans, rot = self.get_transform(
                self.target_frame, self.source_frame, self.angle_offset
            )
            if trans is None or rot is None:
                continue

            twist = self.compute_cmd_vel(trans, rot)
            self.cmd_vel_pub.publish(twist)
            rate.sleep()


if __name__ == "__main__":
    try:
        FrameAligner().run()
    except rospy.ROSInterruptException:
        pass