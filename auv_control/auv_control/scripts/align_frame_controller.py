#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import Twist
from auv_msgs.srv import AlignFrameController, AlignFrameControllerResponse
import auv_common_lib.control.enable_state as enable_state
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class FrameAligner:
    def __init__(self):
        rospy.init_node("frame_aligner_node", anonymous=True)
        self.listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher("/taluy/cmd_vel", Twist, queue_size=10)
        self.active = False
        self.source_frame = ""
        self.target_frame = ""
        self.angle_offset = 0.0

        # Initialize the enable signal handler with a timeout duration
        enable_timeout = rospy.get_param("~enable_timeout", 5.0)  # seconds

        self.enable_handler = enable_state.ControlEnableHandler(enable_timeout)

        # Service for setting frames and starting alignment
        rospy.Service(
            "align_frame_controller", AlignFrameController, self.handle_align_request
        )
        # Service for canceling control
        rospy.Service("cancel_control", Trigger, self.handle_cancel_request)

    def handle_align_request(self, req):
        if not self.enable_handler.is_enabled():
            message = "Control enable signal not active. Cannot start alignment."
            rospy.logerr(message)
            return AlignFrameControllerResponse(success=False, message=message)

        self.source_frame = req.source_frame
        self.target_frame = req.target_frame
        self.angle_offset = req.angle_offset
        self.active = True
        rospy.loginfo(
            f"Aligning {self.source_frame} to {self.target_frame} with angle offset {self.angle_offset}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req):
        if not self.enable_handler.is_enabled():
            message = "Control enable signal not active. Cannot cancel alignment."
            rospy.logerr(message)
            return TriggerResponse(success=False, message=message)

        self.active = False
        rospy.loginfo("Control canceled")
        return TriggerResponse(success=True, message="Control deactivated")

    def get_transform(self):
        try:
            # Get the current transform
            trans, rot = self.listener.lookupTransform(
                self.target_frame, self.source_frame, rospy.Time(0)
            )

            # Apply the angle offset to the rotation
            roll, pitch, yaw = euler_from_quaternion(rot)
            yaw -= self.angle_offset  # Adjust yaw with angle offset
            new_rot = quaternion_from_euler(roll, pitch, yaw)

            return trans, new_rot
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            return None, None

    def compute_cmd_vel(self, trans, rot):
        twist = Twist()
        # Set linear velocities based on translation differences
        twist.linear.x = trans[0]
        twist.linear.y = trans[1]

        # Convert quaternion to Euler angles and set angular velocity
        _, _, yaw = euler_from_quaternion(rot)
        twist.angular.z = yaw

        return twist

    def run(self):
        rate = rospy.Rate(10)  # Hz
        while not rospy.is_shutdown():
            if not self.active or not self.enable_handler.is_enabled():
                rate.sleep()
                continue

            trans, rot = self.get_transform()
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
