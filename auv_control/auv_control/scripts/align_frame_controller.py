#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import angles
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import AlignFrameController, AlignFrameControllerResponse
from tf.transformations import euler_from_quaternion
from typing import Tuple, Optional


class AlignFrameControllerNode:
    def __init__(self) -> None:
        rospy.init_node("align_frame_controller")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

        self.rate = rospy.get_param("~rate", 0.1)
        self.linear_kp = rospy.get_param("~linear_kp", 0.55)
        self.angular_kp = rospy.get_param("~angular_kp", 0.45)
        self.max_linear_velocity = rospy.get_param("~max_linear_velocity", 0.8)
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity", 0.9)

        self.active = False
        self.source_frame = ""
        self.target_frame = ""
        self.angle_offset = 0.0

        self.killswitch_sub = rospy.Subscriber(
            "propulsion_board/status", Bool, self.killswitch_callback
        )

        rospy.Service(
            "align_frame/start", AlignFrameController, self.handle_align_request
        )
        rospy.Service("cancel_control", Trigger, self.handle_cancel_request)

    def killswitch_callback(self, msg: Bool) -> None:
        if not msg.data:
            self.active = False

    def handle_align_request(
        self, req: AlignFrameController
    ) -> AlignFrameControllerResponse:
        self.source_frame = req.source_frame
        self.target_frame = req.target_frame
        self.angle_offset = req.angle_offset
        self.active = True
        rospy.loginfo(
            f"Aligning {self.source_frame} to {self.target_frame} with angle offset {self.angle_offset}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req) -> TriggerResponse:
        self.active = False
        rospy.loginfo("Control canceled")
        # Publish a zero velocity command to clear old velocity commands
        self.cmd_vel_pub.publish(Twist())
        return TriggerResponse(success=True, message="Control deactivated")

    def get_error(
        self,
        source_frame: str,
        target_frame: str,
        angle_offset: float,
        time: rospy.Time = rospy.Time(0),
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                source_frame, target_frame, time, rospy.Duration(2.0)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation
            trans_error = (trans.x, trans.y, trans.z)
            _, _, yaw_error = euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
            yaw_error = angles.normalize_angle(yaw_error + angle_offset)
            return trans_error, yaw_error
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            return None, None

    @staticmethod
    def constrain(value: float, limit: float) -> float:
        return max(min(value, limit), -limit)

    def compute_cmd_vel(
        self, trans_error: Tuple[float, float, float], yaw_error: float
    ) -> Twist:
        twist = Twist()
        twist.linear.x = self.constrain(
            trans_error[0] * self.linear_kp, self.max_linear_velocity
        )
        twist.linear.y = self.constrain(
            trans_error[1] * self.linear_kp, self.max_linear_velocity
        )
        # twist.linear.z = self.constrain(trans_error[2] * self.linear_kp, self.max_linear_velocity)
        twist.angular.z = self.constrain(
            yaw_error * self.angular_kp, self.max_angular_velocity
        )
        return twist

    def step(self) -> None:
        trans_error, yaw_error = self.get_error(
            self.source_frame, self.target_frame, self.angle_offset
        )
        if trans_error is None or yaw_error is None:
            return
        self.enable_pub.publish(Bool(data=True))
        cmd_vel = self.compute_cmd_vel(trans_error, yaw_error)
        self.cmd_vel_pub.publish(cmd_vel)

    def spin(self) -> None:
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.active:
                self.step()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = AlignFrameControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
