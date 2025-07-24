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
from dynamic_reconfigure.server import Server
from auv_control.cfg import AlignFrameConfig


class AlignFrameControllerNode:
    def __init__(self) -> None:
        rospy.init_node("align_frame_controller")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

        self.rate = rospy.get_param("~rate", 20)
        self.linear_kp = rospy.get_param("~linear_kp", 1.0)
        self.linear_kd = rospy.get_param("~linear_kd", 0.7)
        self.angular_kp = rospy.get_param("~angular_kp", 0.8)
        self.angular_kd = rospy.get_param("~angular_kd", 0.3)
        self.max_linear_velocity = rospy.get_param("~max_linear_velocity", 0.6)
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity", 0.4)

        self.srv = Server(AlignFrameConfig, self.reconfigure_callback)

        self.active = False
        self.source_frame = ""
        self.target_frame = ""
        self.angle_offset = 0.0
        self.keep_orientation = False
        self.req_max_linear_velocity = None
        self.req_max_angular_velocity = None

        self.prev_trans_error = None
        self.prev_yaw_error = None
        self.last_step_time = None

        self.killswitch_sub = rospy.Subscriber(
            "propulsion_board/status", Bool, self.killswitch_callback
        )

        rospy.Service(
            "align_frame/start", AlignFrameController, self.handle_align_request
        )
        rospy.Service("cancel_control", Trigger, self.handle_cancel_request)

    def reconfigure_callback(self, config, level):
        """Handles dynamic reconfigure updates for controller gains."""
        self.linear_kp = config.linear_kp
        self.linear_kd = config.linear_kd
        self.angular_kp = config.angular_kp
        self.angular_kd = config.angular_kd
        rospy.loginfo(
            f"Updated gains: Linear Kp={self.linear_kp}, Linear Kd={self.linear_kd}, "
            f"Angular Kp={self.angular_kp}, Angular Kd={self.angular_kd}"
        )
        return config

    def killswitch_callback(self, msg: Bool) -> None:
        if not msg.data:
            self.active = False

    def handle_align_request(
        self, req: AlignFrameController
    ) -> AlignFrameControllerResponse:
        # Check if the transform exists before starting
        if not self.tf_buffer.can_transform(
            req.source_frame, req.target_frame, rospy.Time.now(), rospy.Duration(1.0)
        ):
            rospy.logwarn(
                f"Align request aborted: Transform from '{req.target_frame}' to '{req.source_frame}' not available."
            )
            return AlignFrameControllerResponse(
                success=False,
                message=f"Transform from '{req.target_frame}' to '{req.source_frame}' not available.",
            )

        self.source_frame = req.source_frame
        self.target_frame = req.target_frame
        self.angle_offset = req.angle_offset
        self.keep_orientation = req.keep_orientation
        # Limit max velocities to default if request exceeds default or is not set
        if hasattr(req, "max_linear_velocity") and req.max_linear_velocity > 0:
            self.req_max_linear_velocity = min(
                req.max_linear_velocity, self.max_linear_velocity
            )
        else:
            self.req_max_linear_velocity = self.max_linear_velocity
        if hasattr(req, "max_angular_velocity") and req.max_angular_velocity > 0:
            self.req_max_angular_velocity = min(
                req.max_angular_velocity, self.max_angular_velocity
            )
        else:
            self.req_max_angular_velocity = self.max_angular_velocity
        self.active = True
        self.prev_trans_error = None
        self.prev_yaw_error = None
        self.last_step_time = None
        rospy.loginfo(
            f"Aligning {self.source_frame} to {self.target_frame} with angle offset {self.angle_offset}, max_linear_velocity {self.req_max_linear_velocity}, max_angular_velocity {self.req_max_angular_velocity}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req) -> TriggerResponse:
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1.0)  # Allow time for the command to be processed
        self.active = False
        rospy.loginfo("Control canceled")
        # Publish a zero velocity command to clear old velocity commands
        return TriggerResponse(success=True, message="Control deactivated")

    def get_error(
        self,
        source_frame: str,
        target_frame: str,
        angle_offset: float,
        time: rospy.Time = rospy.Time.now(),
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
        self,
        trans_error: Tuple[float, float, float],
        yaw_error: float,
        dt: float,
    ) -> Twist:
        twist = Twist()

        # Proportional terms
        linear_p = [val * self.linear_kp for val in trans_error]
        angular_p = yaw_error * self.angular_kp

        # Derivative terms
        if (
            self.prev_trans_error is not None
            and self.prev_yaw_error is not None
            and dt > 0
        ):
            trans_error_deriv = [
                (trans_error[i] - self.prev_trans_error[i]) / dt for i in range(3)
            ]
            yaw_error_deriv = (yaw_error - self.prev_yaw_error) / dt
        else:
            trans_error_deriv = [0, 0, 0]
            yaw_error_deriv = 0

        linear_d = [val * self.linear_kd for val in trans_error_deriv]
        angular_d = yaw_error_deriv * self.angular_kd

        # Sum P and D terms
        max_lin = (
            self.req_max_linear_velocity
            if self.req_max_linear_velocity is not None
            else self.max_linear_velocity
        )
        max_ang = (
            self.req_max_angular_velocity
            if self.req_max_angular_velocity is not None
            else self.max_angular_velocity
        )
        twist.linear.x = self.constrain(linear_p[0] + linear_d[0], max_lin)
        twist.linear.y = self.constrain(linear_p[1] + linear_d[1], max_lin)
        # twist.linear.z = self.constrain(linear_p[2] + linear_d[2], max_lin)
        twist.angular.z = (
            0.0
            if self.keep_orientation
            else self.constrain(angular_p + angular_d, max_ang)
        )

        return twist

    def step(self) -> None:
        current_time = rospy.Time.now()
        if self.last_step_time is None:
            self.last_step_time = current_time
            return

        dt = (current_time - self.last_step_time).to_sec()

        trans_error, yaw_error = self.get_error(
            self.source_frame, self.target_frame, self.angle_offset
        )
        if trans_error is None or yaw_error is None:
            return

        self.enable_pub.publish(Bool(data=True))
        cmd_vel = self.compute_cmd_vel(trans_error, yaw_error, dt)
        self.cmd_vel_pub.publish(cmd_vel)

        self.prev_trans_error = trans_error
        self.prev_yaw_error = yaw_error
        self.last_step_time = current_time

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
