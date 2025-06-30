#!/usr/bin/env python3

import rospy
import tf.transformations
import math

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.msg import PropsYaw
from auv_msgs.srv import VisualServoing, VisualServoingResponse
from dynamic_reconfigure.server import Server
from auv_control.cfg import VisualServoingConfig
from sensor_msgs.msg import Imu


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


#! angle info / imu info should be in real time, same timestamp, etc.


class VisualServoingController:
    def __init__(self):
        rospy.init_node("visual_servoing_controller", anonymous=True)
        rospy.loginfo("Visual Servoing Controller node started")

        self.kp_gain = rospy.get_param("~kp_gain", 1.0)
        self.kd_gain = rospy.get_param("~kd_gain", 0.2)
        self.rate_hz = rospy.get_param("~rate_hz", 10.0)

        # State
        self.active = False
        self.target_prop = ""
        self.service_start_time = None
        self.last_error = None
        self.last_time = None
        # State from sensors - updated in callbacks
        self.current_yaw = 0.0  # Robot's yaw in the world frame (from IMU)
        self.last_prop_yaw_in_robot = 0.0
        self.angular_velocity_z = 0.0  # Robot's angular velocity (from IMU)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.control_enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "/visual_servoing/yaw_error", PropsYaw, self.prop_yaw_callback, queue_size=1
        )
        rospy.Subscriber("/imu/data", Imu, self.imu_callback, queue_size=1)
        # Services
        rospy.Service(
            "visual_servoing/start", VisualServoing, self.handle_start_request
        )
        rospy.Service("visual_servoing/cancel", Trigger, self.handle_cancel_request)

        # Dynamic reconfigure
        self.srv = Server(VisualServoingConfig, self.reconfigure_callback)

    def imu_callback(self, msg: Imu):
        orientation_q = msg.orientation
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        )
        self.angular_velocity_z = msg.angular_velocity.z

    def prop_yaw_callback(self, msg: PropsYaw):
        if not self.active or msg.object != self.target_prop:
            return

    def control_step(self):
        """
        Executes one iteration of the PD control loop
        """
        # desired absolute heading
        target_yaw_in_world = self.current_yaw + self.last_prop_yaw_in_robot

        # error is the shortest angular distance between where we want to be and where we are.
        error = normalize_angle(target_yaw_in_world - self.current_yaw)
        p_signal = self.kp_gain * error
        d_signal = self.kd_gain * self.angular_velocity_z

        pd_signal = p_signal + d_signal

        twist = Twist()
        twist.angular.z = pd_signal
        self.cmd_vel_pub.publish(twist)

    def reconfigure_callback(self, config, level):
        if "kp_gain" in config:
            self.kp_gain = config.kp_gain
        if "kd_gain" in config:
            self.kd_gain = config.kd_gain
        rospy.loginfo(f"Updated gains: Kp={self.kp_gain}, kd={self.kd_gain}")
        return config

    def handle_start_request(self, req: VisualServoing) -> VisualServoingResponse:
        if self.active:
            return VisualServoingResponse(
                success=False, message="VS Controller is already active."
            )

        self.target_prop = req.target_prop
        self.active = True
        self.service_start_time = rospy.Time.now()
        self.last_error = None
        self.last_time = None
        rospy.loginfo(f"Visual servoing started for target: {self.target_prop}")
        return VisualServoingResponse(
            success=True, message="Visual servoing activated."
        )

    def handle_cancel_request(self, req: Trigger) -> TriggerResponse:
        if not self.active:
            return TriggerResponse(success=False, message="Controller is not active.")

        self.active = False
        self.cmd_vel_pub.publish(Twist())  # Stop the vehicle
        rospy.sleep(1.0)
        self.control_enable_pub.publish(Bool(data=False))
        rospy.loginfo("Visual servoing cancelled by request.")
        return TriggerResponse(success=True, message="Visual servoing deactivated.")

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.active:
                # timeout after 10 000s
                if (rospy.Time.now() - self.service_start_time).to_sec() > 10000.0:
                    rospy.loginfo("Timed out")
                    self.active = False
                    self.cmd_vel_pub.publish(Twist())
                    self.control_enable_pub.publish(Bool(False))
                else:
                    self.control_enable_pub.publish(Bool(True))
                    self.control_step()
            rate.sleep()


if __name__ == "__main__":
    try:
        controller = VisualServoingController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass
