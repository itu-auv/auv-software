#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.msg import VisualFeature
from auv_msgs.srv import VisualServoing, VisualServoingResponse
from dynamic_reconfigure.server import Server
from auv_control.cfg import VisualServoingConfig


class VisualServoingController:
    def __init__(self):
        rospy.init_node("visual_servoing_controller", anonymous=True)
        rospy.loginfo("Visual Servoing Controller node started")

        # Load parameters
        self.u_desired = rospy.get_param("~u_desired", 320.0)
        self.v_desired = rospy.get_param("~v_desired", 240.0)
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/taluy/cameras/cam_front/camera_info"
        )
        self.kp_gain = 1.0
        self.kd_gain = 0.2
        self.rate_hz = rospy.get_param("~rate_hz", 10.0)

        # Camera intrinsics (will be updated by the callback)
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        # State
        self.active = False
        self.target_prop = ""
        self.service_start_time = None
        self.last_feature = None
        self.last_error = None
        self.last_time = None

        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.control_enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "/visual_features", VisualFeature, self.feature_callback, queue_size=1
        )
        rospy.Subscriber(
            camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1
        )

        # Services
        rospy.Service(
            "visual_servoing/start", VisualServoing, self.handle_start_request
        )
        rospy.Service("visual_servoing/cancel", Trigger, self.handle_cancel_request)

        # Dynamic reconfigure
        self.srv = Server(VisualServoingConfig, self.reconfigure_callback)

    def reconfigure_callback(self, config, level):
        self.kp_gain = config.kp_gain
        self.kd_gain = config.kd_gain
        rospy.loginfo(f"Updated gains: Kp={self.kp_gain}, kd={self.kd_gain}")
        return config

    def handle_start_request(self, req: VisualServoing) -> VisualServoingResponse:
        if self.active:
            return VisualServoingResponse(
                success=False, message="IVS Controller is already active."
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

    def camera_info_callback(self, msg: CameraInfo):
        # Store camera intrinsic parameters from the CameraInfo message
        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        rospy.loginfo_once(
            f"Intrinsics fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

    def feature_callback(self, msg: VisualFeature):
        self.last_feature = msg

    def step(self):
        feature = self.last_feature
        if not feature or feature.object_name != self.target_prop or self.fx == 0.0:
            return

        rospy.loginfo_throttle(
            5,
            f"Feature callback received for {feature.object_name}. Target: {self.target_prop}. Active: {self.active}",
        )

        u, v, Z = feature.u, feature.v, feature.Z

        rospy.loginfo_throttle(5, f"Received feature: u={u}, v={v}, Z={Z}")

        if Z <= 0:
            rospy.logwarn_throttle(
                1, f"Invalid depth Z={Z}. Skipping control calculation."
            )
            return

        error = np.array([[u - self.u_desired], [v - self.v_desired]])

        current_time = rospy.Time.now()

        # PD Control
        error_dot = np.zeros_like(error)
        if self.last_error is not None and self.last_time is not None:
            dt = (current_time - self.last_time).to_sec()
            if dt > 0:
                error_dot = (error - self.last_error) / dt

        self.last_error = error
        self.last_time = current_time

        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy

        # Interaction matrix L maps camera velocity to feature velocity (vel_feature = L * v_cam).
        # Equals to Jacobian of the image feature with respect to the camera's 6-dof velocity.
        # Each row corresponds to a feature (u, v), and each column to a velocity component.

        # Row for u_dot
        L_u = [
            -self.fx / Z,
            0,
            x_norm * self.fx / Z,
            x_norm * y_norm * self.fx,
            -(1 + x_norm**2) * self.fx,
            y_norm * self.fx,
        ]

        # Row for v_dot
        L_v = [
            0,
            -self.fy / Z,
            y_norm * self.fy / Z,
            (1 + y_norm**2) * self.fy,
            -x_norm * y_norm * self.fy,
            -x_norm * self.fy,
        ]

        L = np.array([L_u, L_v])

        # PD control law
        pd_signal = self.kp_gain * error + self.kd_gain * error_dot

        try:
            v_cam = -np.linalg.pinv(L).dot(pd_signal)
        except np.linalg.LinAlgError:
            rospy.logerr("Cannot invert L.")
            return

        twist = Twist()
        # Linear Velocities
        # Robot X (forward) = Camera Z (forward)
        # twist.linear.x = v_cam[2, 0]
        # Robot Y (left) = -Camera X (right)
        twist.linear.y = round(-v_cam[0, 0], 2)
        # Robot Z (up) = -Camera Y (down)
        # twist.linear.z = -v_cam[1, 0]

        # Angular Velocities
        # Robot Roll (around X) = Camera Yaw (around Z)
        # twist.angular.x = v_cam[5, 0]
        # Robot Pitch (around Y) = -Camera Roll (around X)
        # twist.angular.y = -v_cam[3, 0]
        # Robot Yaw (around Z) = -Camera Pitch (around Y)
        twist.angular.z = round(-v_cam[4, 0], 2)
        self.cmd_vel_pub.publish(twist)

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
                    self.step()
            rate.sleep()


if __name__ == "__main__":
    try:
        controller = VisualServoingController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass
