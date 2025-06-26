#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.msg import VisualFeature
from auv_msgs.srv import VisualServoing, VisualServoingResponse


class VisualServoingController:
    def __init__(self):
        rospy.init_node("visual_servoing_controller", anonymous=True)
        rospy.loginfo("Visual Servoing Controller node started")

        # Load parameters
        self.u_desired = rospy.get_param("~u_desired", 320.0)
        self.v_desired = rospy.get_param("~v_desired", 240.0)
        self.lambda_gain = rospy.get_param("~lambda", 0.1)
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/taluy/cameras/cam_front/camera_info"
        )

        # Camera intrinsics (will be updated by the callback)
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        # State
        self.active = False
        self.target_prop = ""
        self.service_start_time = None

        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel_ibvs", Twist, queue_size=10)
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

    def handle_start_request(self, req: VisualServoing) -> VisualServoingResponse:
        if self.active:
            return VisualServoingResponse(
                success=False, message="IVS Controller is already active."
            )

        self.target_prop = req.target_prop
        self.active = True
        self.service_start_time = rospy.Time.now()
        rospy.loginfo(f"Visual servoing started for target: {self.target_prop}")
        self.control_enable_pub.publish(Bool(data=True))
        return VisualServoingResponse(
            success=True, message="Visual servoing activated."
        )

    def handle_cancel_request(self, req: Trigger) -> TriggerResponse:
        if not self.active:
            return TriggerResponse(success=False, message="Controller is not active.")

        self.active = False
        self.cmd_vel_pub.publish(Twist())  # Stop the vehicle
        self.control_enable_pub.publish(Bool(data=False))
        rospy.loginfo("Visual servoing cancelled by request.")
        return TriggerResponse(success=True, message="Visual servoing deactivated.")

    def camera_info_callback(self, msg: CameraInfo):
        # Store camera intrinsic parameters from the CameraInfo message
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        rospy.loginfo_once(
            f"Camera intrinsics loaded: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

    def feature_callback(self, msg: VisualFeature):
        if not self.active or msg.object_name != self.target_prop:
            return

        if self.fx == 0.0:
            rospy.logwarn_throttle(
                5, "Waiting for camera info... Visual servoing is paused."
            )
            return

        # Current feature coordinates
        u = msg.u
        v = msg.v
        Z = msg.Z

        if Z <= 0:
            rospy.logwarn_throttle(
                1, f"Invalid depth Z={Z}. Skipping control calculation."
            )
            return

        # Error vector in the image plane
        error = np.array([[u - self.u_desired], [v - self.v_desired]])

        # Normalized coordinates
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy

        # Interaction matrix L maps camera velocity to feature velocity (vel_feature = L * v_cam).
        # Equals to Jacobian of the image feature with respect to the camera's 6-dof velocity.
        # Each row corresponds to a feature (u, v), and each column to a velocity component.

        # Row for u_dot
        L_u = [
            -1 / Z,  # Effect of linear velocity Vx
            0,  # Effect of linear velocity Vy
            x_norm / Z,  # Effect of linear velocity Vz
            x_norm * y_norm,  # Effect of angular velocity Wx
            -(1 + x_norm**2),  # Effect of angular velocity Wy
            y_norm,  # Effect of angular velocity Wz
        ]

        # Row for v_dot (how v changes with camera velocity)
        L_v = [
            0,  # Effect of linear velocity Vx
            -1 / Z,  # Effect of linear velocity Vy
            y_norm / Z,  # Effect of linear velocity Vz
            1 + y_norm**2,  # Effect of angular velocity Wx
            -x_norm * y_norm,  # Effect of angular velocity Wy
            -x_norm,  # Effect of angular velocity Wz
        ]

        L = np.array([L_u, L_v])

        # Moore Penrose pseudo-inverse of L
        try:
            L_inv = np.linalg.pinv(L)
        except np.linalg.LinAlgError:
            rospy.logerr("Could not compute pseudo-inverse of the interaction matrix.")
            return

        # control law: v_cam = -lambda * L_inv * e
        camera_velocity = -self.lambda_gain * np.dot(L_inv, error)

        # Create and publish the Twist message
        twist_msg = Twist()
        twist_msg.linear.x = camera_velocity[0, 0]
        twist_msg.linear.y = camera_velocity[1, 0]
        twist_msg.linear.z = camera_velocity[2, 0]
        twist_msg.angular.x = camera_velocity[3, 0]
        twist_msg.angular.y = camera_velocity[4, 0]
        twist_msg.angular.z = camera_velocity[5, 0]

        self.cmd_vel_pub.publish(twist_msg)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.active and self.service_start_time:
                if (rospy.Time.now() - self.service_start_time).to_sec() > 10.0:
                    rospy.loginfo("Visual servoing timed out after 10 seconds.")
                    self.active = False
                    self.cmd_vel_pub.publish(Twist())  # Stop the vehicle
                    self.control_enable_pub.publish(Bool(data=False))

            rate.sleep()


if __name__ == "__main__":
    try:
        controller = VisualServoingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
