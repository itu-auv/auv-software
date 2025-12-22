#!/usr/bin/env python3

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    PoseStamped,
    Twist,
    PoseWithCovarianceStamped,
    PointStamped,
)
from std_srvs.srv import Trigger, TriggerResponse
from auv_msgs.srv import (
    SetDepth,
    SetDepthRequest,
    SetDepthResponse,
    AlignFrameController,
    AlignFrameControllerResponse,
)
from robot_localization.srv import SetPose, SetPoseRequest
from tf.transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_multiply,
)
from auv_common_lib.control.enable_state import ControlEnableHandler
from threading import Lock
from tf2_geometry_msgs import do_transform_point


class ReferencePosePublisherNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize subscribers
        self.odometry_sub = rospy.Subscriber(
            "odometry", Odometry, self.odometry_callback, tcp_nodelay=True
        )
        self.set_depth_service = rospy.Service(
            "set_depth", SetDepth, self.target_depth_handler
        )
        self.cmd_vel_sub = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, tcp_nodelay=True
        )
        self.reset_odometry_service = rospy.Service(
            "reset_odometry", Trigger, self.reset_odometry_handler
        )
        self.align_frame_service = rospy.Service(
            "align_frame/start", AlignFrameController, self.handle_align_request
        )
        self.cancel_control_service = rospy.Service(
            "align_frame/cancel", Trigger, self.handle_cancel_request
        )

        self.set_pose_client = rospy.ServiceProxy("set_pose", SetPose)

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("cmd_pose", PoseStamped, queue_size=10)

        self.control_enable_handler = ControlEnableHandler(1.0)

        # 6-DOF target pose (for tracking mode)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_depth = -0.4  # z
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_heading = 0.0  # yaw

        # Align frame state
        self.align_frame_active = False
        self.use_align_frame_depth = False

        self.last_cmd_time = rospy.Time.now()
        self.target_frame_id = "odom"
        self.state_lock = Lock()
        self.latest_odometry = Odometry()

        # reset odometry service
        self.set_pose_req = SetPoseRequest()
        self.set_pose_req.pose = PoseWithCovarianceStamped()
        self.set_pose_req.pose.header.stamp = rospy.Time.now()
        self.set_pose_req.pose.header.frame_id = "odom"
        self.is_resetting = False

        # Parameters
        self.namespace = rospy.get_param("~namespace", "taluy")
        self.base_frame = self.namespace + "/base_link"
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.command_timeout = rospy.get_param("~command_timeout", 0.1)

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        with self.state_lock:
            if self.align_frame_active and self.use_align_frame_depth:
                return SetDepthResponse(
                    success=False,
                    message="Cannot set depth while align_frame is active and use_depth is true",
                )

            self.target_depth = self.get_transformed_depth(
                self.target_frame_id,
                req.frame_id,
                req.target_depth,
            )
            return SetDepthResponse(
                success=True,
                message=f"Target depth set to {self.target_depth} in frame {self.target_frame_id}",
            )

    def handle_align_request(
        self, req: AlignFrameController
    ) -> AlignFrameControllerResponse:
        with self.state_lock:
            if not self.control_enable_handler.is_enabled():
                return AlignFrameControllerResponse(
                    success=False,
                    message="Cannot align to frame while control is disabled",
                )

            self.use_align_frame_depth = req.use_depth

            t = self.tf_buffer.lookup_transform(
                self.base_frame, req.source_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            self.target_x = t.transform.translation.x
            self.target_y = t.transform.translation.y

            if req.use_depth:
                self.target_depth = t.transform.translation.z
            else:
                self.target_depth = self.get_transformed_depth(
                    req.target_frame,
                    self.target_frame_id,
                    self.target_depth,
                )

            if not req.keep_orientation:
                quaternion = [
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ]

                if req.angle_offset != 0.0:
                    q_rot = quaternion_from_euler(0, 0, req.angle_offset)
                    quaternion = quaternion_multiply(quaternion, q_rot)

                self.target_roll, self.target_pitch, self.target_heading = (
                    euler_from_quaternion(quaternion)
                )

            self.target_frame_id = req.target_frame
            self.align_frame_active = True

        rospy.loginfo(
            f"Aligning {req.source_frame} to {req.target_frame} with angle offset {req.angle_offset}"
        )
        return AlignFrameControllerResponse(success=True, message="Alignment started")

    def handle_cancel_request(self, req) -> TriggerResponse:
        with self.state_lock:
            if not self.align_frame_active:
                return TriggerResponse(
                    success=False, message="Alignment is not active."
                )

            self.align_frame_active = False
            self.use_align_frame_depth = False
            self.set_target_to_odometry()

        rospy.loginfo("Align frame control canceled")
        return TriggerResponse(success=True, message="Alignment deactivated")

    def reset_odometry_handler(self, req):
        if self.is_resetting:
            return TriggerResponse(
                success=False, message="Odometry reset already in progress."
            )

        self.is_resetting = True
        rospy.logdebug("Starting odometry reset.")

        try:
            self.set_pose_client(self.set_pose_req)
            rospy.logdebug("Called set_pose service.")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"Service call failed: {e}")
            self.is_resetting = False
            return TriggerResponse(success=False, message=f"Service call failed: {e}")

        rospy.logdebug("Waiting for heading to settle.")
        rospy.sleep(2.0)

        self.is_resetting = False
        rospy.logdebug("Odometry reset finished.")
        return TriggerResponse(success=True, message="Odometry reset successfully.")

    def odometry_callback(self, msg):
        if self.control_enable_handler.is_enabled() and not self.is_resetting:
            return

        self.latest_odometry = msg

        self.set_target_to_odometry()

    def set_target_to_odometry(self):
        self.target_x = self.latest_odometry.pose.pose.position.x
        self.target_y = self.latest_odometry.pose.pose.position.y

        if self.target_frame_id != "odom":
            self.target_depth = self.get_transformed_depth(
                "odom",
                self.target_frame_id,
                self.target_depth,
            )
            self.target_frame_id = "odom"

        quaternion = [
            self.latest_odometry.pose.pose.orientation.x,
            self.latest_odometry.pose.pose.orientation.y,
            self.latest_odometry.pose.pose.orientation.z,
            self.latest_odometry.pose.pose.orientation.w,
        ]
        _, _, self.target_heading = euler_from_quaternion(quaternion)

        self.target_roll = 0.0
        self.target_pitch = 0.0

    def get_transformed_depth(
        self, target_frame: str, source_frame: str, target_depth: float
    ) -> float:
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        p = PointStamped()
        p.header.stamp = transform.header.stamp
        p.header.frame_id = source_frame
        p.point.x = 0.0
        p.point.y = 0.0
        p.point.z = target_depth

        p_in_target = do_transform_point(p, transform)
        return p_in_target.point.z

    def cmd_vel_callback(self, msg: Twist):
        if (
            not self.control_enable_handler.is_enabled()
            or self.is_resetting
            or self.align_frame_active
        ):
            return

        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
        dt = min(dt, self.command_timeout)

        self.target_x += msg.linear.x * dt
        self.target_y += msg.linear.y * dt
        self.target_depth += msg.linear.z * dt
        self.target_roll += msg.angular.x * dt
        self.target_pitch += msg.angular.y * dt
        self.target_heading += msg.angular.z * dt

        self.last_cmd_time = rospy.Time.now()

    def control_loop(self):
        cmd_pose_stamped = PoseStamped()
        cmd_pose_stamped.header.stamp = rospy.Time.now()
        cmd_pose_stamped.header.frame_id = self.target_frame_id

        cmd_pose_stamped.pose.position.x = self.target_x
        cmd_pose_stamped.pose.position.y = self.target_y
        cmd_pose_stamped.pose.position.z = self.target_depth

        quaternion = quaternion_from_euler(
            self.target_roll, self.target_pitch, self.target_heading
        )
        cmd_pose_stamped.pose.orientation.x = quaternion[0]
        cmd_pose_stamped.pose.orientation.y = quaternion[1]
        cmd_pose_stamped.pose.orientation.z = quaternion[2]
        cmd_pose_stamped.pose.orientation.w = quaternion[3]

        self.cmd_pose_pub.publish(cmd_pose_stamped)

    def run(self):
        rate = rospy.Rate(self.update_rate)

        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("reference_pose_publisher_node")
        reference_pose_publisher_node = ReferencePosePublisherNode()
        reference_pose_publisher_node.run()
    except rospy.ROSInterruptException:
        pass
