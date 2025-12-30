#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, PoseWithCovarianceStamped
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse
from robot_localization.srv import SetPose, SetPoseRequest
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from auv_common_lib.control.enable_state import ControlEnableHandler
import dynamic_reconfigure.client
from threading import Lock
import tf2_ros


class ReferencePosePublisherNode:
    def __init__(self):
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
        self.set_pose_client = rospy.ServiceProxy("set_pose", SetPose)
        self.set_heading_control_service = rospy.Service(
            "set_heading_control", SetBool, self.set_heading_control_handler
        )

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("cmd_pose", Pose, queue_size=10)

        self.control_enable_handler = ControlEnableHandler(1.0)

        # Initialize internal state
        self.target_depth = -0.4
        self.target_heading = 0.0
        self.last_cmd_time = rospy.Time.now()
        self.target_frame_id = ""
        self.is_resetting = False
        self.state_lock = Lock()  # To protect shared state
        self.is_heading_control_enabled = True

        self.set_pose_req = SetPoseRequest()
        self.set_pose_req.pose = PoseWithCovarianceStamped()
        self.set_pose_req.pose.header.stamp = rospy.Time.now()
        self.set_pose_req.pose.header.frame_id = "odom"

        # Parameters
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.command_timeout = rospy.get_param("~command_timeout", 0.1)

        # Dynamic reconfigure client
        self.dyn_client = dynamic_reconfigure.client.Client(
            "/taluy/auv_control_node", timeout=30
        )

        # TF2 buffer & listener and frame params
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "taluy/base_link")
        self.tf_lookup_timeout = rospy.get_param("~tf_lookup_timeout", 2.0)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Initialize stored PID parameters
        self.stored_kp_5 = 0.0
        self.stored_ki_5 = 0.0
        self.stored_kd_5 = 0.0
        self.heading_gains_stored = False

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        external_frame = req.external_frame
        internal_frame = req.internal_frame

        if not external_frame:
            external_frame = self.odom_frame

        if not internal_frame:
            internal_frame = self.base_frame

        robot_namespace = (
            self.base_frame.rsplit("/", 1)[0] + "/" if "/" in self.base_frame else ""
        )

        if robot_namespace and not internal_frame.startswith(robot_namespace):
            msg = f"Internal frame '{internal_frame}' does not start with robot namespace '{robot_namespace}'. Operation cancelled."
            rospy.logerr(msg)
            return SetDepthResponse(success=False, message=msg)

        rospy.loginfo(
            f"Alignment Request: Internal='{internal_frame}' -> External='{external_frame}' (Offset: {req.target_depth})"
        )

        try:
            t_odom_ext = self.tf_buffer.lookup_transform(
                self.odom_frame,
                external_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_lookup_timeout),
            )

            lookup_time = t_odom_ext.header.stamp
            t_base_int = self.tf_buffer.lookup_transform(
                self.base_frame,
                internal_frame,
                lookup_time,
                rospy.Duration(self.tf_lookup_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"TF lookup failed: {e}")
            return SetDepthResponse(success=False, message=f"TF lookup failed: {e}")

        target_z_world = t_odom_ext.transform.translation.z
        tool_offset_z = t_base_int.transform.translation.z

        base_z_desired = (target_z_world + req.target_depth) - tool_offset_z

        self.target_depth = base_z_desired
        self.target_frame_id = self.odom_frame

        success_msg = f"Aligned '{internal_frame}' to '{external_frame}'. Base set to Z={base_z_desired:.3f}"
        rospy.loginfo(success_msg)

        return SetDepthResponse(success=True, message=success_msg)

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
        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        _, _, current_heading = euler_from_quaternion(quaternion)

        if not self.is_heading_control_enabled:
            self.target_heading = current_heading
        elif not self.control_enable_handler.is_enabled() and not self.is_resetting:
            self.target_heading = current_heading

    def set_heading_control_handler(self, req):
        with self.state_lock:
            self.is_heading_control_enabled = req.data
            if not self.heading_gains_stored:
                try:
                    current_config = self.dyn_client.get_configuration(timeout=5)
                    self.stored_kp_5 = current_config["kp_5"]
                    self.stored_ki_5 = current_config["ki_5"]
                    self.stored_kd_5 = current_config["kd_5"]
                    self.heading_gains_stored = True
                except (
                    rospy.ServiceException,
                    rospy.ROSException,
                    dynamic_reconfigure.client.DynamicReconfigureTimeout,
                ) as e:
                    rospy.logerr(f"Failed to get current controller config: {e}")
                    return SetBoolResponse(
                        success=False,
                        message="Failed to get current controller config.",
                    )

            if self.is_heading_control_enabled:
                params = {
                    "kp_5": self.stored_kp_5,
                    "ki_5": self.stored_ki_5,
                    "kd_5": self.stored_kd_5,
                }
                self.dyn_client.update_configuration(params)
                return SetBoolResponse(success=True, message="Heading control enabled.")
            else:
                params = {"kp_5": 0.0, "ki_5": 0.0, "kd_5": 0.0}
                self.dyn_client.update_configuration(params)
                return SetBoolResponse(
                    success=True, message="Heading control disabled."
                )

    def cmd_vel_callback(self, msg):
        if (
            (not self.control_enable_handler.is_enabled())
            or self.is_resetting
            or not self.is_heading_control_enabled
        ):
            return

        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
        dt = min(dt, self.command_timeout)
        self.target_depth += msg.linear.z * dt
        self.target_heading += msg.angular.z * dt
        self.last_cmd_time = rospy.Time.now()

    def control_loop(self):
        cmd_pose = Pose()

        cmd_pose.position.z = self.target_depth
        # Frame id is odom
        quaternion = quaternion_from_euler(0.0, 0.0, self.target_heading)
        rospy.logdebug(f"heading: {self.target_heading}")
        cmd_pose.orientation.x = quaternion[0]
        cmd_pose.orientation.y = quaternion[1]
        cmd_pose.orientation.z = quaternion[2]
        cmd_pose.orientation.w = quaternion[3]

        self.cmd_pose_pub.publish(cmd_pose)

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
