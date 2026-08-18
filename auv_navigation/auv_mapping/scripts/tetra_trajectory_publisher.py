#!/usr/bin/env python3

"""Publish a configurable frame beyond the front-camera tetra detection."""

import math

import rospy
import tf.transformations
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse

from auv_mapping.cfg import TetraTrajectoryConfig
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class TetraTrajectoryPublisher:
    def __init__(self):
        rospy.init_node("tetra_trajectory_publisher")

        self.enabled = bool(rospy.get_param("~enabled", False))
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.tetra_front_frame = rospy.get_param(
            "~tetra_front_frame", "tetra_front_link"
        )
        self.tetra_further_frame = rospy.get_param(
            "~tetra_further_frame", "tetra_further_link"
        )
        self.tetra_further_distance = float(
            rospy.get_param("~tetra_further_distance", 4.0)
        )
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 0.5))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.minimum_planar_distance = float(
            rospy.get_param("~minimum_planar_distance", 0.05)
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.reconfigure_server = Server(
            TetraTrajectoryConfig, self.reconfigure_callback
        )

        self.set_object_transform = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform.wait_for_service()
        self.enable_service = rospy.Service(
            "toggle_tetra_trajectory", SetBool, self.handle_enable
        )

        rospy.loginfo(
            "Tetra trajectory publisher ready: %s -> %s (%.2f m beyond)",
            self.tetra_front_frame,
            self.tetra_further_frame,
            self.tetra_further_distance,
        )

    def reconfigure_callback(self, config, _level):
        self.tetra_further_distance = config.tetra_further_distance
        return config

    def handle_enable(self, request):
        self.enabled = request.data
        message = "Tetra trajectory publishing {}".format(
            "enabled" if self.enabled else "disabled"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def lookup_transform(self, frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                frame,
                rospy.Time(0),
                rospy.Duration(self.lookup_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(
                5.0,
                "Cannot publish %s; %s -> %s lookup failed: %s",
                self.tetra_further_frame,
                self.odom_frame,
                frame,
                error,
            )
            return None

    def build_further_pose(self, robot_transform, tetra_transform):
        robot = robot_transform.transform.translation
        tetra = tetra_transform.transform.translation
        dx = tetra.x - robot.x
        dy = tetra.y - robot.y
        planar_distance = math.hypot(dx, dy)
        if planar_distance < self.minimum_planar_distance:
            rospy.logwarn_throttle(
                5.0,
                "Cannot build %s: %s and %s overlap in XY",
                self.tetra_further_frame,
                self.robot_frame,
                self.tetra_front_frame,
            )
            return None

        unit_x = dx / planar_distance
        unit_y = dy / planar_distance
        yaw = math.atan2(dy, dx)
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        return Pose(
            position=Point(
                tetra.x + unit_x * self.tetra_further_distance,
                tetra.y + unit_y * self.tetra_further_distance,
                tetra.z,
            ),
            orientation=Quaternion(*quaternion),
        )

    def publish_further_frame(self):
        robot_transform = self.lookup_transform(self.robot_frame)
        if robot_transform is None:
            return
        tetra_transform = self.lookup_transform(self.tetra_front_frame)
        if tetra_transform is None:
            return

        pose = self.build_further_pose(robot_transform, tetra_transform)
        if pose is None:
            return

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.tetra_further_frame
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation

        try:
            response = self.set_object_transform.call(
                SetObjectTransformRequest(transform=transform)
            )
        except rospy.ServiceException as error:
            rospy.logerr_throttle(
                5.0, "Failed to publish %s: %s", self.tetra_further_frame, error
            )
            return
        if not response.success:
            rospy.logwarn_throttle(
                5.0,
                "Failed to set %s: %s",
                self.tetra_further_frame,
                response.message,
            )

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if self.enabled:
                self.publish_further_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        TetraTrajectoryPublisher().run()
    except rospy.ROSInterruptException:
        pass
