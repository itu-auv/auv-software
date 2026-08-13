#!/usr/bin/env python3

import math

import rospy
import tf.transformations
import tf2_ros
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse


class BuoyTrajectoryPublisher:
    """Publish a robot-side approach frame that faces the configured buoy."""

    def __init__(self):
        self.is_enabled = bool(rospy.get_param("~enabled", False))
        self.parent_frame = rospy.get_param("~parent_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.buoy_frame = rospy.get_param("~buoy_frame", "buoy")
        self.close_approach_frame = rospy.get_param(
            "~close_approach_frame", "buoy_close_approach"
        )
        self.approach_distance_m = float(
            rospy.get_param("~approach_distance_m", 3.0)
        )
        self.target_z_offset_m = float(rospy.get_param("~target_z_offset_m", 0.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.lookup_timeout_seconds = float(
            rospy.get_param("~lookup_timeout_seconds", 0.5)
        )
        self.minimum_planar_distance_m = float(
            rospy.get_param("~minimum_planar_distance_m", 0.05)
        )

        if self.approach_distance_m < 0.0:
            raise ValueError("~approach_distance_m must be non-negative")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("~publish_rate_hz must be positive")

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        try:
            self.set_object_transform_service.wait_for_service(timeout=10.0)
        except rospy.ROSException as exc:
            raise RuntimeError(
                "set_object_transform service was not available within 10 seconds"
            ) from exc

        self.set_enable_service = rospy.Service(
            "toggle_buoy_trajectory",
            SetBool,
            self.handle_enable_service,
        )
        rospy.loginfo(
            "[BuoyTrajectory] Ready: %s -> %s, distance=%.2f m, enabled=%s",
            self.buoy_frame,
            self.close_approach_frame,
            self.approach_distance_m,
            self.is_enabled,
        )

    def handle_enable_service(self, request):
        self.is_enabled = request.data
        message = "Buoy trajectory publishing {}".format(
            "enabled" if self.is_enabled else "disabled"
        )
        rospy.loginfo("[BuoyTrajectory] %s", message)
        return SetBoolResponse(success=True, message=message)

    def lookup_transform(self, child_frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.parent_frame,
                child_frame,
                rospy.Time(0),
                rospy.Duration(self.lookup_timeout_seconds),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                5.0,
                "[BuoyTrajectory] Could not look up %s -> %s: %s",
                self.parent_frame,
                child_frame,
                exc,
            )
            return None

    def build_approach_pose(self, robot_transform, buoy_transform):
        robot = robot_transform.transform.translation
        buoy = buoy_transform.transform.translation
        dx = buoy.x - robot.x
        dy = buoy.y - robot.y
        planar_distance_m = math.hypot(dx, dy)
        if planar_distance_m < self.minimum_planar_distance_m:
            rospy.logwarn_throttle(
                5.0,
                "[BuoyTrajectory] Robot and buoy XY positions overlap",
            )
            return None

        unit_x = dx / planar_distance_m
        unit_y = dy / planar_distance_m
        yaw_rad = math.atan2(dy, dx)
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw_rad)
        return Pose(
            position=Point(
                buoy.x - unit_x * self.approach_distance_m,
                buoy.y - unit_y * self.approach_distance_m,
                buoy.z + self.target_z_offset_m,
            ),
            orientation=Quaternion(*quaternion),
        )

    def build_transform_message(self, pose):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.close_approach_frame
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation
        return transform

    def send_transform(self, transform):
        try:
            response = self.set_object_transform_service.call(
                SetObjectTransformRequest(transform=transform)
            )
        except rospy.ServiceException as exc:
            rospy.logerr_throttle(
                5.0,
                "[BuoyTrajectory] Failed to publish %s: %s",
                transform.child_frame_id,
                exc,
            )
            return False
        if not response.success:
            rospy.logwarn_throttle(
                5.0,
                "[BuoyTrajectory] Backend rejected %s: %s",
                transform.child_frame_id,
                response.message,
            )
            return False
        return True

    def publish_close_approach(self):
        robot_transform = self.lookup_transform(self.robot_frame)
        if robot_transform is None:
            return
        buoy_transform = self.lookup_transform(self.buoy_frame)
        if buoy_transform is None:
            return
        pose = self.build_approach_pose(robot_transform, buoy_transform)
        if pose is None:
            return
        self.send_transform(self.build_transform_message(pose))

    def run(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.publish_close_approach()
            rate.sleep()


def main():
    rospy.init_node("buoy_trajectory_publisher", anonymous=False)
    try:
        BuoyTrajectoryPublisher().run()
    except (RuntimeError, ValueError) as exc:
        rospy.logfatal("[BuoyTrajectory] Startup failed: %s", exc)


if __name__ == "__main__":
    main()
