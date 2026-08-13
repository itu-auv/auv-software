#!/usr/bin/env python3

import math

import rospy
import tf.transformations
import tf2_ros
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse

from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class RelativeApproachFramePublisher:
    """Publish an approach frame between a robot and a detected target frame.

    Frame names and geometry are parameters so the same node can later be used
    for another detected object without changing its implementation.
    """

    def __init__(self):
        rospy.init_node("pinger_teknofest_trajectory_publisher")

        self.is_enabled = rospy.get_param("~enabled", False)
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")
        self.source_frame = rospy.get_param("~source_frame", "pinger_bbox")
        self.output_frame = rospy.get_param("~output_frame", "pinger_close_approach")
        self.approach_distance = float(rospy.get_param("~approach_distance", 2.0))
        self.z_offset = float(rospy.get_param("~z_offset", 0.0))
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 0.5))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.minimum_planar_distance = float(
            rospy.get_param("~minimum_planar_distance", 0.05)
        )

        if self.approach_distance < 0.0:
            raise ValueError("~approach_distance must be non-negative")
        if self.publish_rate <= 0.0:
            raise ValueError("~publish_rate must be positive")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.set_enable_service = rospy.Service(
            "toggle_pinger_teknofest_trajectory",
            SetBool,
            self.handle_enable_service,
        )

        rospy.loginfo(
            "Relative approach publisher initialized: %s -> %s (%.2f m)",
            self.source_frame,
            self.output_frame,
            self.approach_distance,
        )

    def handle_enable_service(self, request):
        self.is_enabled = request.data
        message = "Pinger Teknofest trajectory publishing {}".format(
            "enabled" if self.is_enabled else "disabled"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def lookup_transform(self, child_frame_id):
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                child_frame_id,
                rospy.Time(0),
                rospy.Duration(self.lookup_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                5.0,
                "Could not look up %s -> %s: %s",
                self.odom_frame,
                child_frame_id,
                exc,
            )
            return None

    def build_approach_pose(self, robot_transform, source_transform):
        robot = robot_transform.transform.translation
        source = source_transform.transform.translation

        dx = source.x - robot.x
        dy = source.y - robot.y
        planar_distance = math.hypot(dx, dy)
        if planar_distance < self.minimum_planar_distance:
            rospy.logwarn_throttle(
                5.0,
                "Cannot build %s: %s and %s have the same XY position",
                self.output_frame,
                self.robot_frame,
                self.source_frame,
            )
            return None

        unit_x = dx / planar_distance
        unit_y = dy / planar_distance
        approach_x = source.x - unit_x * self.approach_distance
        approach_y = source.y - unit_y * self.approach_distance

        # Match the other trajectory frames: orient the published frame along
        # the base_link -> output-frame vector. When both positions overlap,
        # fall back to the base_link -> source-frame direction.
        approach_dx = approach_x - robot.x
        approach_dy = approach_y - robot.y
        if math.hypot(approach_dx, approach_dy) < self.minimum_planar_distance:
            approach_dx = dx
            approach_dy = dy
        yaw = math.atan2(approach_dy, approach_dx)
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)

        return Pose(
            position=Point(
                approach_x,
                approach_y,
                source.z + self.z_offset,
            ),
            orientation=Quaternion(*quaternion),
        )

    def build_transform_message(self, child_frame_id, pose):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation = pose.position
        transform.transform.rotation = pose.orientation
        return transform

    def send_transform(self, transform):
        request = SetObjectTransformRequest(transform=transform)
        try:
            response = self.set_object_transform_service.call(request)
        except rospy.ServiceException as exc:
            rospy.logerr_throttle(
                5.0,
                "Failed to publish %s: %s",
                transform.child_frame_id,
                exc,
            )
            return False

        if not response.success:
            rospy.logwarn_throttle(
                5.0,
                "Failed to set transform for %s: %s",
                transform.child_frame_id,
                response.message,
            )
            return False
        return True

    def publish_approach_frame(self):
        robot_transform = self.lookup_transform(self.robot_frame)
        if robot_transform is None:
            return

        source_transform = self.lookup_transform(self.source_frame)
        if source_transform is None:
            return

        approach_pose = self.build_approach_pose(robot_transform, source_transform)
        if approach_pose is None:
            return

        transform = self.build_transform_message(self.output_frame, approach_pose)
        self.send_transform(transform)

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.publish_approach_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        RelativeApproachFramePublisher().run()
    except (rospy.ROSInterruptException, ValueError) as exc:
        rospy.logerr("Pinger Teknofest trajectory publisher stopped: %s", exc)
