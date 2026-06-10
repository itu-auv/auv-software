#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped, Point, Quaternion
from std_msgs.msg import Float32
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class PingerFramePublisher:
    def __init__(self):
        rospy.init_node("pinger_frame_publisher_node")

        self.is_collecting = False
        self.samples = []
        self.current_leg_samples = []
        self.pinger_pose = None

        # Parameters
        self.pinger_frame = rospy.get_param("~pinger_frame", "pinger_frame")
        self.waypoint_frame = rospy.get_param("~waypoint_frame", "pinger_waypoint")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.topic_name = rospy.get_param(
            "~topic_name", "/taluy/acoustic/hydrophone/base_angle"
        )
        self.leg_distance = rospy.get_param("~leg_distance", 2.0)
        self.waypoint_sequence = [
            ("forward", 1.0, 0.0),
            ("left", 0.0, 1.0),
            ("right", 0.0, -1.0),
        ]
        self.waypoint_sequence_idx = 0
        self.outlier_threshold = rospy.get_param(
            "~outlier_threshold", 0.26
        )  # ~15 degrees in radians
        self.min_raw_samples = rospy.get_param("~min_raw_samples", 15)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Service to broadcast transforms to the object map tf server
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        rospy.loginfo(
            "PingerFramePublisher: Waiting for set_object_transform service..."
        )
        self.set_object_transform_service.wait_for_service()

        # Services for SMACH
        self.toggle_collection_srv = rospy.Service(
            "toggle_pinger_collection", SetBool, self.handle_toggle_collection
        )
        self.compute_position_srv = rospy.Service(
            "compute_pinger_position", Trigger, self.handle_compute_position
        )
        self.clear_data_srv = rospy.Service(
            "clear_pinger_data", Trigger, self.handle_clear_data
        )
        self.publish_waypoint_srv = rospy.Service(
            "publish_pinger_waypoint", Trigger, self.handle_publish_waypoint
        )
        self.has_enough_samples_srv = rospy.Service(
            "has_enough_pinger_samples", Trigger, self.handle_has_enough_samples
        )

        # Hydrophone subscriber
        self.sub = rospy.Subscriber(self.topic_name, Float32, self.angle_callback)

        rospy.loginfo("PingerFramePublisher initialized successfully.")

    def handle_toggle_collection(self, req):
        if req.data:
            self.current_leg_samples = []
            self.is_collecting = True
            msg = "Started pinger data collection."
            rospy.loginfo(msg)
        else:
            self.is_collecting = False
            if self.current_leg_samples:
                filtered_samples, median_angle = self.get_filtered_current_leg_samples()
                num_removed = len(self.current_leg_samples) - len(filtered_samples)
                self.samples.extend(filtered_samples)
                self.current_leg_samples = []
                msg = (
                    "Stopped pinger data collection. "
                    f"Collected {len(filtered_samples) + num_removed} samples, "
                    f"kept {len(filtered_samples)} (removed {num_removed} outliers, "
                    f"median angle: {math.degrees(median_angle):.1f}°)."
                )
                rospy.loginfo(msg)
            else:
                msg = "Stopped pinger data collection. No samples collected."
                rospy.loginfo(msg)
        return SetBoolResponse(success=True, message=msg)

    def get_filtered_current_leg_samples(self):
        if not self.current_leg_samples:
            return [], 0.0

        angles = [s["angle_world"] for s in self.current_leg_samples]
        angles_np = np.array(angles)
        diffs = np.arctan2(
            np.sin(angles_np[:, None] - angles_np[None, :]),
            np.cos(angles_np[:, None] - angles_np[None, :]),
        )
        sum_abs_diffs = np.sum(np.abs(diffs), axis=1)
        median_idx = np.argmin(sum_abs_diffs)
        median_angle = angles_np[median_idx]

        filtered_samples = []
        for sample in self.current_leg_samples:
            diff = abs(
                np.arctan2(
                    np.sin(sample["angle_world"] - median_angle),
                    np.cos(sample["angle_world"] - median_angle),
                )
            )
            if diff <= self.outlier_threshold:
                filtered_samples.append(sample)

        return filtered_samples, median_angle

    def handle_has_enough_samples(self, req):
        raw_count = len(self.current_leg_samples)
        enough_samples = raw_count >= self.min_raw_samples
        msg = f"Raw pinger samples: {raw_count}/{self.min_raw_samples}"
        if enough_samples:
            rospy.loginfo(msg)
        else:
            rospy.loginfo_throttle(2.0, msg)
        return TriggerResponse(success=enough_samples, message=msg)

    def handle_clear_data(self, req):
        self.samples = []
        self.current_leg_samples = []
        self.pinger_pose = None
        self.waypoint_sequence_idx = 0
        msg = (
            "Cleared all collected pinger samples, reset pinger position, "
            "and reset waypoint sequence."
        )
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    def handle_publish_waypoint(self, req):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(2.0),
            )
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            rx = transform.transform.rotation.x
            ry = transform.transform.rotation.y
            rz = transform.transform.rotation.z
            rw = transform.transform.rotation.w

            _, _, yaw = tf.transformations.euler_from_quaternion([rx, ry, rz, rw])

            waypoint_name, forward_sign, lateral_sign = self.waypoint_sequence[
                self.waypoint_sequence_idx % len(self.waypoint_sequence)
            ]

            # Waypoint target in robot-local coordinates:
            # +X forward, +Y left, -Y right.
            target_x = tx + self.leg_distance * (
                forward_sign * math.cos(yaw) + lateral_sign * -math.sin(yaw)
            )
            target_y = ty + self.leg_distance * (
                forward_sign * math.sin(yaw) + lateral_sign * math.cos(yaw)
            )

            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.waypoint_frame
            t.transform.translation.x = target_x
            t.transform.translation.y = target_y
            t.transform.translation.z = tz
            t.transform.rotation.x = rx
            t.transform.rotation.y = ry
            t.transform.rotation.z = rz
            t.transform.rotation.w = rw

            req_trans = SetObjectTransformRequest()
            req_trans.transform = t
            resp = self.set_object_transform_service.call(req_trans)

            if resp.success:
                self.waypoint_sequence_idx += 1
                msg = (
                    f"Published {waypoint_name} waypoint '{self.waypoint_frame}' "
                    f"at x={target_x:.2f}, y={target_y:.2f}"
                )
                rospy.loginfo(msg)
                return TriggerResponse(success=True, message=msg)
            else:
                return TriggerResponse(
                    success=False,
                    message=f"set_object_transform failed: {resp.message}",
                )

        except Exception as e:
            msg = f"Failed to publish waypoint frame: {e}"
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

    def angle_callback(self, msg):
        if not self.is_collecting:
            return

        try:
            # Look up current base_link in odom frame
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y

            rx = transform.transform.rotation.x
            ry = transform.transform.rotation.y
            rz = transform.transform.rotation.z
            rw = transform.transform.rotation.w

            # Convert quaternion to yaw
            _, _, yaw = tf.transformations.euler_from_quaternion([rx, ry, rz, rw])

            # msg.data is the angle in body frame
            angle_body = msg.data
            angle_world = yaw + angle_body

            # Store the robot position and the calculated absolute bearing direction
            self.current_leg_samples.append(
                {"pos": (tx, ty), "angle_world": angle_world}
            )

            rospy.loginfo_throttle(
                2.0,
                f"Collected sample #{len(self.current_leg_samples)}: pos=({tx:.2f}, {ty:.2f}), "
                f"angle_body={math.degrees(angle_body):.1f}°, "
                f"angle_world={math.degrees(angle_world):.1f}°",
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(2.0, f"TF lookup failed: {e}")

    def handle_compute_position(self, req):
        if len(self.samples) < 2:
            return TriggerResponse(
                success=False,
                message=f"Not enough samples to compute position. Count: {len(self.samples)}",
            )

        # Build linear system to find the intersection of bearing lines
        A = np.zeros((2, 2))
        b = np.zeros(2)

        for sample in self.samples:
            theta = sample["angle_world"]
            nx = -math.sin(theta)
            ny = math.cos(theta)
            n = np.array([nx, ny])
            r = np.array(sample["pos"])

            A += np.outer(n, n)
            b += np.dot(n, r) * n

        try:
            p = np.linalg.solve(A, b)
            self.pinger_pose = Pose()
            self.pinger_pose.position.x = p[0]
            self.pinger_pose.position.y = p[1]
            self.pinger_pose.position.z = 0.0  # assume same depth level or 0
            self.pinger_pose.orientation.w = 1.0

            msg = f"Computed pinger position: x={p[0]:.3f}, y={p[1]:.3f}"
            rospy.loginfo(msg)

            # Publish the initial transform immediately
            self.send_pinger_transform()

            return TriggerResponse(success=True, message=msg)

        except np.linalg.LinAlgError:
            return TriggerResponse(
                success=False, message="Intersection failed: singular matrix"
            )

    def send_pinger_transform(self):
        if self.pinger_pose is None:
            return

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.pinger_frame
        t.transform.translation.x = self.pinger_pose.position.x
        t.transform.translation.y = self.pinger_pose.position.y
        t.transform.translation.z = self.pinger_pose.position.z
        t.transform.rotation = self.pinger_pose.orientation

        req = SetObjectTransformRequest()
        req.transform = t
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(f"Failed to publish {self.pinger_frame}: {resp.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set_object_transform failed: {e}")

    def spin(self):
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if self.pinger_pose is not None:
                self.send_pinger_transform()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PingerFramePublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
