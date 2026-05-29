#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pinger triangulation trajectory publisher.

The acoustic hydrophone only provides a *bearing* (a direction) toward the
pinger, not a range. To recover an actual position we triangulate: the vehicle
takes a bearing measurement from several known spots, each bearing defines a ray
in the odom plane, and the pinger is where those rays cross.

Responsibilities (all frame creation lives here, the smach only orchestrates):
  - On enable, broadcast the four sample target frames offset +/- sample_distance
    along X/Y of `mission_start_link` (frozen into odom by the object map server).
  - Buffer the robot-frame bearing published by the hydrophone node.
  - `record_pinger_bearing` (Trigger): circular-average the recent bearing buffer,
    look up the vehicle pose in odom, and store a ray (origin + direction).
  - `compute_pinger_frame` (Trigger): least-squares intersect the stored rays and
    broadcast the `pinger` frame.
"""

import math
import threading
from collections import deque

import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64
from std_srvs.srv import (
    SetBool,
    SetBoolResponse,
    Trigger,
    TriggerResponse,
)
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class PingerTrajectoryPublisher:
    def __init__(self):
        rospy.init_node("pinger_trajectory_publisher")

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.mission_start_frame = rospy.get_param(
            "~mission_start_frame", "mission_start_link"
        )
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")
        self.pinger_frame = rospy.get_param("~pinger_frame", "pinger")

        # Distance of each sample frame from the mission start, along its X/Y axes.
        self.sample_distance = rospy.get_param("~sample_distance", 2.0)

        # How far back (seconds) to average bearings when a measurement is recorded.
        self.average_window = rospy.get_param("~average_window", 3.0)
        # Maximum age of buffered bearings to keep.
        self.buffer_duration = rospy.get_param("~buffer_duration", 10.0)

        angle_topic = rospy.get_param("~angle_topic", "acoustic/hydrophone/angle")

        # name -> (dx, dy) offset in the mission_start_link frame
        self.sample_frames = {
            "pinger_sample_xp": (self.sample_distance, 0.0),
            "pinger_sample_xn": (-self.sample_distance, 0.0),
            "pinger_sample_yp": (0.0, self.sample_distance),
            "pinger_sample_yn": (0.0, -self.sample_distance),
        }

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._lock = threading.Lock()
        self._bearing_buffer = deque()  # entries: (stamp_sec, bearing_rad)
        self._rays = []  # entries: (px, py, dx, dy) in odom

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.angle_sub = rospy.Subscriber(angle_topic, Float64, self.angle_callback)

        self.enable_service = rospy.Service(
            "toggle_pinger_trajectory", SetBool, self.handle_enable
        )
        self.record_service = rospy.Service(
            "record_pinger_bearing", Trigger, self.handle_record_bearing
        )
        self.compute_service = rospy.Service(
            "compute_pinger_frame", Trigger, self.handle_compute_pinger
        )

        rospy.loginfo(
            "PingerTrajectoryPublisher ready (sample_distance=%.2fm, angle_topic=%s).",
            self.sample_distance,
            rospy.resolve_name(angle_topic),
        )

    # ------------------------------------------------------------------ bearings

    def angle_callback(self, msg: Float64):
        now = rospy.Time.now().to_sec()
        with self._lock:
            self._bearing_buffer.append((now, math.radians(msg.data)))
            cutoff = now - self.buffer_duration
            while self._bearing_buffer and self._bearing_buffer[0][0] < cutoff:
                self._bearing_buffer.popleft()

    def _average_recent_bearing(self):
        """Circular mean (radians) of bearings within the averaging window."""
        now = rospy.Time.now().to_sec()
        cutoff = now - self.average_window
        with self._lock:
            recent = [b for (t, b) in self._bearing_buffer if t >= cutoff]
        if not recent:
            return None, 0
        mean = math.atan2(
            float(np.mean(np.sin(recent))), float(np.mean(np.cos(recent)))
        )
        return mean, len(recent)

    # ------------------------------------------------------------------ services

    def handle_enable(self, request: SetBool) -> SetBoolResponse:
        if not request.data:
            message = "Pinger trajectory disabled."
            rospy.loginfo(message)
            return SetBoolResponse(success=True, message=message)

        # Reset accumulated rays for a fresh triangulation run.
        with self._lock:
            self._rays = []

        if not self._broadcast_sample_frames():
            message = "Failed to broadcast pinger sample frames (no mission start TF?)."
            rospy.logerr(message)
            return SetBoolResponse(success=False, message=message)

        message = "Pinger trajectory enabled, sample frames broadcast."
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def handle_record_bearing(self, request: Trigger) -> TriggerResponse:
        bearing, count = self._average_recent_bearing()
        if bearing is None:
            message = "No bearing samples available to record."
            rospy.logwarn(message)
            return TriggerResponse(success=False, message=message)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(4.0),
            )
        except tf2_ros.TransformException as e:
            message = f"Failed to look up robot pose for bearing: {e}"
            rospy.logerr(message)
            return TriggerResponse(success=False, message=message)

        px = transform.transform.translation.x
        py = transform.transform.translation.y
        q = transform.transform.rotation
        _, _, yaw = tf_conversions.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        # Bearing is in the robot frame; rotate into odom to get the world ray.
        direction = yaw + bearing
        dx = math.cos(direction)
        dy = math.sin(direction)

        with self._lock:
            self._rays.append((px, py, dx, dy))
            num_rays = len(self._rays)

        message = (
            f"Recorded ray #{num_rays} from ({px:.2f}, {py:.2f}) "
            f"heading {math.degrees(direction):.1f}deg (avg of {count} samples)."
        )
        rospy.loginfo(message)
        return TriggerResponse(success=True, message=message)

    def handle_compute_pinger(self, request: Trigger) -> TriggerResponse:
        with self._lock:
            rays = list(self._rays)

        if len(rays) < 2:
            message = f"Need at least 2 rays to triangulate, have {len(rays)}."
            rospy.logwarn(message)
            return TriggerResponse(success=False, message=message)

        point = self._least_squares_intersection(rays)
        if point is None:
            message = "Ray intersection is degenerate (rays nearly parallel)."
            rospy.logerr(message)
            return TriggerResponse(success=False, message=message)

        if not self._broadcast_pinger_frame(point):
            message = "Failed to broadcast the pinger frame."
            rospy.logerr(message)
            return TriggerResponse(success=False, message=message)

        message = f"Pinger frame placed at ({point[0]:.2f}, {point[1]:.2f}) in odom."
        rospy.loginfo(message)
        return TriggerResponse(success=True, message=message)

    # ------------------------------------------------------------------ geometry

    @staticmethod
    def _least_squares_intersection(rays):
        """
        Least-squares intersection of 2D lines, each defined by a point p and a
        unit direction d. Minimizes the sum of squared perpendicular distances:
        solve (sum M_i) x = sum M_i p_i with M_i = I - d_i d_i^T.
        """
        a_mat = np.zeros((2, 2))
        b_vec = np.zeros(2)
        for px, py, dx, dy in rays:
            d = np.array([dx, dy])
            p = np.array([px, py])
            m = np.eye(2) - np.outer(d, d)
            a_mat += m
            b_vec += m @ p

        if abs(np.linalg.det(a_mat)) < 1e-9:
            return None
        try:
            return np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            return None

    # ----------------------------------------------------------------- broadcast

    def _broadcast_sample_frames(self) -> bool:
        ok = True
        for frame_name, (dx, dy) in self.sample_frames.items():
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            # Parent is mission_start_link; the object map server freezes the
            # resolved pose into its odom reference at injection time.
            transform.header.frame_id = self.mission_start_frame
            transform.child_frame_id = frame_name
            transform.transform.translation.x = dx
            transform.transform.translation.y = dy
            transform.transform.translation.z = 0.0
            transform.transform.rotation.w = 1.0
            ok = self._send_transform(transform) and ok
        return ok

    def _broadcast_pinger_frame(self, point) -> bool:
        # Orient the frame to face from the mission start toward the pinger.
        yaw = 0.0
        z = 0.0
        try:
            ms = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.mission_start_frame,
                rospy.Time(0),
                rospy.Duration(4.0),
            )
            yaw = math.atan2(
                point[1] - ms.transform.translation.y,
                point[0] - ms.transform.translation.x,
            )
            z = ms.transform.translation.z
        except tf2_ros.TransformException as e:
            rospy.logwarn(
                "Could not look up mission start for pinger orientation: %s", e
            )

        q = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.pinger_frame
        transform.transform.translation.x = float(point[0])
        transform.transform.translation.y = float(point[1])
        transform.transform.translation.z = z
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        return self._send_transform(transform)

    def _send_transform(self, transform: TransformStamped) -> bool:
        request = SetObjectTransformRequest()
        request.transform = transform
        try:
            response = self.set_object_transform_service.call(request)
            if not response.success:
                rospy.logerr(
                    "Failed to set transform for %s: %s",
                    transform.child_frame_id,
                    response.message,
                )
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr("set_object_transform call failed: %s", e)
            return False


if __name__ == "__main__":
    try:
        node = PingerTrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
