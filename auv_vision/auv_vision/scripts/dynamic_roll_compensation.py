#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
import message_filters

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class CameraRollStabilizer:
    """
    - Subscribes: Odometry + Rectified Color Image (time-synchronized)
    - Publishes:  Corrected Image (+TF from camera optical -> stabilized optical)
    - Correction: cancels camera roll, rotating image and broadcasting a Z-axis rotation in the optical frame.
    """

    def __init__(self):
        rospy.init_node("camera_roll_stabilizer", anonymous=True)
        rospy.loginfo("Camera roll stabilizer node starting...")

        # Message_filters sync tuning
        self.sync_queue = int(rospy.get_param("~sync_queue_size", 10))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))  # seconds

        # If no odometry arrives within this window, publish a placeholder image
        # telling the operator to start localization.
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 1.0))
        self.last_odom_time = rospy.Time(0)

        # Only recompute warp matrix if angle changes beyond this (degrees) or dims change
        self.recalc_threshold_deg = float(rospy.get_param("~recalc_threshold_deg", 0.1))

        # Border behavior for rotation (constant black)
        self.border_mode = cv2.BORDER_CONSTANT

        # -------- Publishers / TF --------
        self.img_pub = rospy.Publisher("camera/image_corrected", Image, queue_size=2)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # -------- Converters / State --------
        self.bridge = CvBridge()
        self.lock = threading.Lock()  # protect small shared state if needed later

        self.last_dimensions = None  # (w, h)
        self.last_angle_deg = None
        self.M = None  # cached 2x3 rotation matrix

        img_sub = message_filters.Subscriber(
            "camera/image_rect_color", Image, queue_size=1
        )
        odom_sub = message_filters.Subscriber("odometry", Odometry, queue_size=1)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, odom_sub], queue_size=self.sync_queue, slop=self.sync_slop
        )
        self.sync.registerCallback(self.synced_callback)

        # Track odom arrival independently; timer publishes placeholder if stale.
        rospy.Subscriber("odometry", Odometry, self._odom_watchdog, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self._placeholder_tick)

    def synced_callback(self, img_msg: Image, odom_msg: Odometry):
        """
        Called with time-synchronized Image and Odometry.
        We:
          1) Compute roll from odometry quaternion.
          2) Compute correction = -roll (cancel roll).
          3) Broadcast TF rotation about Z in optical frame by correction.
          4) Rotate image by correction (degrees) and publish.
        """
        # 1) Roll from odometry
        q = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ]

        if not self._valid_quat(q):
            rospy.logwarn_throttle(
                5.0,
                "[camera_roll_stabilizer] Received invalid quaternion; skipping frame.",
            )
            return

        try:
            roll, pitch, yaw = euler_from_quaternion(q)
        except Exception as e:
            rospy.logwarn_throttle(
                5.0, "[camera_roll_stabilizer] Euler conversion failed: %s", str(e)
            )
            return

        # 2) Correction angle (radians)
        correction_rad = -roll
        correction_deg = math.degrees(correction_rad)

        # 3) Broadcast TF in optical frame: rotation about Z by correction
        #    (Image rotation corresponds to spin around optical axis -> Z in optical frame)
        self._broadcast_stabilized_tf(img_msg.header.stamp, correction_rad)

        # 4) Rotate image and publish
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(
                5.0, "[camera_roll_stabilizer] CvBridge decode failed: %s", str(e)
            )
            return

        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)

        # Cache / reuse warp matrix when angle/dimensions unchanged
        if (
            self.M is None
            or self.last_dimensions != (w, h)
            or self.last_angle_deg is None
            or abs(self.last_angle_deg - correction_deg) > self.recalc_threshold_deg
        ):
            self.M = cv2.getRotationMatrix2D(center, correction_deg, 1.0)
            self.last_angle_deg = correction_deg
            self.last_dimensions = (w, h)

        # Warp
        rotated = cv2.warpAffine(
            cv_image,
            self.M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
        )

        try:
            out_msg = self.bridge.cv2_to_imgmsg(rotated, encoding="bgr8")
            # Keep original header/time and frame
            out_msg.header = img_msg.header
            self.img_pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logwarn_throttle(
                5.0, "[camera_roll_stabilizer] CvBridge encode failed: %s", str(e)
            )

    def _odom_watchdog(self, odom_msg: Odometry):
        self.last_odom_time = odom_msg.header.stamp

    def _placeholder_tick(self, _event):
        if (rospy.Time.now() - self.last_odom_time).to_sec() < self.odom_timeout:
            return

        h, w = 480, 640
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        text = "bro localization baslat"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        org = ((w - text_w) // 2, (h + text_h) // 2)
        cv2.putText(
            placeholder, text, org, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA
        )

        out_msg = self.bridge.cv2_to_imgmsg(placeholder, encoding="bgr8")
        out_msg.header.stamp = rospy.Time.now()
        self.img_pub.publish(out_msg)

    # ---------------- Helpers ----------------
    def _broadcast_stabilized_tf(self, stamp, correction_rad: float):
        """
        Publish a TransformStamped from parent optical frame to a stabilized optical frame
        with rotation about Z by 'correction_rad' (cancelling roll in image space).
        """
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = rospy.get_param(
            "~camera_optical_frame", "taluy/base_link/front_camera_optical_link"
        )
        t.child_frame_id = rospy.get_param(
            "~camera_optical_frame_stabilized",
            "taluy/base_link/front_camera_optical_link_stabilized",
        )

        # No translation offset; pure spin around optical axis
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Rotate about Z in optical frame
        q_rot = quaternion_from_euler(0.0, 0.0, correction_rad)
        t.transform.rotation.x = q_rot[0]
        t.transform.rotation.y = q_rot[1]
        t.transform.rotation.z = q_rot[2]
        t.transform.rotation.w = q_rot[3]

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def _valid_quat(q):
        # reject NaNs and zero-length
        if any((x != x) for x in q):  # NaN check
            return False
        norm = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        return 0.5 < norm < 2.0  # loose sanity bounds

    def spin(self):
        rospy.loginfo("Camera roll stabilizer node is up.")
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraRollStabilizer()
        node.spin()
    except rospy.ROSInterruptException:
        pass
