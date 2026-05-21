#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy
import threading
import time

import cv2
import message_filters
import rospy
import tf2_ros

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class CameraRollStabilizer:
    def __init__(self):
        rospy.init_node("camera_roll_stabilizer", anonymous=True)
        rospy.loginfo("Camera roll stabilizer node starting...")

        self.sync_queue = int(rospy.get_param("~sync_queue_size", 10))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.recalc_threshold_deg = float(rospy.get_param("~recalc_threshold_deg", 0.1))
        self.passthrough_threshold_deg = float(
            rospy.get_param("~passthrough_threshold_deg", 1.0)
        )
        self.fallback_odom_timeout = float(
            rospy.get_param("~fallback_odom_timeout", 0.5)
        )
        self.crop_to_valid_region = bool(
            rospy.get_param("~crop_to_valid_region", True)
        )

        self.border_mode = cv2.BORDER_CONSTANT

        self.camera_optical_frame = rospy.get_param(
            "~camera_optical_frame", "taluy/base_link/front_camera_optical_link"
        )
        self.camera_optical_frame_stabilized = rospy.get_param(
            "~camera_optical_frame_stabilized",
            "taluy/base_link/front_camera_optical_link_stabilized",
        )

        self.img_pub = rospy.Publisher("camera/image_stabilized", Image, queue_size=2)
        self.info_pub = rospy.Publisher(
            "camera/camera_info_stabilized", CameraInfo, queue_size=2
        )
        self.debug_img_pub = rospy.Publisher(
            "camera/debug/image_stabilized_crop", Image, queue_size=1
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.last_dimensions = None
        self.last_angle_deg = None
        self.last_center = None
        self.rotation_matrix = None
        self.crop_rect = None
        self.camera_info = None
        self.last_odom_time = None

        self.perf_log_period = float(rospy.get_param("~perf_log_period", 5.0))
        self.perf_frame_count = 0
        self.perf_total_time = 0.0
        self.perf_max_time = 0.0
        self.perf_window_start = None

        img_sub = message_filters.Subscriber(
            "camera/image_rect_color", Image, queue_size=1
        )
        odom_sub = message_filters.Subscriber("odometry", Odometry, queue_size=1)
        rospy.Subscriber("odometry", Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(
            "camera/camera_info",
            CameraInfo,
            self._camera_info_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "camera/image_rect_color",
            Image,
            self._fallback_image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, odom_sub], queue_size=self.sync_queue, slop=self.sync_slop
        )
        self.sync.registerCallback(self.synced_callback)

    def synced_callback(self, img_msg: Image, odom_msg: Odometry):
        t_start = time.perf_counter()
        try:
            self._synced_callback_impl(img_msg, odom_msg)
        finally:
            self._record_frame_time(time.perf_counter() - t_start)

    def _record_frame_time(self, elapsed: float):
        now = time.perf_counter()
        if self.perf_window_start is None:
            self.perf_window_start = now

        self.perf_frame_count += 1
        self.perf_total_time += elapsed
        if elapsed > self.perf_max_time:
            self.perf_max_time = elapsed

        window = now - self.perf_window_start
        if self.perf_log_period > 0.0 and window >= self.perf_log_period:
            avg_ms = (self.perf_total_time / self.perf_frame_count) * 1000.0
            max_ms = self.perf_max_time * 1000.0
            hz = self.perf_frame_count / window if window > 0.0 else 0.0
            rospy.loginfo(
                "[camera_roll_stabilizer] frames=%d window=%.2fs rate=%.2f Hz "
                "avg=%.2f ms max=%.2f ms",
                self.perf_frame_count,
                window,
                hz,
                avg_ms,
                max_ms,
            )
            self.perf_frame_count = 0
            self.perf_total_time = 0.0
            self.perf_max_time = 0.0
            self.perf_window_start = now

    def _synced_callback_impl(self, img_msg: Image, odom_msg: Odometry):
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
            roll, _, _ = euler_from_quaternion(q)
        except Exception as e:
            rospy.logwarn_throttle(
                5.0, "[camera_roll_stabilizer] Euler conversion failed: %s", str(e)
            )
            return

        correction_rad = -roll
        correction_deg = math.degrees(correction_rad)
        self._broadcast_stabilized_tf(img_msg.header.stamp, correction_rad)

        if (
            self.img_pub.get_num_connections() == 0
            and self.info_pub.get_num_connections() == 0
            and self.debug_img_pub.get_num_connections() == 0
        ):
            rospy.logdebug_throttle(
                5.0,
                "[camera_roll_stabilizer] No stabilized image/info subscribers; skipping image rotation.",
            )
            return

        if abs(correction_deg) < self.passthrough_threshold_deg:
            self._publish_passthrough(img_msg, (0, 0, img_msg.width, img_msg.height))
            return

        h = img_msg.height
        w = img_msg.width
        center = self._rotation_center(w, h)

        if (
            self.rotation_matrix is None
            or self.last_dimensions != (w, h)
            or self.last_center != center
            or self.last_angle_deg is None
            or abs(self.last_angle_deg - correction_deg) > self.recalc_threshold_deg
        ):
            self.rotation_matrix = cv2.getRotationMatrix2D(center, correction_deg, 1.0)
            self.last_angle_deg = correction_deg
            self.last_dimensions = (w, h)
            self.last_center = center
            self.crop_rect = self._compute_valid_crop_rect(
                w, h, center, correction_rad
            )

        crop_rect = self.crop_rect or (0, 0, w, h)
        self._publish_stabilized_camera_info(img_msg, crop_rect)

        publish_image = self.img_pub.get_num_connections() > 0
        publish_debug = self.debug_img_pub.get_num_connections() > 0
        if not publish_image and not publish_debug:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(
                5.0, "[camera_roll_stabilizer] CvBridge decode failed: %s", str(e)
            )
            return

        rotated = cv2.warpAffine(
            cv_image,
            self.rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
        )
        self._publish_crop_debug_image(rotated, img_msg, crop_rect)

        if not publish_image:
            return

        x0, y0, x1, y1 = crop_rect
        rotated = rotated[y0:y1, x0:x1]

        try:
            out_msg = self.bridge.cv2_to_imgmsg(rotated, encoding="bgr8")
            out_msg.header.stamp = img_msg.header.stamp
            out_msg.header.seq = img_msg.header.seq
            out_msg.header.frame_id = self.camera_optical_frame_stabilized
            self.img_pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logwarn_throttle(
                5.0, "[camera_roll_stabilizer] CvBridge encode failed: %s", str(e)
            )

    def _odom_callback(self, _msg: Odometry):
        self.last_odom_time = rospy.Time.now()

    def _camera_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.camera_info = msg

    def _fallback_image_callback(self, img_msg: Image):
        if (
            self.img_pub.get_num_connections() == 0
            and self.info_pub.get_num_connections() == 0
        ):
            return

        if self.last_odom_time is None:
            rospy.logwarn_throttle(
                5.0,
                "[camera_roll_stabilizer] can't publish stabilized front camera image: localization unavailable",
            )
            return

        odom_age = (rospy.Time.now() - self.last_odom_time).to_sec()
        if odom_age > self.fallback_odom_timeout:
            rospy.logwarn_throttle(
                5.0,
                "[camera_roll_stabilizer] can't publish stabilized front camera image: localization unavailable",
            )

    def _publish_passthrough(self, img_msg: Image, crop_rect):
        self._publish_stabilized_camera_info(img_msg, crop_rect)
        if self.img_pub.get_num_connections() == 0:
            return

        out_msg = Image()
        out_msg.header.stamp = img_msg.header.stamp
        out_msg.header.seq = img_msg.header.seq
        out_msg.header.frame_id = self.camera_optical_frame_stabilized
        out_msg.height = img_msg.height
        out_msg.width = img_msg.width
        out_msg.encoding = img_msg.encoding
        out_msg.is_bigendian = img_msg.is_bigendian
        out_msg.step = img_msg.step
        out_msg.data = img_msg.data
        self.img_pub.publish(out_msg)

    def _publish_crop_debug_image(self, rotated, img_msg: Image, crop_rect):
        if self.debug_img_pub.get_num_connections() == 0:
            return

        x0, y0, x1, y1 = crop_rect
        debug_image = rotated.copy()
        cv2.rectangle(debug_image, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            out_msg.header.stamp = img_msg.header.stamp
            out_msg.header.seq = img_msg.header.seq
            out_msg.header.frame_id = self.camera_optical_frame_stabilized
            self.debug_img_pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logwarn_throttle(
                5.0,
                "[camera_roll_stabilizer] Crop debug image encode failed: %s",
                str(e),
            )

    def _rotation_center(self, width: int, height: int):
        with self.lock:
            camera_info = copy.deepcopy(self.camera_info)

        if (
            camera_info is not None
            and camera_info.width == width
            and camera_info.height == height
        ):
            return (float(camera_info.K[2]), float(camera_info.K[5]))

        if camera_info is not None:
            rospy.logwarn_throttle(
                5.0,
                "[camera_roll_stabilizer] CameraInfo dimensions (%sx%s) do not match image dimensions (%sx%s); rotating around image center.",
                camera_info.width,
                camera_info.height,
                width,
                height,
            )

        return (width * 0.5, height * 0.5)

    def _compute_valid_crop_rect(
        self, width: int, height: int, center, angle_rad: float
    ):
        if not self.crop_to_valid_region:
            return (0, 0, width, height)
        return self._largest_inscribed_rect(
            width, height, center[0], center[1], angle_rad
        )

    @staticmethod
    def _largest_inscribed_rect(
        width: int, height: int, cx: float, cy: float, angle_rad: float
    ):
        """Closed-form largest axis-aligned rectangle, centered at (cx, cy),
        that fits inside the source rectangle (width x height) rotated by
        `angle_rad` around (cx, cy).

        For half-sizes (rw, rh) the four rotated-edge inequalities collapse
        (with s=|sin a|, c=|cos a|, d_x=min(cx, W-cx), d_y=min(cy, H-cy)) to:
            c*rw + s*rh <= d_x
            s*rw + c*rh <= d_y
        Maximising rw*rh has two regimes:
          - both constraints bind  iff sin(2a) <= min(d_x,d_y)/max(d_x,d_y)
            => 2x2 linear closed form.
          - else only the tighter constraint binds; the other axis is set
            by the unconstrained-area optimum on a single line.
        """
        angle = abs(angle_rad) % math.pi
        if angle > 0.5 * math.pi:
            angle = math.pi - angle

        s = math.sin(angle)
        c = math.cos(angle)
        if s < 1e-9:
            return (0, 0, width, height)

        d_x = min(cx, width - cx)
        d_y = min(cy, height - cy)
        if d_x <= 0.0 or d_y <= 0.0:
            return (0, 0, width, height)

        sin_2a = 2.0 * s * c
        ratio = min(d_x, d_y) / max(d_x, d_y)

        if sin_2a <= ratio:
            cos_2a = c * c - s * s
            rw = (d_x * c - d_y * s) / cos_2a
            rh = (d_y * c - d_x * s) / cos_2a
        else:
            if d_x <= d_y:
                rw = d_x / (2.0 * c)
                rh = d_x / (2.0 * s)
            else:
                rw = d_y / (2.0 * s)
                rh = d_y / (2.0 * c)

        if (
            not math.isfinite(rw)
            or not math.isfinite(rh)
            or rw <= 0.0
            or rh <= 0.0
        ):
            return (0, 0, width, height)

        crop_w = max(1, min(width, int(math.floor(2.0 * rw)) - 2))
        crop_h = max(1, min(height, int(math.floor(2.0 * rh)) - 2))

        x0 = int(math.ceil(cx - 0.5 * crop_w))
        y0 = int(math.ceil(cy - 0.5 * crop_h))
        x0 = max(0, min(width - crop_w, x0))
        y0 = max(0, min(height - crop_h, y0))
        return (x0, y0, x0 + crop_w, y0 + crop_h)

    def _publish_stabilized_camera_info(self, img_msg: Image, crop_rect):
        if self.info_pub.get_num_connections() == 0:
            return

        with self.lock:
            camera_info = copy.deepcopy(self.camera_info)

        if camera_info is None:
            rospy.logwarn_throttle(
                5.0,
                "[camera_roll_stabilizer] No CameraInfo received; cannot publish stabilized CameraInfo.",
            )
            return

        x0, y0, x1, y1 = crop_rect
        stabilized = camera_info
        stabilized.header.stamp = img_msg.header.stamp
        stabilized.header.seq = img_msg.header.seq
        stabilized.header.frame_id = self.camera_optical_frame_stabilized
        stabilized.width = x1 - x0
        stabilized.height = y1 - y0

        stabilized.K = list(stabilized.K)
        stabilized.P = list(stabilized.P)
        stabilized.K[2] -= x0
        stabilized.K[5] -= y0
        stabilized.P[2] -= x0
        stabilized.P[6] -= y0

        stabilized.roi.x_offset = 0
        stabilized.roi.y_offset = 0
        stabilized.roi.width = stabilized.width
        stabilized.roi.height = stabilized.height
        stabilized.roi.do_rectify = False

        self.info_pub.publish(stabilized)

    def _broadcast_stabilized_tf(self, stamp, correction_rad: float):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.camera_optical_frame
        t.child_frame_id = self.camera_optical_frame_stabilized

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        q_rot = quaternion_from_euler(0.0, 0.0, correction_rad)
        t.transform.rotation.x = q_rot[0]
        t.transform.rotation.y = q_rot[1]
        t.transform.rotation.z = q_rot[2]
        t.transform.rotation.w = q_rot[3]

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def _valid_quat(q):
        if not all(math.isfinite(x) for x in q):
            return False
        norm = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        return 0.5 < norm < 2.0

    def spin(self):
        rospy.loginfo("Camera roll stabilizer node is up.")
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CameraRollStabilizer()
        node.spin()
    except rospy.ROSInterruptException:
        pass
