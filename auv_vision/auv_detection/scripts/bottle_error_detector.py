#!/usr/bin/env python3

"""
Bottom-Cam Visual Servo Errors (ROS1)

Subscribes:
  - sensor_msgs/Image (mono8 mask) -> ~mask_topic

Publishes:
  - std_msgs/Float32MultiArray -> bottle_vsc_errors
      data[0] = e_vert_px (piksel)
      data[1] = e_horiz_px (piksel)
      data[2] = angle_to_horizontal (radyan)
      data[3] = width_px (piksel)
  - sensor_msgs/Image -> bottle_vsc_viz
"""

import math
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from skimage.morphology import skeletonize


def fit_line_from_points(points_xy: np.ndarray):
    if points_xy is None or len(points_xy) < 2:
        return None
    v = cv2.fitLine(points_xy.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(x) for x in v.flatten()]
    return vx, vy, x0, y0


def angle_to_horizontal_from_v(vx: float, vy: float) -> float:
    ang = -math.atan2(vy, vx)
    if ang > math.pi / 2:
        ang -= math.pi
    elif ang < -math.pi / 2:
        ang += math.pi
    return ang


def vertical_height_on_col(binary_mask: np.ndarray, col_idx: int):
    h, w = binary_mask.shape
    col_idx = int(np.clip(col_idx, 0, w - 1))
    col = binary_mask[:, col_idx]
    ys = np.flatnonzero(col)
    if ys.size == 0:
        return None, None, 0
    y_min = int(ys.min())
    y_max = int(ys.max())
    return y_min, y_max, int(y_max - y_min + 1)


def global_vertical_extent(binary_mask: np.ndarray):
    ys, xs = np.where(binary_mask)
    if ys.size == 0:
        return None, None, 0
    y_min = int(ys.min())
    y_max = int(ys.max())
    return y_min, y_max, int(y_max - y_min + 1)


class VSCErrorNode(object):
    def __init__(self):
        self.mask_topic = rospy.get_param("~mask_topic", "pipe_mask")

        self.flip_x = bool(rospy.get_param("~flip_x", False))
        self.flip_y = bool(rospy.get_param("~flip_y", False))

        self.min_area_px = int(rospy.get_param("~min_area_px", 500))

        self.errors_topic = rospy.get_param("~errors_topic", "bottle_vsc_errors")
        self.viz_topic = rospy.get_param("~viz_topic", "bottle_vsc_viz")

        self.bridge = CvBridge()
        self.pub_err = rospy.Publisher(
            self.errors_topic, Float32MultiArray, queue_size=1
        )
        self.pub_viz = rospy.Publisher(self.viz_topic, Image, queue_size=1)
        self.sub = rospy.Subscriber(
            self.mask_topic, Image, self.cb_mask, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(
            "[vsc_errors] Subscribing: %s | Publishing errors: %s | viz: %s",
            self.mask_topic,
            self.errors_topic,
            self.viz_topic,
        )

    def cb_mask(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        if self.flip_x:
            mask = np.ascontiguousarray(np.fliplr(mask))
        if self.flip_y:
            mask = np.ascontiguousarray(np.flipud(mask))

        h, w = mask.shape[:2]
        u0, v0 = w * 6.5 / 8.0, h * 0.5

        area = int(np.count_nonzero(mask))
        if area < self.min_area_px:
            self._publish_errors_nan()
            self._publish_viz(mask, None, None, None, center=None)
            return

        binary = mask > 127

        skel = None
        try:
            skel = skeletonize(binary)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[vsc_errors] skeletonize failed: %s", e)
            skel = None

        line_fit = None
        if skel is not None and np.any(skel):
            ys, xs = np.where(skel)
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            if pts.shape[0] >= 2:
                line_fit = fit_line_from_points(pts)

        if line_fit is None:
            _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = _ret[1] if len(_ret) == 3 else _ret[0]
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                pts = contours[0].reshape(-1, 2).astype(np.float32)
                if pts.shape[0] >= 2:
                    line_fit = fit_line_from_points(pts)

        angle_to_horizontal = float("nan")
        if line_fit is not None:
            vx, vy, x0, y0 = line_fit
            angle_to_horizontal = angle_to_horizontal_from_v(vx, vy)

        cx, cy = float("nan"), float("nan")
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

        width_line_x = int(round(cx)) if not math.isnan(cx) else int(round(u0))
        yT, yB, height_on_col = vertical_height_on_col(binary, width_line_x)
        if height_on_col == 0:
            yT, yB, height_on_col = global_vertical_extent(binary)

        width_px = float(height_on_col) if height_on_col is not None else float("nan")

        e_vert_px = float("nan") if math.isnan(cy) else (v0 - cy)
        e_horiz_px = float("nan") if math.isnan(cx) else (u0 - cx)

        out = Float32MultiArray()
        out.data = [
            float(e_vert_px),
            float(e_horiz_px),
            float(angle_to_horizontal),
            float(width_px),
        ]
        self.pub_err.publish(out)

        width_segment = (
            (yT, yB, width_line_x) if (yT is not None and yB is not None) else None
        )
        self._publish_viz(mask, line_fit, width_segment, (u0, v0), center=(cx, cy))

    def _publish_errors_nan(self):
        out = Float32MultiArray()
        nan = float("nan")
        out.data = [nan, nan, nan, nan]
        self.pub_err.publish(out)

    def _publish_viz(self, mask, line_fit, width_segment, center_lines, center=None):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        h, w = mask.shape[:2]

        if center_lines is not None:
            u0, v0 = center_lines
            cv2.line(vis, (int(u0), 0), (int(u0), h - 1), (255, 255, 255), 2)
            cv2.line(vis, (0, int(v0)), (w - 1, int(v0)), (255, 255, 255), 2)
            rospy.loginfo_throttle(
                5.0, f"Viz: img_size=({w}x{h}), u0={u0:.1f}, v0={v0:.1f}"
            )

        if line_fit is not None:
            vx, vy, x0, y0 = line_fit
            p1 = (int(x0 - 2000 * vx), int(y0 - 2000 * vy))
            p2 = (int(x0 + 2000 * vx), int(y0 + 2000 * vy))
            cv2.line(vis, p1, p2, (0, 0, 255), 2)

        if width_segment is not None:
            yT, yB, xCol = width_segment
            cv2.line(vis, (int(xCol), int(yT)), (int(xCol), int(yB)), (0, 165, 255), 2)
            cv2.circle(vis, (int(xCol), int(yT)), 3, (0, 165, 255), -1)
            cv2.circle(vis, (int(xCol), int(yB)), 3, (0, 165, 255), -1)

        if center is not None and not any(math.isnan(c) for c in center):
            cx, cy = center
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)

        try:
            msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            self.pub_viz.publish(msg)
        except Exception as e:
            rospy.logwarn("viz publish failed: %s", e)


def main():
    rospy.init_node("bottom_cam_vsc_errors")
    VSCErrorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
