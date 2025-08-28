#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipe Line Follower (ROS1)
- Subscribes to a binary pipe mask (white=pipe, black=background)
- Uses two simple strategies in combination:
  (A) Pseudo line sensors at a "front gate" column band to compute vertical offset error
  (B) Global orientation from cv2.fitLine to reduce yaw error (keep pipe horizontal)
- Publishes geometry_msgs/Twist to cmd_vel

Assumptions
- Incoming image enc: mono8
- White pixels (255) mark the pipe region
- Camera looks downward; vehicle forward corresponds to image X (columns) from left→right
  (If your camera is rotated/flipped, adjust rotate_cw / flip_{x,y} params.)

Debug
- Publishes an annotated overlay image to ~debug_image (sensor bands, fitted line, etc.)
"""

import math
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_srvs.srv import SetBool, SetBoolResponse


def saturate(x, lo, hi):
    return max(lo, min(hi, x))


def rot90n(img, n_cw):
    """Rotate image by n*90 degrees clockwise."""
    n_cw = int(n_cw) % 4
    if n_cw == 0:
        return img
    elif n_cw == 1:
        return np.ascontiguousarray(np.rot90(img, k=3))
    elif n_cw == 2:
        return np.ascontiguousarray(np.rot90(img, k=2))
    else:  # 3
        return np.ascontiguousarray(np.rot90(img, k=1))


class PipeLineFollower(object):
    def __init__(self):
        # ---- Topics ----
        self.mask_topic = rospy.get_param("~mask_topic", "pipe_mask")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/taluy/cmd_vel")
        self.debug_topic = rospy.get_param("~debug_topic", "pipe_follow_debug")

        # ---- Image orientation controls ----
        # Bring your image into the assumed frame: columns≈forward (left→right), rows≈lateral (top→bottom)
        self.rotate_cw = rospy.get_param("~rotate_cw", 0)  # 0,1,2,3 (×90° clockwise)
        self.flip_x = rospy.get_param("~flip_x", False)  # horizontal flip
        self.flip_y = rospy.get_param("~flip_y", False)  # vertical flip

        # ---- Sensor band config ----
        self.n_sensors = int(rospy.get_param("~n_sensors", 5))  # 3 or 5 typical
        self.sensor_x_frac = float(
            rospy.get_param("~sensor_x_frac", 0.20)
        )  # gate center as fraction of width
        self.sensor_width_frac = float(
            rospy.get_param("~sensor_width_frac", 0.10)
        )  # gate width fraction
        self.sensor_min_fill = float(
            rospy.get_param("~sensor_min_fill", 0.02)
        )  # min occupancy to count band

        # ---- Control gains & limits ----
        self.k_y = float(
            rospy.get_param("~k_y", 1.50)
        )  # lateral alignment gain from sensor bands
        self.k_ang = float(rospy.get_param("~k_ang", 1.00))  # orientation (angle) gain
        self.v_fwd = float(
            rospy.get_param("~v_fwd", 0.25)
        )  # nominal forward speed (m/s)
        self.v_min = float(rospy.get_param("~v_min", 0.05))  # min forward speed
        self.ang_limit = float(rospy.get_param("~ang_limit", 0.6))  # rad/s
        self.lin_limit = float(rospy.get_param("~lin_limit", 1))  # m/s
        self.slowdown_angle = float(rospy.get_param("~slowdown_angle", 0.8))

        # If the sign feels wrong in your setup, flip one of these:
        self.invert_y_error = bool(rospy.get_param("~invert_y_error", False))
        self.invert_ang_error = bool(rospy.get_param("~invert_ang_error", False))

        # ---- Validity and search behavior ----
        self.min_pipe_area_px = int(rospy.get_param("~min_pipe_area_px", 1500))
        self.search_omega = float(
            rospy.get_param("~search_omega", 0.25)
        )  # when pipe not visible

        # ---- Debug ----
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.pub_debug = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        self.sub_mask = rospy.Subscriber(
            self.mask_topic, Image, self.cb_mask, queue_size=1, buff_size=2**24
        )

        rospy.loginfo(
            "[pipe_line_follower] started. Subscribing to %s, publishing %s",
            self.mask_topic,
            self.cmd_vel_topic,
        )

        # Enable/disable service (RENAMED to absolute and friendly path)
        # Was "~enable" -> now "/pipe_follower/enable"
        self.enabled = True
        self.srv_enable = rospy.Service(
            "/pipe_follower/enable",
            SetBool,
            self.cb_enable_service,
        )

    def cb_enable_service(self, req):
        self.enabled = bool(req.data)
        rospy.loginfo("[pipe_line_follower] enabled set to: %s", self.enabled)
        return SetBoolResponse(success=True, message=f"Enabled: {self.enabled}")

    # ---------- Core callback ----------
    def cb_mask(self, msg):
        rospy.loginfo("[pipe_line_follower] got mask image")
        if not self.enabled:
            return
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("cv_bridge err: %s", e)
            return

        # Bring into assumed orientation
        if self.rotate_cw:
            mask = rot90n(mask, self.rotate_cw)
        if self.flip_x:
            mask = np.ascontiguousarray(np.fliplr(mask))
        if self.flip_y:
            mask = np.ascontiguousarray(np.flipud(mask))

        h, w = mask.shape[:2]

        # ---------- Validity check ----------
        pipe_area = int(np.count_nonzero(mask))  # white pixels
        have_pipe = pipe_area >= self.min_pipe_area_px

        if not have_pipe:
            # Simple search: rotate to reacquire
            tw = Twist()
            tw.angular.z = saturate(self.search_omega, -self.ang_limit, self.ang_limit)
            tw.linear.x = 0.0
            self.pub_cmd.publish(tw)
            if self.publish_debug:
                self._publish_debug(mask, None, None, None, note="SEARCH")
            return

        # ---------- A) Orientation control via fitLine ----------
        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(_ret) == 3:
            _img_out, contours, _hier = _ret
        else:
            contours, _hier = _ret
        contours = list(contours) if contours is not None else []
        if len(contours) == 0:
            tw = Twist()
            tw.angular.z = saturate(self.search_omega, -self.ang_limit, self.ang_limit)
            tw.linear.x = 0.0
            self.pub_cmd.publish(tw)
            if self.publish_debug:
                self._publish_debug(mask, None, None, None, note="SEARCH(no-contours)")
            return
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pts = contours[0].reshape(-1, 2)

        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = math.atan2(float(vy), float(vx))  # radians; 0 means horizontal left→right
        # Map to [-pi/2, +pi/2] to avoid flipping
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        ang_err = angle  # target=0 (horizontal)
        if self.invert_ang_error:
            ang_err = -ang_err

        # ---------- B) Pseudo-sensor alignment at a front gate ----------
        x_center = int(saturate(self.sensor_x_frac, 0.0, 1.0) * w)
        half_gate = int(0.5 * saturate(self.sensor_width_frac, 0.01, 0.9) * w)
        xs = max(0, x_center - half_gate)
        xe = min(w, x_center + half_gate)
        gate = mask[:, xs:xe]

        # Split gate into N horizontal bands; compute occupancy per band
        n = max(3, int(self.n_sensors))
        band_h = h // n
        occ = []
        centers_y = []
        for i in range(n):
            ys = i * band_h
            ye = h if i == n - 1 else (i + 1) * band_h
            roi = gate[ys:ye, :]
            fill = float(np.count_nonzero(roi)) / float(roi.size)
            occ.append(fill)
            centers_y.append(0.5 * (ys + ye))
        occ = np.array(occ, dtype=np.float32)
        centers_y = np.array(centers_y, dtype=np.float32)

        # Ignore bands with almost no signal
        mask_valid = occ >= self.sensor_min_fill
        if np.any(mask_valid):
            y_meas = np.average(centers_y[mask_valid], weights=occ[mask_valid])
        else:
            # Fallback to global centroid if gate is empty
            m = cv2.moments(mask, binaryImage=True)
            if m["m00"] > 1e-3:
                y_meas = m["m01"] / m["m00"]
            else:
                y_meas = h * 0.5

        y_err = (y_meas - (0.5 * h)) / (0.5 * h)  # [-1, +1]; + if measured below center
        if self.invert_y_error:
            y_err = -y_err

        # ---------- Command synthesis ----------
        use_linear_y = rospy.get_param("~use_linear_y", False)
        tw = Twist()
        if use_linear_y:
            # Use linear.y for lateral correction, angular.z for orientation only
            w_z = -self.k_ang * ang_err
            w_z = saturate(w_z, -self.ang_limit, self.ang_limit)
            v_y = saturate(self.k_y * y_err, -self.lin_limit, self.lin_limit)
            tw.linear.y = v_y
            tw.angular.z = w_z
        else:
            # Default: combine both in angular.z
            w_z = -self.k_ang * ang_err - self.k_y * y_err
            w_z = saturate(w_z, -self.ang_limit, self.ang_limit)
            tw.angular.z = w_z
        # Slow down when misaligned by angle
        speed_scale = 1.0 / (1.0 + self.slowdown_angle * abs(ang_err))
        v_x = saturate(self.v_fwd * speed_scale, self.v_min, self.lin_limit)
        tw.linear.x = v_x
        self.pub_cmd.publish(tw)

        if self.publish_debug:
            debug_info = {
                "x_center": x_center,
                "xs": xs,
                "xe": xe,
                "y_meas": float(y_meas),
                "ang_err": float(ang_err),
                "y_err": float(y_err),
                "occ": occ.tolist(),
                "use_linear_y": use_linear_y,
            }
            self._publish_debug(mask, (vx, vy, x0, y0), (xs, xe), y_meas, debug_info)

    # ---------- Debug draw ----------
    def _publish_debug(self, mask, fit, gate_xsxe, y_meas, extra=None, note=None):
        h, w = mask.shape[:2]
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Gate band
        if gate_xsxe is not None:
            xs, xe = gate_xsxe
            cv2.rectangle(vis, (xs, 0), (xe, h - 1), (0, 255, 255), 1)
        # Fitted line
        if fit is not None:
            vx, vy, x0, y0 = fit
            x0, y0, vx, vy = float(x0), float(y0), float(vx), float(vy)

            def line_point(t):
                return int(x0 + t * vx), int(y0 + t * vy)

            p1 = line_point(-2000)
            p2 = line_point(+2000)
            cv2.line(vis, p1, p2, (0, 0, 255), 2)
        # Desired center line
        cv2.line(vis, (0, int(h / 2)), (w - 1, int(h / 2)), (255, 0, 0), 1)
        # Measured y
        if y_meas is not None:
            y = int(y_meas)
            cv2.line(vis, (0, y), (w - 1, y), (0, 255, 0), 1)
        # Note / text
        if note:
            cv2.putText(
                vis,
                str(note),
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        if isinstance(extra, dict):
            txt = "ang_err={:+.2f}  y_err={:+.2f}".format(
                extra.get("ang_err", 0.0), extra.get("y_err", 0.0)
            )
            cv2.putText(
                vis,
                txt,
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 220, 50),
                2,
                cv2.LINE_AA,
            )
        try:
            self.pub_debug.publish(CvBridge().cv2_to_imgmsg(vis, encoding="bgr8"))
        except Exception:
            pass


def main():
    rospy.init_node("pipe_line_follower")
    PipeLineFollower()
    rospy.spin()


if __name__ == "__main__":
    main()
