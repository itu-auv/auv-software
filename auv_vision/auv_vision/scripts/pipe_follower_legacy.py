#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse


class PipeFollowerLegacy:
    def __init__(self):
        self.k_p = rospy.get_param("~k_p", 1.5)
        self.max_angular_z = rospy.get_param("~max_angular_z", 0.8)
        self.linear_x = rospy.get_param("~linear_x", 0.2)
        self.camera_forward_direction = rospy.get_param(
            "~camera_forward_direction", "right"
        ).lower()
        if self.camera_forward_direction not in ("right", "left", "up", "down"):
            rospy.logwarn(
                f"Invalid camera_forward_direction '{self.camera_forward_direction}', using 'right'"
            )
            self.camera_forward_direction = "right"
        self.is_enabled = False

        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_debug = None

        self.sub_mask = rospy.Subscriber("seg_mask", Image, self.cb_mask, queue_size=1)
        self.start_service = rospy.Service("~enable", Trigger, self.cb_start)
        self.stop_service = rospy.Service("~disable", Trigger, self.cb_stop)

    def cb_start(self, req):
        self.is_enabled = True
        if self.pub_debug is None:
            self.pub_debug = rospy.Publisher(
                "debug_image/compressed", CompressedImage, queue_size=1
            )
        return TriggerResponse(success=True, message="Pipe follower legacy enabled")

    def cb_stop(self, req):
        self.is_enabled = False
        self.pub_cmd.publish(Twist())
        if self.pub_debug is not None:
            self.pub_debug.unregister()
            self.pub_debug = None
        return TriggerResponse(success=True, message="Pipe follower legacy disabled")

    def cb_mask(self, msg):
        if not self.is_enabled:
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(_ret) == 3:
            _img_out, contours, _hier = _ret
        else:
            contours, _hier = _ret

        contours = list(contours) if contours is not None else []
        if len(contours) == 0:
            return
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        pts = largest_contour.reshape(-1, 2)

        tw = Twist()
        debug_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        h, w = mask.shape[:2]

        if pts is not None:
            line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line.flatten()

            forward_vectors = {
                "right": (1.0, 0.0),
                "left": (-1.0, 0.0),
                "up": (0.0, -1.0),
                "down": (0.0, 1.0),
            }
            fx, fy = forward_vectors[self.camera_forward_direction]
            if vx * fx + vy * fy < 0:
                vx = -vx
                vy = -vy

            line_angle = math.atan2(-vy, vx)
            forward_angle = math.atan2(-fy, fx)
            yaw_err = line_angle - forward_angle
            yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

            tw.angular.z = self.k_p * yaw_err

            tw.angular.z = max(
                -self.max_angular_z, min(self.max_angular_z, tw.angular.z)
            )

            t1 = -max(h, w)
            t2 = max(h, w)
            p1 = (int(x0 + t1 * vx), int(y0 + t1 * vy))
            p2 = (int(x0 + t2 * vx), int(y0 + t2 * vy))

            cv2.line(debug_vis, p1, p2, (0, 0, 255), 2)
            cv2.putText(
                debug_vis,
                f"yaw err: {yaw_err:.2f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            tw.angular.z = 0.0
            cv2.putText(
                debug_vis,
                "no pipe",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        tw.linear.x = self.linear_x

        self.pub_cmd.publish(tw)

        if self.pub_debug is not None:
            arrow_margin = 18
            arrow_len = 54
            arrow_x = max(arrow_margin, w - 45)
            arrow_y = 32
            if self.camera_forward_direction == "right":
                arrow_start = (
                    max(arrow_margin, w - arrow_len - arrow_margin),
                    arrow_y,
                )
                arrow_end = (w - arrow_margin, arrow_y)
            elif self.camera_forward_direction == "left":
                arrow_start = (w - arrow_margin, arrow_y)
                arrow_end = (
                    max(arrow_margin, w - arrow_len - arrow_margin),
                    arrow_y,
                )
            elif self.camera_forward_direction == "up":
                arrow_start = (
                    arrow_x,
                    min(h - arrow_margin, arrow_y + arrow_len // 2),
                )
                arrow_end = (arrow_x, arrow_margin)
            else:
                arrow_start = (arrow_x, arrow_margin)
                arrow_end = (
                    arrow_x,
                    min(h - arrow_margin, arrow_y + arrow_len // 2),
                )
            cv2.arrowedLine(
                debug_vis,
                arrow_start,
                arrow_end,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.35,
            )
            ok, encoded = cv2.imencode(
                ".jpg", debug_vis, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ok:
                debug_msg = CompressedImage()
                debug_msg.header = msg.header
                debug_msg.format = "jpeg"
                debug_msg.data = np.array(encoded).tobytes()
                self.pub_debug.publish(debug_msg)


def main():
    rospy.init_node("pipe_follower_legacy")
    PipeFollowerLegacy()
    rospy.spin()


if __name__ == "__main__":
    main()
