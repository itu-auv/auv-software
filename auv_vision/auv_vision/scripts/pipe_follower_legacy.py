#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class PipeFollowerLegacy:
    def __init__(self):
        self.k_p = rospy.get_param("~k_p", 1.5)
        self.max_angular_z = rospy.get_param("~max_angular_z", 0.8)

        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_debug = rospy.Publisher("debug_image", Image, queue_size=1)

        self.sub_mask = rospy.Subscriber("seg_mask", Image, self.cb_mask, queue_size=1)

    def cb_mask(self, msg):
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

            yaw_err = math.atan2(-vy, vx)

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

        tw.linear.x = 0.2

        self.pub_cmd.publish(tw)

        try:
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(debug_vis, encoding="bgr8")
            )
        except Exception:
            pass


def main():
    rospy.init_node("pipe_follower_legacy")
    PipeFollowerLegacy()
    rospy.spin()


if __name__ == "__main__":
    main()
