#!/usr/bin/env python3
import rospy, cv2, numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger, SetBool, SetBoolResponse


class VisualServo:
    def __init__(self):
        self.bridge = CvBridge()

        self.max_lin = rospy.get_param("~max_lin", 0.8)
        self.max_ang = rospy.get_param("~max_ang", 1.2)

        self.enabled = False

        # PID gains
        p = rospy.get_param("~pid_yaw", {"kp": 0.35, "ki": 0.01, "kd": 0.04})
        self.kp, self.ki, self.kd = p["kp"], p["ki"], p["kd"]
        lp = rospy.get_param("~pid_lat", {"kp": 0.4, "ki": 0.0, "kd": 0.02})
        self.kp_lat, self.ki_lat, self.kd_lat = lp["kp"], lp["ki"], lp["kd"]

        self.integral = 0.0
        self.last_err = 0.0
        self.last_t = rospy.Time.now()

        self.bootstrap_secs = rospy.get_param("~bootstrap_secs", 5.0)
        self.started = rospy.Time.now()
        self.bootstrap_done = False

        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.mask_sub = rospy.Subscriber("pipe_mask", Image, self.cb_mask, queue_size=1)

        rospy.wait_for_service("/pipe_plan/commit_bootstrap")
        self.commit_bootstrap = rospy.ServiceProxy(
            "/pipe_plan/commit_bootstrap", Trigger
        )
        rospy.Service("/pipe_servo/enable", SetBool, self.srv_enable)

    def srv_enable(self, req):
        self.enabled = req.data
        if not self.enabled:
            self.vel_pub.publish(Twist())
        return SetBoolResponse(
            success=True, message="enabled" if self.enabled else "disabled"
        )

    def cb_mask(self, msg: Image):

        if not self.enabled:
            return

        now = rospy.Time.now()
        dt = (now - self.last_t).to_sec()
        self.last_t = now

        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        h, w = mask.shape[:2]

        # bottom 60% ROI
        roi = mask[int(0.4 * h) :, :]
        seg_h = roi.shape[0] // 5
        mids = []
        for i in range(5):
            seg = roi[i * seg_h : (i + 1) * seg_h, :]
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + i * seg_h
            mids.append((cx, cy))

        tw = Twist()
        tw.linear.x = 0.4 * self.max_lin
        tw.angular.z = 0.5 * self.max_ang

        if len(mids) >= 2:
            # position/lateral error
            pos_err_px = mids[-1][0] - (w // 2)

            # angle error: last two centroid vectors
            dx = float(mids[-1][0] - mids[-2][0])
            dy = float(mids[-1][1] - mids[-2][1] + 1e-6)
            ang_err = np.degrees(
                np.arctan2(dy, dx)
            )  # no correction needed for "up+" on screen

            # small lateral contribution (keep px scale weak)
            total_err = ang_err + 0.01 * pos_err_px

            self.integral += total_err * dt
            der = (total_err - self.last_err) / dt if dt > 0 else 0.0
            yaw = self.kp * total_err + self.ki * self.integral + self.kd * der
            self.last_err = total_err

            tw.linear.x = self.max_lin
            tw.angular.z = np.clip(yaw, -self.max_ang, self.max_ang)

        self.vel_pub.publish(tw)

        # When the bootstrap period is over (once)
        if (
            not self.bootstrap_done
            and (now - self.started).to_sec() >= self.bootstrap_secs
        ):
            try:
                _ = self.commit_bootstrap()
                self.bootstrap_done = True
                rospy.loginfo("Committed bootstrap frame.")
            except Exception as e:
                rospy.logerr("commit_bootstrap failed: %s", e)


if __name__ == "__main__":
    rospy.init_node("pipe_visual_servo_node")
    VisualServo()
    rospy.spin()
