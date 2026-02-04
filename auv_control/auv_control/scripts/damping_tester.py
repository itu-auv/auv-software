#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import rospy
from geometry_msgs.msg import Twist, WrenchStamped
from nav_msgs.msg import Odometry

# >>> BURAYI kendi paket adınla değiştir <<<
from auv_msgs.srv import SetDampingTarget, SetDampingTargetResponse


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def target_token(x: float) -> str:
    """0.3 -> 0p300, -0.3 -> n0p300"""
    sign = "n" if x < 0 else ""
    ax = abs(x)
    return f"{sign}{ax:.3f}".replace(".", "p")


@dataclass
class ActiveTest:
    axis_name: str  # surge / sway / yaw
    target: float
    phase: str  # PREHOLD / SETTLE / RECORD
    phase_start: rospy.Time
    csv_path: str


class DampingStepCmdVelLogger:
    """
    STEP test:
      PREHOLD: cmd_vel=0 for pre_hold_time
      STEP -> SETTLE: cmd_vel=target for settle_time (hıza otursun)
      RECORD: cmd_vel=target for record_time (csv log)
      DONE: cmd_vel=0 and close csv

    cmd_vel only (NO wrench publishing).
    Wrench is logged by subscribing to ~wrench_topic.
    """

    def __init__(self):
        rospy.init_node("damping_step_cmdvel_logger", anonymous=False)
        rospy.loginfo("damping_step_cmdvel_logger started")

        # -------- Topics (placeholder, sen değiştireceksin)
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "odometry")
        # Bu topic MUTLAKA net body wrench (tau) yayınlamalı:
        self.wrench_topic = rospy.get_param("~wrench_topic", "wrench")

        # -------- Timing (STEP olduğundan settle'i uzun tutmak mantıklı)
        self.pre_hold_time = float(rospy.get_param("~pre_hold_time", 2.0))
        self.settle_time = float(rospy.get_param("~settle_time", 5.0))
        self.record_time = float(rospy.get_param("~record_time", 10.0))

        # -------- Rates
        self.cmd_rate = float(rospy.get_param("~cmd_rate", 50.0))
        self.log_rate = float(rospy.get_param("~log_rate", 50.0))

        # -------- Logging
        self.log_dir = os.path.expanduser(
            rospy.get_param("~log_dir", "~/damping_tests")
        )
        self.file_prefix = rospy.get_param("~file_prefix", "damping_step")
        os.makedirs(self.log_dir, exist_ok=True)

        # İstersen sadece RECORD fazını yaz:
        self.log_only_record = bool(rospy.get_param("~log_only_record", True))

        # -------- State
        self._lock = threading.Lock()
        self.active: Optional[ActiveTest] = None
        self.stop_requested = False

        self.last_odom: Optional[Odometry] = None
        self.last_wrench: Optional[WrenchStamped] = None
        self.curr_cmd = Twist()

        self.csv_file = None
        self.csv_writer = None

        # -------- Pub/Sub
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=50, tcp_nodelay=True
        )
        rospy.Subscriber(
            self.wrench_topic,
            WrenchStamped,
            self._wrench_cb,
            queue_size=50,
            tcp_nodelay=True,
        )

        # -------- Services
        rospy.Service("~surge", SetDampingTarget, self._srv_surge)  # linear.x
        rospy.Service("~sway", SetDampingTarget, self._srv_sway)  # linear.y
        rospy.Service("~yaw", SetDampingTarget, self._srv_yaw)  # angular.z
        rospy.loginfo("Services: ~surge ~sway ~yaw  (enable, target)")

        # -------- Timers
        rospy.Timer(rospy.Duration(1.0 / self.cmd_rate), self._cmd_timer_cb)
        rospy.Timer(rospy.Duration(1.0 / self.log_rate), self._log_timer_cb)

    # --- Subscribers
    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self.last_odom = msg

    def _wrench_cb(self, msg: WrenchStamped):
        with self._lock:
            self.last_wrench = msg

    # --- Services
    def _srv_surge(self, req):
        return self._handle(axis_name="surge", enable=req.enable, target=req.target)

    def _srv_sway(self, req):
        return self._handle(axis_name="sway", enable=req.enable, target=req.target)

    def _srv_yaw(self, req):
        return self._handle(axis_name="yaw", enable=req.enable, target=req.target)

    def _handle(self, axis_name: str, enable: bool, target: float):
        with self._lock:
            if not enable:
                # emergency stop
                if self.active is None:
                    self.cmd_pub.publish(Twist())
                    return SetDampingTargetResponse(
                        True, "No active test. Published cmd_vel=0."
                    )
                self.stop_requested = True
                return SetDampingTargetResponse(
                    True, f"Stop requested for {self.active.axis_name}."
                )

            # start
            if self.active is not None:
                return SetDampingTargetResponse(
                    False, "A test is already running. Stop it first (enable=false)."
                )

            token = target_token(target)
            fname = f"{self.file_prefix}_{axis_name}_{token}_{now_str()}.csv"
            csv_path = os.path.join(self.log_dir, fname)
            self._open_csv_locked(csv_path)

            self.active = ActiveTest(
                axis_name=axis_name,
                target=float(target),
                phase="PREHOLD",
                phase_start=rospy.Time.now(),
                csv_path=csv_path,
            )
            self.stop_requested = False
            self.curr_cmd = Twist()  # start 0

            msg = f"Started {axis_name} target={target}. CSV={csv_path}"
            rospy.loginfo(msg)
            return SetDampingTargetResponse(True, msg)

    # --- Timers
    def _cmd_timer_cb(self, _evt):
        with self._lock:
            if self.active is None:
                return

            if self.stop_requested:
                self._finish_locked("STOP_REQUESTED")
                return

            t_now = rospy.Time.now()
            dt = (t_now - self.active.phase_start).to_sec()

            if self.active.phase == "PREHOLD":
                # cmd_vel=0
                self.curr_cmd = Twist()
                self.cmd_pub.publish(self.curr_cmd)

                if dt >= self.pre_hold_time:
                    # STEP happens now: we switch to SETTLE and will publish target
                    self.active.phase = "SETTLE"
                    self.active.phase_start = t_now
                return

            if self.active.phase == "SETTLE":
                # cmd_vel=target (STEP already applied)
                self.curr_cmd = self._make_cmd(
                    self.active.axis_name, self.active.target
                )
                self.cmd_pub.publish(self.curr_cmd)

                if dt >= self.settle_time:
                    self.active.phase = "RECORD"
                    self.active.phase_start = t_now
                return

            if self.active.phase == "RECORD":
                self.curr_cmd = self._make_cmd(
                    self.active.axis_name, self.active.target
                )
                self.cmd_pub.publish(self.curr_cmd)

                if dt >= self.record_time:
                    self._finish_locked("COMPLETED")
                return

            self._finish_locked("UNKNOWN_PHASE")

    def _log_timer_cb(self, _evt):
        with self._lock:
            if self.active is None or self.csv_writer is None:
                return

            # sadece RECORD yazmak istersen:
            if self.log_only_record and self.active.phase != "RECORD":
                return

            if self.last_odom is None:
                return

            # measured velocities (placeholder: Odometry.twist)
            u = self.last_odom.twist.twist.linear.x
            v = self.last_odom.twist.twist.linear.y
            w = self.last_odom.twist.twist.linear.z
            p = self.last_odom.twist.twist.angular.x
            q = self.last_odom.twist.twist.angular.y
            r = self.last_odom.twist.twist.angular.z

            # wrench (must be provided by wrench_topic)
            fx = fy = fz = tx = ty = tz = float("nan")
            if self.last_wrench is not None:
                fx = self.last_wrench.wrench.force.x
                fy = self.last_wrench.wrench.force.y
                fz = self.last_wrench.wrench.force.z
                tx = self.last_wrench.wrench.torque.x
                ty = self.last_wrench.wrench.torque.y
                tz = self.last_wrench.wrench.torque.z
            else:
                rospy.logwarn_throttle(
                    2.0, "No wrench received yet -> wrench columns will be NaN!"
                )

            row = [
                rospy.Time.now().to_sec(),
                self.active.axis_name,
                self.active.phase,
                self.active.target,
                self.curr_cmd.linear.x,
                self.curr_cmd.linear.y,
                self.curr_cmd.angular.z,
                u,
                v,
                w,
                p,
                q,
                r,
                fx,
                fy,
                fz,
                tx,
                ty,
                tz,
            ]

            try:
                self.csv_writer.writerow(row)
                self.csv_file.flush()
            except Exception as e:
                rospy.logerr(f"CSV write failed: {e}")
                self._finish_locked("CSV_WRITE_ERROR")

    # --- helpers
    def _make_cmd(self, axis_name: str, target: float) -> Twist:
        cmd = Twist()
        if axis_name == "surge":
            cmd.linear.x = target
        elif axis_name == "sway":
            cmd.linear.y = target
        elif axis_name == "yaw":
            cmd.angular.z = target
        return cmd

    def _open_csv_locked(self, csv_path: str):
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = [
            "t",
            "axis_name",
            "phase",
            "target",
            "cmd_lin_x",
            "cmd_lin_y",
            "cmd_ang_z",
            "odom_lin_x",
            "odom_lin_y",
            "odom_lin_z",
            "odom_ang_x",
            "odom_ang_y",
            "odom_ang_z",
            "wrench_force_x",
            "wrench_force_y",
            "wrench_force_z",
            "wrench_torque_x",
            "wrench_torque_y",
            "wrench_torque_z",
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()

    def _close_csv_locked(self):
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        self.csv_file = None
        self.csv_writer = None

    def _finish_locked(self, reason: str):
        # cmd_vel=0
        self.cmd_pub.publish(Twist())

        axis = self.active.axis_name if self.active else "none"
        path = self.active.csv_path if self.active else ""
        rospy.loginfo(f"Test finished: reason={reason} axis={axis} csv={path}")

        self.active = None
        self.stop_requested = False


if __name__ == "__main__":
    try:
        node = DampingStepCmdVelLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
