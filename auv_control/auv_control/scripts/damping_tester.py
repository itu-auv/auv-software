#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import dynamic_reconfigure.client
import rospy
import message_filters
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry

import tf.transformations

from std_srvs.srv import Trigger

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


class DampingStepWrenchLogger:
    """
    STEP test:
      PREHOLD: cmd_wrench=0 for pre_hold_time
      STEP -> SETTLE: cmd_wrench=target for settle_time
      RECORD: cmd_wrench=target for record_time (csv log)
      DONE: cmd_wrench=0 and close csv

    The published wrench is added at the end of the MultiDOF PID output.
    """

    AXIS_GAIN_INDICES = {
        "surge": (0, 6),
        "sway": (1, 7),
        "yaw": (5, 11),
    }

    def __init__(self):
        rospy.init_node("damping_step_wrench_logger", anonymous=False)
        rospy.loginfo("damping_step_wrench_logger started")

        # -------- Topics
        self.cmd_wrench_topic = rospy.get_param("~cmd_wrench_topic", "cmd_wrench")
        self.odom_topic = rospy.get_param("~odom_topic", "odometry")
        self.wrench_topic = rospy.get_param("~wrench_topic", "wrench")

        # -------- Timing
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

        self.log_only_record = bool(rospy.get_param("~log_only_record", True))
        self.controller_reconfigure_server = rospy.resolve_name(
            rospy.get_param("~controller_reconfigure_server", "auv_control_node")
        )

        # -------- State
        self._lock = threading.Lock()
        self.active: Optional[ActiveTest] = None
        self.stop_requested = False

        self.last_odom: Optional[Odometry] = None
        self.last_wrench: Optional[WrenchStamped] = None
        self.curr_cmd = WrenchStamped()

        self.csv_file = None
        self.csv_writer = None
        self.reconfigure_client = None
        self.saved_pid_gains = None
        self.saved_pid_axis = None

        # -------- Pub/Sub
        self.cmd_wrench_pub = rospy.Publisher(
            self.cmd_wrench_topic, WrenchStamped, queue_size=10
        )

        # Time Synchronization of Odom and Wrench with Message Filters
        self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        self.wrench_sub = message_filters.Subscriber(self.wrench_topic, WrenchStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.wrench_sub], queue_size=50, slop=0.05
        )
        self.ts.registerCallback(self._sync_cb)

        # -------- Services
        self._reset_odom_srv = rospy.ServiceProxy("reset_odometry", Trigger)

        rospy.Service("~surge", SetDampingTarget, self._srv_surge)  # linear.x
        rospy.Service("~sway", SetDampingTarget, self._srv_sway)  # linear.y
        rospy.Service("~yaw", SetDampingTarget, self._srv_yaw)  # angular.z
        rospy.loginfo("Services: ~surge ~sway ~yaw  (enable, target)")

        # -------- Timers
        rospy.Timer(rospy.Duration(1.0 / self.cmd_rate), self._cmd_timer_cb)
        rospy.on_shutdown(self._restore_pid_gains)

    # --- Subscribers
    def _sync_cb(self, odom_msg: Odometry, wrench_msg: WrenchStamped):
        with self._lock:
            self.last_odom = odom_msg
            self.last_wrench = wrench_msg
            self._try_log_data(odom_msg, wrench_msg)

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
                if self.active is None:
                    self.cmd_wrench_pub.publish(WrenchStamped())
                    self._restore_pid_gains_locked()
                    return SetDampingTargetResponse(
                        True, "No active test. Published cmd_wrench=0."
                    )
                self.stop_requested = True
                return SetDampingTargetResponse(
                    True, f"Stop requested for {self.active.axis_name}."
                )

            if self.active is not None:
                return SetDampingTargetResponse(
                    False, "A test is already running. Stop it first (enable=false)."
                )

            if not self._zero_pid_gains_for_axis_locked(axis_name):
                return SetDampingTargetResponse(
                    False, f"Failed to zero PID gains for axis {axis_name}."
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
            self.curr_cmd = WrenchStamped()

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
                self.curr_cmd = WrenchStamped()
                self.cmd_wrench_pub.publish(self.curr_cmd)

                if dt >= self.pre_hold_time:
                    self.active.phase = "SETTLE"
                    self.active.phase_start = t_now
                return

            if self.active.phase == "SETTLE":
                self.curr_cmd = self._make_cmd(
                    self.active.axis_name, self.active.target
                )
                self.cmd_wrench_pub.publish(self.curr_cmd)

                if dt >= self.settle_time:
                    self.active.phase = "RECORD"
                    self.active.phase_start = t_now
                return

            if self.active.phase == "RECORD":
                self.curr_cmd = self._make_cmd(
                    self.active.axis_name, self.active.target
                )
                self.cmd_wrench_pub.publish(self.curr_cmd)

                if dt >= self.record_time:
                    self._finish_locked("COMPLETED")
                return

            self._finish_locked("UNKNOWN_PHASE")

    def _try_log_data(self, odom: Odometry, wrench: WrenchStamped):
        if self.active is None or self.csv_writer is None:
            return

        if self.log_only_record and self.active.phase != "RECORD":
            return

        # Lineer ve açısal hızları al
        u = odom.twist.twist.linear.x
        v = odom.twist.twist.linear.y
        w = odom.twist.twist.linear.z
        p = odom.twist.twist.angular.x
        q = odom.twist.twist.angular.y
        r = odom.twist.twist.angular.z

        # Quaternion'dan Euler açılarına (Roll, Pitch, Yaw) dönüşüm
        ori = odom.pose.pose.orientation
        quaternion = [ori.x, ori.y, ori.z, ori.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)

        # Kuvvet ve Tork verilerini al
        fx = wrench.wrench.force.x
        fy = wrench.wrench.force.y
        fz = wrench.wrench.force.z
        tx = wrench.wrench.torque.x
        ty = wrench.wrench.torque.y
        tz = wrench.wrench.torque.z

        log_time = odom.header.stamp.to_sec()

        # Row listesine Euler açılarını (roll, pitch, yaw) da ekliyoruz
        row = [
            log_time,
            self.active.axis_name,
            self.active.phase,
            self.active.target,
            self.curr_cmd.wrench.force.x,
            self.curr_cmd.wrench.force.y,
            self.curr_cmd.wrench.torque.z,
            u,
            v,
            w,
            p,
            q,
            r,
            roll,  # EKLENDİ
            pitch,  # EKLENDİ
            yaw,  # EKLENDİ
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
    def _make_cmd(self, axis_name: str, target: float) -> WrenchStamped:
        cmd = WrenchStamped()
        if axis_name == "surge":
            cmd.wrench.force.x = target
        elif axis_name == "sway":
            cmd.wrench.force.y = target
        elif axis_name == "yaw":
            cmd.wrench.torque.z = target
        return cmd

    def _ensure_reconfigure_client_locked(self) -> bool:
        if self.reconfigure_client is not None:
            return True

        try:
            self.reconfigure_client = dynamic_reconfigure.client.Client(
                self.controller_reconfigure_server, timeout=5
            )
            return True
        except Exception as exc:
            rospy.logwarn(
                "Failed to connect to controller reconfigure server %s: %s",
                self.controller_reconfigure_server,
                exc,
            )
            return False

    def _zero_pid_gains_for_axis_locked(self, axis_name: str) -> bool:
        if self.saved_pid_gains is not None:
            return True

        if not self._ensure_reconfigure_client_locked():
            return False

        try:
            current_cfg = self.reconfigure_client.get_configuration()
        except Exception as exc:
            rospy.logwarn("Failed to read controller configuration: %s", exc)
            return False

        if not current_cfg:
            rospy.logwarn("Controller configuration is empty")
            return False

        gain_updates = {}
        self.saved_pid_gains = {}
        self.saved_pid_axis = axis_name

        for idx in self.AXIS_GAIN_INDICES[axis_name]:
            for prefix in ("kp", "ki", "kd"):
                key = f"{prefix}_{idx}"
                self.saved_pid_gains[key] = current_cfg.get(key, 0.0)
                gain_updates[key] = 0.0

        try:
            self.reconfigure_client.update_configuration(gain_updates)
        except Exception as exc:
            rospy.logwarn("Failed to zero PID gains for %s: %s", axis_name, exc)
            self.saved_pid_gains = None
            self.saved_pid_axis = None
            return False

        return True

    def _restore_pid_gains_locked(self):
        if self.reconfigure_client is None or self.saved_pid_gains is None:
            return

        try:
            self._reset_odom_srv()
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to call odometry reset service: %s", exc)

        try:
            self.reconfigure_client.update_configuration(self.saved_pid_gains)
        except Exception as exc:
            rospy.logwarn(
                "Failed to restore PID gains for %s: %s",
                self.saved_pid_axis,
                exc,
            )
            return

        self.saved_pid_gains = None
        self.saved_pid_axis = None

    def _restore_pid_gains(self):
        with self._lock:
            self._restore_pid_gains_locked()

    def _open_csv_locked(self, csv_path: str):
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = [
            "t",
            "axis_name",
            "phase",
            "target",
            "cmd_force_x",
            "cmd_force_y",
            "cmd_torque_z",
            "odom_lin_x",
            "odom_lin_y",
            "odom_lin_z",
            "odom_ang_x",
            "odom_ang_y",
            "odom_ang_z",
            "roll",
            "pitch",
            "yaw",
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
        self.cmd_wrench_pub.publish(WrenchStamped())
        self._restore_pid_gains_locked()
        self._close_csv_locked()

        axis = self.active.axis_name if self.active else "none"
        path = self.active.csv_path if self.active else ""
        rospy.loginfo(f"Test finished: reason={reason} axis={axis} csv={path}")

        self.active = None
        self.stop_requested = False


if __name__ == "__main__":
    try:
        node = DampingStepWrenchLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
