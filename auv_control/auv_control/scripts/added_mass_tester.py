#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import rospy
import message_filters
from geometry_msgs.msg import AccelWithCovarianceStamped, Twist, WrenchStamped
from nav_msgs.msg import Odometry

from auv_msgs.srv import SetAddedMassTarget, SetAddedMassTargetResponse


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
    amplitude: float  # A  (m/s or rad/s)
    frequency: float  # f  (Hz)
    phase: str  # PREHOLD / RECORD
    phase_start: rospy.Time
    csv_path: str


class AddedMassSineCmdVelLogger:
    """
    SINUSOIDAL test for added-mass identification:
      PREHOLD : cmd_vel=0 for pre_hold_time  (let vehicle settle)
      RECORD  : cmd_vel = A*sin(2πf·t) for record_time  (CSV log)
      DONE    : cmd_vel=0 and close CSV

    cmd_vel only (NO wrench publishing).
    Wrench is logged by subscribing to ~wrench_topic.
    """

    def __init__(self):
        rospy.init_node("added_mass_sine_cmdvel_logger", anonymous=False)
        rospy.loginfo("added_mass_sine_cmdvel_logger started")

        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "odometry")
        self.wrench_topic = rospy.get_param("~wrench_topic", "wrench")
        self.accel_topic = rospy.get_param("~accel_topic", "acceleration")

        self.pre_hold_time = float(rospy.get_param("~pre_hold_time", 2.0))
        self.record_time = float(rospy.get_param("~record_time", 20.0))

        self.cmd_rate = float(rospy.get_param("~cmd_rate", 50.0))

        self.log_dir = os.path.expanduser(
            rospy.get_param("~log_dir", "~/added_mass_tests")
        )
        self.file_prefix = rospy.get_param("~file_prefix", "added_mass_sine")
        os.makedirs(self.log_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.active: Optional[ActiveTest] = None
        self.stop_requested = False

        self.last_odom: Optional[Odometry] = None
        self.last_wrench: Optional[WrenchStamped] = None
        self.last_accel: Optional[AccelWithCovarianceStamped] = None
        self.curr_cmd = Twist()

        self.csv_file = None
        self.csv_writer = None

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        self.wrench_sub = message_filters.Subscriber(self.wrench_topic, WrenchStamped)
        self.accel_sub = message_filters.Subscriber(
            self.accel_topic, AccelWithCovarianceStamped
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.wrench_sub, self.accel_sub], queue_size=50, slop=0.05
        )
        self.ts.registerCallback(self._sync_cb)

        rospy.Service("~surge", SetAddedMassTarget, self._srv_surge)
        rospy.Service("~sway", SetAddedMassTarget, self._srv_sway)
        rospy.Service("~yaw", SetAddedMassTarget, self._srv_yaw)
        rospy.loginfo("Services: ~surge ~sway ~yaw  (enable, amplitude, frequency)")

        rospy.Timer(rospy.Duration(1.0 / self.cmd_rate), self._cmd_timer_cb)

    def _sync_cb(
        self,
        odom_msg: Odometry,
        wrench_msg: WrenchStamped,
        accel_msg: AccelWithCovarianceStamped,
    ):
        with self._lock:
            self.last_odom = odom_msg
            self.last_wrench = wrench_msg
            self.last_accel = accel_msg
            self._try_log_data(odom_msg, wrench_msg, accel_msg)

    def _srv_surge(self, req):
        return self._handle(
            axis_name="surge",
            enable=req.enable,
            amplitude=req.amplitude,
            frequency=req.frequency,
        )

    def _srv_sway(self, req):
        return self._handle(
            axis_name="sway",
            enable=req.enable,
            amplitude=req.amplitude,
            frequency=req.frequency,
        )

    def _srv_yaw(self, req):
        return self._handle(
            axis_name="yaw",
            enable=req.enable,
            amplitude=req.amplitude,
            frequency=req.frequency,
        )

    def _handle(self, axis_name: str, enable: bool, amplitude: float, frequency: float):
        with self._lock:
            if not enable:
                if self.active is None:
                    self.cmd_pub.publish(Twist())
                    return SetAddedMassTargetResponse(
                        True, "No active test. Published cmd_vel=0."
                    )
                self.stop_requested = True
                return SetAddedMassTargetResponse(
                    True, f"Stop requested for {self.active.axis_name}."
                )

            # start
            if self.active is not None:
                return SetAddedMassTargetResponse(
                    False, "A test is already running. Stop it first (enable=false)."
                )

            if frequency <= 0:
                return SetAddedMassTargetResponse(False, "frequency must be > 0.")
            if amplitude <= 0:
                return SetAddedMassTargetResponse(False, "amplitude must be > 0.")

            token = f"{axis_name}_A{target_token(amplitude)}_F{target_token(frequency)}"
            fname = f"{self.file_prefix}_{token}_{now_str()}.csv"
            csv_path = os.path.join(self.log_dir, fname)
            self._open_csv_locked(csv_path)

            self.active = ActiveTest(
                axis_name=axis_name,
                amplitude=float(amplitude),
                frequency=float(frequency),
                phase="PREHOLD",
                phase_start=rospy.Time.now(),
                csv_path=csv_path,
            )
            self.stop_requested = False
            self.curr_cmd = Twist()

            msg = (
                f"Started {axis_name} A={amplitude} f={frequency}Hz. " f"CSV={csv_path}"
            )
            rospy.loginfo(msg)
            return SetAddedMassTargetResponse(True, msg)

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
                self.curr_cmd = Twist()
                self.cmd_pub.publish(self.curr_cmd)

                if dt >= self.pre_hold_time:
                    self.active.phase = "RECORD"
                    self.active.phase_start = t_now
                return

            if self.active.phase == "RECORD":
                # t_rec: time elapsed since RECORD started
                t_rec = dt
                omega = 2.0 * math.pi * self.active.frequency
                value = self.active.amplitude * math.sin(omega * t_rec)

                self.curr_cmd = self._make_cmd(self.active.axis_name, value)
                self.cmd_pub.publish(self.curr_cmd)

                if t_rec >= self.record_time:
                    self._finish_locked("COMPLETED")
                return

            self._finish_locked("UNKNOWN_PHASE")

    def _try_log_data(
        self, odom: Odometry, wrench: WrenchStamped, accel: AccelWithCovarianceStamped
    ):
        """
        Called when synchronized data arrives.
        Writes to CSV if there is an active test in RECORD phase.
        """
        if self.active is None or self.csv_writer is None:
            return

        if self.active.phase != "RECORD":
            return

        # Compute analytical sinusoidal values at current time
        t_rec = (rospy.Time.now() - self.active.phase_start).to_sec()
        omega = 2.0 * math.pi * self.active.frequency
        cmd_value = self.active.amplitude * math.sin(omega * t_rec)
        cmd_accel = self.active.amplitude * omega * math.cos(omega * t_rec)

        # Odom velocities
        u = odom.twist.twist.linear.x
        v = odom.twist.twist.linear.y
        w = odom.twist.twist.linear.z
        p = odom.twist.twist.angular.x
        q = odom.twist.twist.angular.y
        r = odom.twist.twist.angular.z

        # Measured acceleration
        ax = accel.accel.accel.linear.x
        ay = accel.accel.accel.linear.y
        az = accel.accel.accel.linear.z

        # Wrench
        fx = wrench.wrench.force.x
        fy = wrench.wrench.force.y
        fz = wrench.wrench.force.z
        tx = wrench.wrench.torque.x
        ty = wrench.wrench.torque.y
        tz = wrench.wrench.torque.z

        log_time = odom.header.stamp.to_sec()

        row = [
            log_time,
            self.active.axis_name,
            self.active.phase,
            self.active.amplitude,
            self.active.frequency,
            t_rec,
            cmd_value,
            cmd_accel,
            self.curr_cmd.linear.x,
            self.curr_cmd.linear.y,
            self.curr_cmd.angular.z,
            u,
            v,
            w,
            p,
            q,
            r,
            ax,
            ay,
            az,
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
    def _make_cmd(self, axis_name: str, value: float) -> Twist:
        cmd = Twist()
        if axis_name == "surge":
            cmd.linear.x = value
        elif axis_name == "sway":
            cmd.linear.y = value
        elif axis_name == "yaw":
            cmd.angular.z = value
        return cmd

    def _open_csv_locked(self, csv_path: str):
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = [
            "t",
            "axis_name",
            "phase",
            "amplitude",
            "frequency",
            "t_rec",
            "cmd_value",
            "cmd_accel",
            "cmd_lin_x",
            "cmd_lin_y",
            "cmd_ang_z",
            "odom_lin_x",
            "odom_lin_y",
            "odom_lin_z",
            "odom_ang_x",
            "odom_ang_y",
            "odom_ang_z",
            "meas_accel_x",
            "meas_accel_y",
            "meas_accel_z",
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
        self.cmd_pub.publish(Twist())

        axis = self.active.axis_name if self.active else "none"
        path = self.active.csv_path if self.active else ""
        rospy.loginfo(f"Test finished: reason={reason} axis={axis} csv={path}")

        self._close_csv_locked()
        self.active = None
        self.stop_requested = False


if __name__ == "__main__":
    try:
        node = AddedMassSineCmdVelLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
