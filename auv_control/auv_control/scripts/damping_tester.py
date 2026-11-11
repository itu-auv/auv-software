#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from auv_control.control_clients import (
    ControlClients,
    ControllerParameters,
    log,
    parameter,
    init_ros,
)
import message_filters
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry

import tf_transformations

from std_srvs.srv import Trigger


def TriggerResponse(success, message):
    return Trigger.Response(success=success, message=message)


from auv_msgs.srv import SetDampingTarget


def SetDampingTargetResponse(success, message):
    return SetDampingTarget.Response(success=success, message=message)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def date_str() -> str:
    return datetime.now().strftime("%Y%m%d")


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
    phase_start: Time
    csv_path: str


class DampingStepWrenchLogger(Node):
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
        super().__init__("damping_step_wrench_logger")
        self._control_clients = ControlClients(self.get_namespace())
        log(self, "info", "damping_step_wrench_logger started")

        self.cmd_wrench_topic = parameter(self, "~cmd_wrench_topic", "cmd_wrench")
        self.odom_topic = parameter(self, "~odom_topic", "odometry")
        self.wrench_topic = parameter(self, "~wrench_topic", "wrench")

        self.pre_hold_time = float(parameter(self, "~pre_hold_time", 4.0))
        self.settle_time = float(parameter(self, "~settle_time", 5.0))
        self.record_time = float(parameter(self, "~record_time", 10.0))

        self.cmd_rate = float(parameter(self, "~cmd_rate", 50.0))
        self.log_rate = float(parameter(self, "~log_rate", 50.0))

        self.stabilize_time = float(parameter(self, "~stabilize_time", 4.0))

        log_dir = os.path.expanduser(parameter(self, "~log_dir", "~/damping_tests"))
        if bool(parameter(self, "~date_subdir", True)):
            log_dir = os.path.join(log_dir, date_str())
        self.log_dir = log_dir
        self.file_prefix = parameter(self, "~file_prefix", "damping_step")
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_only_record = bool(parameter(self, "~log_only_record", True))
        self.controller_reconfigure_server = parameter(
            self, "controller_reconfigure_server", "auv_control_node"
        )

        self._lock = threading.RLock()
        self.active: Optional[ActiveTest] = None
        self.stop_requested = False

        self.sequence_targets = []
        self.sequence_axis = ""
        self.sequence_idx = 0
        self.sequence_state = "IDLE"  # IDLE, STABILIZING, TESTING
        self.sequence_timer = self.get_clock().now()

        self.last_odom: Optional[Odometry] = None
        self.last_wrench: Optional[WrenchStamped] = None
        self.curr_cmd = WrenchStamped()

        self.csv_file = None
        self.csv_writer = None
        self.reconfigure_client = None
        self.saved_pid_gains = None
        self.saved_pid_axis = None

        self.cmd_wrench_pub = self.create_publisher(
            WrenchStamped, self.cmd_wrench_topic, 10
        )

        self.odom_sub = message_filters.Subscriber(self, Odometry, self.odom_topic)
        self.wrench_sub = message_filters.Subscriber(
            self, WrenchStamped, self.wrench_topic
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.wrench_sub], queue_size=50, slop=0.05
        )
        self.ts.registerCallback(self._sync_cb)

        self._reset_odom_srv = lambda: self._control_clients.service(
            Trigger, self.resolve_service_name("reset_odometry"), Trigger.Request()
        )

        self.create_service(
            SetDampingTarget, "~/surge", lambda req, res: self._srv_surge(req)
        )  # linear.x
        self.create_service(
            SetDampingTarget, "~/sway", lambda req, res: self._srv_sway(req)
        )  # linear.y
        self.create_service(
            SetDampingTarget, "~/yaw", lambda req, res: self._srv_yaw(req)
        )  # angular.z
        log(self, "info", "Services: ~surge ~sway ~yaw  (enable, target)")

        self.create_service(
            Trigger, "~/surge_seq", lambda req, res: self._srv_surge_seq(req)
        )
        self.create_service(
            Trigger, "~/sway_seq", lambda req, res: self._srv_sway_seq(req)
        )
        self.create_service(
            Trigger, "~/yaw_seq", lambda req, res: self._srv_yaw_seq(req)
        )
        self.create_service(
            Trigger, "~/stop_seq", lambda req, res: self._srv_stop_seq(req)
        )
        log(self, "info", "Sequence Services: ~surge_seq ~sway_seq ~yaw_seq ~stop_seq")

        self.create_timer(1.0 / self.cmd_rate, self._cmd_timer_cb)

    def _sync_cb(self, odom_msg: Odometry, wrench_msg: WrenchStamped):
        with self._lock:
            self.last_odom = odom_msg
            self.last_wrench = wrench_msg
            self._try_log_data(odom_msg, wrench_msg)

    def _srv_surge(self, req):
        return self._handle(axis_name="surge", enable=req.enable, target=req.target)

    def _srv_sway(self, req):
        return self._handle(axis_name="sway", enable=req.enable, target=req.target)

    def _srv_yaw(self, req):
        return self._handle(axis_name="yaw", enable=req.enable, target=req.target)

    def _srv_surge_seq(self, req):
        return self._handle_seq("surge")

    def _srv_sway_seq(self, req):
        return self._handle_seq("sway")

    def _srv_yaw_seq(self, req):
        return self._handle_seq("yaw")

    def _srv_stop_seq(self, req):
        with self._lock:
            if self.sequence_state != "IDLE":
                log(self, "info", "Stopping sequence...")
                self.sequence_state = "IDLE"
            if self.active is not None:
                self.stop_requested = True
            return TriggerResponse(True, "Stopped sequence and active tests.")

    def _handle_seq(self, axis_name: str) -> TriggerResponse:
        with self._lock:
            if self.sequence_state != "IDLE" or self.active is not None:
                return TriggerResponse(
                    False, "A test or sequence is already running. Stop it first."
                )

            # ADD PARAMETERS TO LAUNCH
            if axis_name == "yaw":
                default_targets = [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0]
            else:
                default_targets = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
            targets = parameter(self, f"~{axis_name}_targets", default_targets)

            full_seq = []
            for t in targets:
                full_seq.append(float(t))
                full_seq.append(-float(t))

            self.sequence_targets = full_seq
            self.sequence_axis = axis_name
            self.sequence_idx = 0

            # Start stabilization phase before triggering first test
            self.sequence_state = "STABILIZING"
            self.sequence_timer = self.get_clock().now()

            msg = f"Started '{axis_name}' sequence. Stabilizing for {self.stabilize_time}s..."
            log(self, "info", msg)
            return TriggerResponse(True, msg)

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
                phase_start=self.get_clock().now(),
                csv_path=csv_path,
            )
            self.stop_requested = False
            self.curr_cmd = WrenchStamped()

            msg = f"Started {axis_name} target={target}. CSV={csv_path}"
            log(self, "info", msg)
            return SetDampingTargetResponse(True, msg)

    # --- Timers
    def _cmd_timer_cb(self):
        with self._lock:
            # 1) SEQUENCE MANAGER
            if self.sequence_state == "STABILIZING":
                dt_stab = (
                    self.get_clock().now() - self.sequence_timer
                ).nanoseconds / 1e9
                if dt_stab >= self.stabilize_time:
                    if self.sequence_idx >= len(self.sequence_targets):
                        log(
                            self,
                            "info",
                            f"*** Sequence for {self.sequence_axis} COMPLETED ***",
                        )
                        self.sequence_state = "IDLE"
                    else:
                        target = self.sequence_targets[self.sequence_idx]
                        self.sequence_idx += 1
                        log(
                            self,
                            "info",
                            f"Sequence -> Starting {self.sequence_axis} target: {target}",
                        )

                        self._handle(self.sequence_axis, enable=True, target=target)
                        self.sequence_state = "TESTING"

            # 2) ACTIVE TEST MANAGER
            if self.active is None:
                return

            if self.stop_requested:
                self._finish_locked("STOP_REQUESTED")
                return

            t_now = self.get_clock().now()
            dt = (t_now - self.active.phase_start).nanoseconds / 1e9

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

        u = odom.twist.twist.linear.x
        v = odom.twist.twist.linear.y
        w = odom.twist.twist.linear.z
        p = odom.twist.twist.angular.x
        q = odom.twist.twist.angular.y
        r = odom.twist.twist.angular.z

        ori = odom.pose.pose.orientation
        quaternion = [ori.x, ori.y, ori.z, ori.w]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(quaternion)

        fx = wrench.wrench.force.x
        fy = wrench.wrench.force.y
        fz = wrench.wrench.force.z
        tx = wrench.wrench.torque.x
        ty = wrench.wrench.torque.y
        tz = wrench.wrench.torque.z

        log_time = Time.from_msg(odom.header.stamp).nanoseconds / 1e9

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
            roll,
            pitch,
            yaw,
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
            log(self, "error", f"CSV write failed: {e}")
            self._finish_locked("CSV_WRITE_ERROR")

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
            self.reconfigure_client = ControllerParameters(
                self._control_clients, self.controller_reconfigure_server, timeout=5
            )
            return True
        except Exception as exc:
            log(
                self,
                "warning",
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
            log(self, "warning", "Failed to read controller configuration: %s", exc)
            return False

        if not current_cfg:
            log(self, "warning", "Controller configuration is empty")
            return False

        gain_updates = {name: list(current_cfg[name]) for name in ("kp", "ki", "kd")}
        self.saved_pid_gains = {}
        self.saved_pid_axis = axis_name

        for name, values in gain_updates.items():
            self.saved_pid_gains[name] = {}
            for idx in self.AXIS_GAIN_INDICES[axis_name]:
                self.saved_pid_gains[name][idx] = values[idx]
                values[idx] = 0.0

        try:
            self.reconfigure_client.update_configuration(gain_updates)
        except Exception as exc:
            log(self, "warning", "Failed to zero PID gains for %s: %s", axis_name, exc)
            self.saved_pid_gains = None
            self.saved_pid_axis = None
            return False

        return True

    def _restore_pid_gains_locked(self):
        if self.reconfigure_client is None or self.saved_pid_gains is None:
            return

        try:
            self._reset_odom_srv()
        except Exception as exc:
            log(self, "warning", "Failed to call odometry reset service: %s", exc)

        try:
            current_cfg = self.reconfigure_client.get_configuration()
            gain_updates = {}
            for name, saved in self.saved_pid_gains.items():
                gain_updates[name] = list(current_cfg[name])
                for idx, value in saved.items():
                    gain_updates[name][idx] = value
            self.reconfigure_client.update_configuration(gain_updates)
        except Exception as exc:
            log(
                self,
                "warning",
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
        log(self, "info", f"Test finished: reason={reason} axis={axis} csv={path}")

        self.active = None
        self.stop_requested = False

        if self.sequence_state == "TESTING":
            if reason == "COMPLETED":
                log(
                    self,
                    "info",
                    f"Sequence -> Test completed. Stabilizing for {self.stabilize_time}s.",
                )
                self.sequence_state = "STABILIZING"
                self.sequence_timer = self.get_clock().now()
            else:
                log(self, "warning", f"Sequence aborted due to condition: {reason}")
                self.sequence_state = "IDLE"


def main():
    init_ros()
    node = DampingStepWrenchLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._restore_pid_gains()
        node._control_clients.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
