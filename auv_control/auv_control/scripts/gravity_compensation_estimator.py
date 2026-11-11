#!/usr/bin/env python3

import json
import math
from collections import deque
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from auv_control.control_clients import (
    ControlClients,
    ControllerParameters,
    log,
    log_throttle,
    parameter,
    init_ros,
)
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import SetBool

SetBoolResponse = SetBool.Response


class GravityCompensationEstimatorNode(Node):
    def __init__(self):
        super().__init__("gravity_compensation_estimator")
        self._control_clients = ControlClients(self.get_namespace())
        self.lock = Lock()

        self.controller_reconfigure_server = parameter(
            self, "controller_reconfigure_server", "auv_control_node"
        )
        self.status_rate = float(parameter(self, "~status_rate", 1.0))

        self.active_min_z = float(parameter(self, "~active_min_z", -1.60))
        self.active_max_z = float(parameter(self, "~active_max_z", -0.35))

        self.stable_window = Duration(
            seconds=float(parameter(self, "~stable_window", 5.0))
        )
        self.sample_window_slack = Duration(
            seconds=float(parameter(self, "~sample_window_slack", 1.0))
        )
        self.min_samples = int(parameter(self, "~min_samples", 80))
        self.z_stability_threshold = float(
            parameter(self, "~z_stability_threshold", 0.03)
        )
        self.max_vertical_velocity = float(
            parameter(self, "~max_vertical_velocity", 0.01)
        )
        self.max_wrench_z_stddev = float(parameter(self, "~max_wrench_z_stddev", 3.0))
        self.max_horizontal_velocity = float(
            parameter(self, "~max_horizontal_velocity", 0.05)
        )
        self.max_horizontal_wrench = float(
            parameter(self, "~max_horizontal_wrench", 5.0)
        )
        self.use_odometry_twist = self._get_bool_param("~use_odometry_twist", False)

        self.min_update_interval = Duration(
            seconds=float(parameter(self, "~min_update_interval", 2.0))
        )
        self.update_alpha = float(parameter(self, "~update_alpha", 0.8))
        self.max_update_step = float(parameter(self, "~max_update_step", 5.0))
        self.update_deadband = float(parameter(self, "~update_deadband", 0.2))
        self.min_gravity_compensation_z = float(
            parameter(self, "~min_gravity_compensation_z", -20.0)
        )
        self.max_gravity_compensation_z = float(
            parameter(self, "~max_gravity_compensation_z", 20.0)
        )
        self.apply_updates = True

        self.data_timeout = Duration(seconds=1.0)
        self.samples = deque()
        self.latest_odometry = None
        self.latest_odometry_time = Time(clock_type=self.get_clock().clock_type)
        self.latest_wrench_time = Time(clock_type=self.get_clock().clock_type)
        self.latest_enable_time = Time(clock_type=self.get_clock().clock_type)
        self.control_enabled = False
        self.last_update_time = Time(clock_type=self.get_clock().clock_type)
        self.reconfigure_client = None
        self.current_gravity_compensation_z = None
        self.estimated_gravity_compensation_z = None
        self.estimated_wrench_z_stddev = None
        self.status_reason = "waiting_for_data"

        self.odometry_sub = self.create_subscription(
            Odometry, "odometry", self.odometry_callback, 1
        )
        self.wrench_sub = self.create_subscription(
            WrenchStamped, "wrench", self.wrench_callback, 1
        )
        self.enable_sub = self.create_subscription(
            Bool, "enable", self.enable_callback, 1
        )
        self.status_pub = self.create_publisher(
            String, "gravity_compensation_estimator/status", 1
        )
        self.estimate_pub = self.create_publisher(
            Float64, "gravity_compensation_estimator/estimated_z", 1
        )
        self.current_pub = self.create_publisher(
            Float64, "gravity_compensation_estimator/current_z", 1
        )
        self.set_apply_updates_srv = self.create_service(
            SetBool,
            "~/set_apply_updates",
            lambda req, res: self.set_apply_updates_callback(req),
        )

        self.status_timer = self.create_timer(
            1.0 / max(self.status_rate, 0.1), self.status_timer_callback
        )

        log(
            self,
            "info",
            "Gravity compensation estimator started: active z range [%.3f, %.3f], "
            "controller reconfigure server %s",
            self.active_min_z,
            self.active_max_z,
            self.controller_reconfigure_server,
        )

    def odometry_callback(self, msg: Odometry) -> None:
        stamp = self._message_time(msg.header.stamp)
        with self.lock:
            self.latest_odometry = msg
            self.latest_odometry_time = stamp

    def enable_callback(self, msg: Bool) -> None:
        with self.lock:
            self.control_enabled = msg.data
            self.latest_enable_time = self.get_clock().now()
            if not self.control_enabled:
                self._reject_locked("control_disabled")

    def set_apply_updates_callback(self, req: SetBool) -> SetBoolResponse:
        with self.lock:
            self.apply_updates = req.data
            state = "enabled" if self.apply_updates else "disabled"
            if self.apply_updates:
                self.samples.clear()
                self.status_reason = "collecting_samples"
            else:
                self.samples.clear()
                self.status_reason = "updates_disabled"

        return SetBoolResponse(
            success=True,
            message=f"Gravity compensation automatic updates {state}",
        )

    def wrench_callback(self, msg: WrenchStamped) -> None:
        now = self._message_time(msg.header.stamp)
        with self.lock:
            self.latest_wrench_time = now
            sample = self._make_sample_locked(msg, now)
            if sample is None:
                return

            self.samples.append(sample)
            self._prune_samples_locked(now)
            self._maybe_update_locked(now)

    def status_timer_callback(self) -> None:
        with self.lock:
            now = self.get_clock().now()
            if self.latest_wrench_time.nanoseconds == 0:
                if not self.samples:
                    self.status_reason = "waiting_for_wrench"
            elif not self._is_recent(self.latest_wrench_time, now, self.data_timeout):
                self._reject_locked("wrench_stale")
            self._publish_status_locked(now)

    def _make_sample_locked(self, msg: WrenchStamped, now: Time):
        if self.latest_odometry is None:
            self._reject_locked("waiting_for_odometry")
            return None

        if not self._is_recent(self.latest_odometry_time, now, self.data_timeout):
            self._reject_locked("odometry_stale")
            return None

        if not self.control_enabled:
            self._reject_locked("control_disabled")
            return None
        if not self._is_recent(self.latest_enable_time, now, self.data_timeout):
            self._reject_locked("enable_stale")
            return None

        odom = self.latest_odometry
        z = odom.pose.pose.position.z
        if z < self.active_min_z or z > self.active_max_z:
            self._reject_locked("depth_out_of_range")
            return None

        body_force = (msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)
        world_force = self._rotate_body_to_world(odom.pose.pose.orientation, body_force)
        body_velocity = (
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        )
        world_velocity = self._rotate_body_to_world(
            odom.pose.pose.orientation, body_velocity
        )
        if world_force is None or world_velocity is None:
            self._reject_locked("invalid_odometry_orientation")
            return None

        if self.use_odometry_twist:
            horizontal_velocity = math.sqrt(
                world_velocity[0] ** 2 + world_velocity[1] ** 2
            )
            if horizontal_velocity > self.max_horizontal_velocity:
                self._reject_locked("horizontal_motion")
                return None
            sample_horizontal_velocity = horizontal_velocity
            sample_horizontal_wrench = 0.0
        else:
            horizontal_wrench = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2)
            if horizontal_wrench > self.max_horizontal_wrench:
                self._reject_locked("horizontal_wrench_active")
                return None
            sample_horizontal_velocity = 0.0
            sample_horizontal_wrench = horizontal_wrench

        self.status_reason = "collecting_samples"
        return {
            "stamp": now,
            "z": z,
            "vertical_velocity": world_velocity[2],
            "horizontal_velocity": sample_horizontal_velocity,
            "horizontal_wrench": sample_horizontal_wrench,
            "world_wrench_z": world_force[2],
        }

    def _maybe_update_locked(self, now: Time) -> None:
        if len(self.samples) < self.min_samples:
            self.status_reason = "collecting_samples"
            return

        window_duration = self.samples[-1]["stamp"] - self.samples[0]["stamp"]
        if window_duration < self.stable_window:
            self.status_reason = "collecting_samples"
            return

        z_values = [sample["z"] for sample in self.samples]
        z_range = max(z_values) - min(z_values)
        if z_range > self.z_stability_threshold:
            self.status_reason = "depth_not_stable"
            return

        max_abs_vertical_velocity = max(
            abs(sample["vertical_velocity"]) for sample in self.samples
        )
        if max_abs_vertical_velocity > self.max_vertical_velocity:
            self.status_reason = "vertical_velocity_not_stable"
            return

        wrench_z_values = [sample["world_wrench_z"] for sample in self.samples]
        estimated_wrench_z = sum(wrench_z_values) / len(wrench_z_values)
        wrench_z_stddev = self._stddev(wrench_z_values, estimated_wrench_z)
        self.estimated_gravity_compensation_z = estimated_wrench_z
        self.estimated_wrench_z_stddev = wrench_z_stddev

        if (
            self.max_wrench_z_stddev > 0.0
            and wrench_z_stddev > self.max_wrench_z_stddev
        ):
            self.status_reason = "wrench_z_not_stable"
            return

        if (
            self.last_update_time.nanoseconds != 0
            and now - self.last_update_time < self.min_update_interval
        ):
            self.status_reason = "cooldown"
            return

        current = self._read_current_compensation_locked()
        if current is None:
            self.status_reason = "reconfigure_unavailable"
            return

        error = estimated_wrench_z - current
        if abs(error) <= self.update_deadband:
            self.status_reason = "within_deadband"
            return

        step = self._clamp(
            self.update_alpha * error, -self.max_update_step, self.max_update_step
        )
        new_value = self._clamp(
            current + step,
            self.min_gravity_compensation_z,
            self.max_gravity_compensation_z,
        )

        if abs(new_value - current) <= 1e-6:
            self.status_reason = "at_limit"
            return

        if not self.apply_updates:
            self.status_reason = "dry_run_update_ready"
            log_throttle(
                self,
                "info",
                5.0,
                "Gravity compensation dry-run update ready: current %.3f, "
                "estimated %.3f, proposed %.3f",
                current,
                estimated_wrench_z,
                new_value,
            )
            return

        if self._write_current_compensation_locked(new_value):
            self.last_update_time = now
            self.samples.clear()
            self.status_reason = "updated"
            log(
                self,
                "info",
                "Updated gravity_compensation_z: %.3f -> %.3f "
                "(estimated %.3f, stddev %.3f, z range %.3f)",
                current,
                new_value,
                estimated_wrench_z,
                wrench_z_stddev,
                z_range,
            )
        else:
            self.status_reason = "update_failed"

    def _ensure_reconfigure_client_locked(self) -> bool:
        if self.reconfigure_client is not None:
            return True

        try:
            self.reconfigure_client = ControllerParameters(
                self._control_clients, self.controller_reconfigure_server, timeout=5
            )
            log(
                self,
                "info",
                "Connected to dynamic reconfigure server: %s",
                self.controller_reconfigure_server,
            )
            return True
        except Exception as exc:
            log_throttle(
                self,
                "warning",
                5.0,
                "Failed to connect to controller reconfigure server %s: %s",
                self.controller_reconfigure_server,
                exc,
            )
            return False

    def _read_current_compensation_locked(self):
        if not self._ensure_reconfigure_client_locked():
            return None

        try:
            config = self.reconfigure_client.get_configuration()
            self.current_gravity_compensation_z = float(
                config["gravity_compensation_z"]
            )
            return self.current_gravity_compensation_z
        except Exception as exc:
            log_throttle(
                self,
                "warning",
                5.0,
                "Failed to read gravity compensation configuration: %s",
                exc,
            )
            self.reconfigure_client = None
            return None

    def _write_current_compensation_locked(self, value: float) -> bool:
        if not self._ensure_reconfigure_client_locked():
            return False

        try:
            config = self.reconfigure_client.update_configuration(
                {"gravity_compensation_z": value}
            )
            self.current_gravity_compensation_z = float(
                config.get("gravity_compensation_z", value)
            )
            return True
        except Exception as exc:
            log_throttle(
                self,
                "warning",
                5.0,
                "Failed to update gravity compensation configuration: %s",
                exc,
            )
            self.reconfigure_client = None
            return False

    def _reject_locked(self, reason: str) -> None:
        self.status_reason = reason
        self.samples.clear()

    def _prune_samples_locked(self, now: Time) -> None:
        max_sample_age = Duration(
            nanoseconds=self.stable_window.nanoseconds
            + self.sample_window_slack.nanoseconds
        )
        while self.samples and now - self.samples[0]["stamp"] > max_sample_age:
            self.samples.popleft()

    def _publish_status_locked(self, now: Time) -> None:
        window_duration = 0.0
        if len(self.samples) >= 2:
            window_duration = (
                self.samples[-1]["stamp"] - self.samples[0]["stamp"]
            ).nanoseconds / 1e9

        max_horizontal_velocity = 0.0
        max_horizontal_wrench = 0.0
        if self.samples:
            max_horizontal_velocity = max(
                sample["horizontal_velocity"] for sample in self.samples
            )
            max_horizontal_wrench = max(
                sample["horizontal_wrench"] for sample in self.samples
            )

        payload = {
            "reason": self.status_reason,
            "sample_count": len(self.samples),
            "window_duration": window_duration,
            "stable_window": self.stable_window.nanoseconds / 1e9,
            "active_min_z": self.active_min_z,
            "active_max_z": self.active_max_z,
            "current_gravity_compensation_z": self.current_gravity_compensation_z,
            "estimated_gravity_compensation_z": self.estimated_gravity_compensation_z,
            "estimated_wrench_z_stddev": self.estimated_wrench_z_stddev,
            "control_enabled": self.control_enabled,
            "apply_updates": self.apply_updates,
            "max_horizontal_velocity": max_horizontal_velocity,
            "max_horizontal_wrench": max_horizontal_wrench,
            "use_odometry_twist": self.use_odometry_twist,
        }
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

        if self.estimated_gravity_compensation_z is not None:
            self.estimate_pub.publish(
                Float64(data=self.estimated_gravity_compensation_z)
            )
        if self.current_gravity_compensation_z is not None:
            self.current_pub.publish(Float64(data=self.current_gravity_compensation_z))

    def _message_time(self, stamp) -> Time:
        value = Time.from_msg(stamp)
        return self.get_clock().now() if value.nanoseconds == 0 else value

    @staticmethod
    def _is_recent(stamp: Time, now: Time, timeout: Duration) -> bool:
        if stamp.nanoseconds == 0:
            return False
        return now - stamp <= timeout

    @staticmethod
    def _rotate_body_to_world(orientation, vector):
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 1e-9:
            return None

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        vx, vy, vz = vector
        return (
            (1.0 - 2.0 * (y * y + z * z)) * vx
            + 2.0 * (x * y - z * w) * vy
            + 2.0 * (x * z + y * w) * vz,
            2.0 * (x * y + z * w) * vx
            + (1.0 - 2.0 * (x * x + z * z)) * vy
            + 2.0 * (y * z - x * w) * vz,
            2.0 * (x * z - y * w) * vx
            + 2.0 * (y * z + x * w) * vy
            + (1.0 - 2.0 * (x * x + y * y)) * vz,
        )

    @staticmethod
    def _stddev(values, mean):
        if len(values) < 2:
            return 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _get_bool_param(self, name: str, default: bool) -> bool:
        value = parameter(self, name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def spin(self) -> None:
        rclpy.spin(self)


def main():
    init_ros()
    node = GravityCompensationEstimatorNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._control_clients.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
