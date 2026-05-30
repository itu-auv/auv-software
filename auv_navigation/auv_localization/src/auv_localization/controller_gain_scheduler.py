from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import rospy
import dynamic_reconfigure.client


# Controller config keys that this scheduler manages.
_AFFECTED_KEYS: tuple = (
    "kp_0",
    "kp_1",
    "kp_6",
    "kp_7",
    "ki_0",
    "ki_1",
    "ki_6",
    "ki_7",
    "kd_0",
    "kd_1",
    "kd_6",
    "kd_7",
)


class ControllerGainScheduler:
    def __init__(
        self,
        server_name: str,
        position_kp_xy: Sequence[float],
        position_ki_xy: Sequence[float],
        position_kd_xy: Sequence[float],
        velocity_kp_xy: Sequence[float],
        velocity_ki_xy: Sequence[float],
        velocity_kd_xy: Sequence[float],
        connect_timeout: float = 1.0,
        log_prefix: str = "ControllerGainScheduler",
    ) -> None:
        self._server_name = server_name
        self._connect_timeout = connect_timeout
        self._log_prefix = log_prefix

        self._overrides: Dict[str, float] = {
            "kp_0": float(position_kp_xy[0]),
            "kp_1": float(position_kp_xy[1]),
            "ki_0": float(position_ki_xy[0]),
            "ki_1": float(position_ki_xy[1]),
            "kd_0": float(position_kd_xy[0]),
            "kd_1": float(position_kd_xy[1]),
            "kp_6": float(velocity_kp_xy[0]),
            "kp_7": float(velocity_kp_xy[1]),
            "ki_6": float(velocity_ki_xy[0]),
            "ki_7": float(velocity_ki_xy[1]),
            "kd_6": float(velocity_kd_xy[0]),
            "kd_7": float(velocity_kd_xy[1]),
        }

        self._client: Optional[dynamic_reconfigure.client.Client] = None
        self._overridden: bool = False
        self._saved_gains: Optional[Dict[str, float]] = None

    @property
    def is_overridden(self) -> bool:
        return self._overridden

    def apply(self, should_override: bool) -> None:
        if bool(should_override) == self._overridden:
            return

        client = self._ensure_client()
        if client is None:
            return

        if should_override:
            self._enter_override(client)
        else:
            self._exit_override(client)

    def update_overrides(
        self,
        position_kp_xy: Sequence[float],
        position_ki_xy: Sequence[float],
        position_kd_xy: Sequence[float],
        velocity_kp_xy: Sequence[float],
        velocity_ki_xy: Sequence[float],
        velocity_kd_xy: Sequence[float],
    ) -> None:
        """Update the override gain values. If an override is currently active,
        the new values are pushed to the controller immediately."""
        self._overrides.update(
            {
                "kp_0": float(position_kp_xy[0]),
                "kp_1": float(position_kp_xy[1]),
                "ki_0": float(position_ki_xy[0]),
                "ki_1": float(position_ki_xy[1]),
                "kd_0": float(position_kd_xy[0]),
                "kd_1": float(position_kd_xy[1]),
                "kp_6": float(velocity_kp_xy[0]),
                "kp_7": float(velocity_kp_xy[1]),
                "ki_6": float(velocity_ki_xy[0]),
                "ki_7": float(velocity_ki_xy[1]),
                "kd_6": float(velocity_kd_xy[0]),
                "kd_7": float(velocity_kd_xy[1]),
            }
        )
        if self._overridden:
            client = self._ensure_client()
            if client is not None:
                try:
                    client.update_configuration(self._overrides)
                    rospy.logwarn(
                        f"{self._log_prefix}: DVL invalid, gain scheduled PID updated – "
                        f"pos_kp=[{position_kp_xy[0]}, {position_kp_xy[1]}] "
                        f"pos_ki=[{position_ki_xy[0]}, {position_ki_xy[1]}] "
                        f"pos_kd=[{position_kd_xy[0]}, {position_kd_xy[1]}] "
                        f"vel_kp=[{velocity_kp_xy[0]}, {velocity_kp_xy[1]}] "
                        f"vel_ki=[{velocity_ki_xy[0]}, {velocity_ki_xy[1]}] "
                        f"vel_kd=[{velocity_kd_xy[0]}, {velocity_kd_xy[1]}]"
                    )
                except Exception as e:
                    rospy.logwarn_throttle(
                        5.0,
                        f"{self._log_prefix}: failed to push updated override gains: {e}",
                    )

    def shutdown(self) -> None:
        """Best-effort restore on node shutdown if an override is active."""
        if not self._overridden:
            return
        client = self._ensure_client()
        if client is None:
            return
        self._exit_override(client)

    def _ensure_client(self) -> Optional[dynamic_reconfigure.client.Client]:
        if self._client is not None:
            return self._client
        try:
            target = rospy.resolve_name(self._server_name)
            self._client = dynamic_reconfigure.client.Client(
                target, timeout=self._connect_timeout
            )
            rospy.loginfo(
                f"{self._log_prefix}: connected to dynamic_reconfigure '{target}'"
            )
        except Exception as e:
            rospy.logwarn_throttle(
                5.0,
                f"{self._log_prefix}: waiting for dynamic_reconfigure server "
                f"'{self._server_name}': {e}",
            )
            self._client = None
        return self._client

    def _enter_override(self, client: dynamic_reconfigure.client.Client) -> None:
        try:
            current_cfg = client.get_configuration(timeout=self._connect_timeout)
        except Exception as e:
            rospy.logwarn_throttle(
                5.0,
                f"{self._log_prefix}: failed to read current cfg before "
                f"override: {e}",
            )
            return

        self._saved_gains = {k: current_cfg.get(k) for k in _AFFECTED_KEYS}

        try:
            client.update_configuration(self._overrides)
        except Exception as e:
            rospy.logwarn_throttle(
                5.0,
                f"{self._log_prefix}: failed to push override gains: {e}",
            )
            return

        self._overridden = True
        rospy.logwarn(
            f"{self._log_prefix}: pushed dvl_invalid_*_xy gains to controller"
        )

    def _exit_override(self, client: dynamic_reconfigure.client.Client) -> None:
        try:
            if self._saved_gains:
                restore = {k: v for k, v in self._saved_gains.items() if v is not None}
                if restore:
                    client.update_configuration(restore)
            rospy.loginfo(f"{self._log_prefix}: restored controller gains")
        except Exception as e:
            rospy.logwarn_throttle(
                5.0,
                f"{self._log_prefix}: failed to restore gains: {e}",
            )
        finally:
            self._overridden = False
            self._saved_gains = None
