#!/usr/bin/env python3
import rospy
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from gazebo_msgs.srv import SetModelConfiguration, ApplyJointEffort, JointRequest
from sensor_msgs.msg import JointState
from typing import Optional, Dict


class GripperService:
    def __init__(self):
        rospy.init_node("gripper_service")

        # Parameters
        self.model_name = rospy.get_param("~model_name", "taluy")
        self.robot_description_param = rospy.get_param(
            "~robot_description_param", f"/{self.model_name}/robot_description"
        )
        self.left_joint = rospy.get_param(
            "~left_joint", f"{self.model_name}/finger_left_joint"
        )
        self.right_joint = rospy.get_param(
            "~right_joint", f"{self.model_name}/finger_right_joint"
        )
        # Whether right joint should mirror left (open by rotating in opposite direction)
        self.mirror_right = bool(rospy.get_param("~mirror_right", True))

        # Angles (radians)
        self.open_angle = float(rospy.get_param("~open_angle", 0.35))
        self.close_angle = float(rospy.get_param("~close_angle", 0.0))

        # Motion profile
        self.move_duration = float(rospy.get_param("~move_duration", 1.0))  # seconds
        self.move_steps = int(rospy.get_param("~move_steps", 30))  # interpolation steps

        # Effort-based control parameters (for stall-on-contact)
        self.use_effort = bool(rospy.get_param("~use_effort_control", True))
        self.kp_effort = float(rospy.get_param("~kp_effort", 20.0))  # Nm/rad
        self.max_effort = float(rospy.get_param("~max_effort", 6.0))  # Nm cap
        self.hold_effort = float(
            rospy.get_param("~hold_effort", 1.5)
        )  # Nm per finger when holding
        self.error_tol = float(
            rospy.get_param("~error_tol", 0.01)
        )  # rad to consider reached
        self.vel_tol = float(
            rospy.get_param("~vel_tol", 0.01)
        )  # rad/s to consider stalled
        self.stall_cycles = int(
            rospy.get_param("~stall_cycles", 5)
        )  # consecutive cycles
        self.dt_effort = float(rospy.get_param("~dt_effort", 0.05))

        # Joint state cache (populated from /<ns>/joint_states)
        self._joint_positions: Dict[str, float] = {}
        self._joint_velocities: Dict[str, float] = {}
        self._js_sub = rospy.Subscriber(
            "joint_states", JointState, self._on_joint_states, queue_size=5
        )

        # Gazebo service proxies
        self.set_model_config = rospy.ServiceProxy(
            "/gazebo/set_model_configuration", SetModelConfiguration
        )
        self.apply_joint_effort = rospy.ServiceProxy(
            "/gazebo/apply_joint_effort", ApplyJointEffort
        )
        self.clear_joint_forces = rospy.ServiceProxy(
            "/gazebo/clear_joint_forces", JointRequest
        )

        # Main SetBool service (True=close, False=open)
        rospy.Service("actuators/gripper/set", SetBool, self.handle_set)
        # Convenience services
        rospy.Service("actuators/gripper/open", Trigger, self.handle_open)
        rospy.Service("actuators/gripper/close", Trigger, self.handle_close)

        rospy.loginfo(
            f"[gripper_service] Ready for model='{self.model_name}', joints=({self.left_joint}, {self.right_joint}), robot_description='{self.robot_description_param}'"
        )

    def _on_joint_states(self, msg: JointState):
        for i, name in enumerate(msg.name):
            self._joint_positions[name] = (
                msg.position[i]
                if i < len(msg.position)
                else self._joint_positions.get(name)
            )
            if i < len(msg.velocity):
                self._joint_velocities[name] = msg.velocity[i]

    def _get_current(self, joint_name: str) -> Optional[float]:
        return self._joint_positions.get(joint_name, None)

    def _get_velocity(self, joint_name: str) -> Optional[float]:
        return self._joint_velocities.get(joint_name, None)

    def _wait_for_service(self, name: str, timeout: float = 5.0) -> bool:
        try:
            rospy.wait_for_service(name, timeout=timeout)
            return True
        except Exception as e:
            rospy.logwarn(f"Service '{name}' not available yet: {e}")
            return False

    def _set_angles_once(self, left_angle: float, right_angle: float) -> bool:
        try:
            resp = self.set_model_config(
                model_name=self.model_name,
                urdf_param_name=self.robot_description_param,
                joint_names=[self.left_joint, self.right_joint],
                joint_positions=[left_angle, right_angle],
            )
            if not resp.success:
                rospy.logwarn(f"SetModelConfiguration failed: {resp.status_message}")
            return resp.success
        except Exception as e:
            rospy.logerr(f"SetModelConfiguration call error: {e}")
            return False

    def _move_smooth(self, left_target: float, right_target: float) -> bool:
        if not self._wait_for_service("/gazebo/set_model_configuration"):
            return False

        # Determine start positions; fall back to last target if unknown
        left_start = self._get_current(self.left_joint)
        right_start = self._get_current(self.right_joint)
        if left_start is None:
            left_start = left_target  # avoid jump if unknown
        if right_start is None:
            right_start = right_target

        steps = max(1, self.move_steps)
        dt = self.move_duration / float(steps)

        ok = True
        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                return False
            alpha = float(i) / float(steps)
            l = (1.0 - alpha) * left_start + alpha * left_target
            r = (1.0 - alpha) * right_start + alpha * right_target
            ok = self._set_angles_once(l, r) and ok
            rospy.sleep(max(0.0, dt))
        return ok

    # Effort-based position approach that stalls on contact
    def _apply_effort_once(self, joint: str, effort: float, duration: float) -> bool:
        try:
            resp = self.apply_joint_effort(
                joint_name=joint,
                effort=float(effort),
                start_time=rospy.Time(0),
                duration=rospy.Duration.from_sec(duration),
            )
            if not resp.success:
                rospy.logwarn(f"ApplyJointEffort failed for {joint}")
            return bool(resp.success)
        except Exception as e:
            rospy.logerr(f"ApplyJointEffort error for {joint}: {e}")
            return False

    def _clear_force(self, joint: str):
        try:
            self.clear_joint_forces(joint_name=joint)
        except Exception:
            pass

    def _move_with_effort(
        self, left_target: float, right_target: float, hold: bool = False
    ) -> bool:
        # Wait for services
        if not (
            self._wait_for_service("/gazebo/apply_joint_effort")
            and self._wait_for_service("/gazebo/clear_joint_forces")
        ):
            rospy.logwarn(
                "Effort services unavailable, falling back to SetModelConfiguration"
            )
            return self._move_smooth(left_target, right_target)

        # Initial clear of any lingering forces
        self._clear_force(self.left_joint)
        self._clear_force(self.right_joint)

        # Control loop
        dt = max(0.01, self.dt_effort)
        stall_count_l = 0
        stall_count_r = 0
        ok = True
        t0 = rospy.Time.now().to_sec()
        timeout = max(self.move_duration, 0.5)

        while not rospy.is_shutdown():
            # Time check to avoid infinite loops
            if rospy.Time.now().to_sec() - t0 > 3.0 * timeout:
                rospy.logwarn("Effort control timeout")
                ok = False
                break

            lp = self._get_current(self.left_joint)
            rp = self._get_current(self.right_joint)
            lv = self._get_velocity(self.left_joint) or 0.0
            rv = self._get_velocity(self.right_joint) or 0.0

            if lp is None or rp is None:
                # Wait a bit for joint states
                rospy.sleep(0.02)
                continue

            le = float(left_target - lp)
            re = float(right_target - rp)

            # PD-like (P only) effort with saturation
            leff = max(-self.max_effort, min(self.max_effort, self.kp_effort * le))
            reff = max(-self.max_effort, min(self.max_effort, self.kp_effort * re))

            # Apply for a short duration
            ok = self._apply_effort_once(self.left_joint, leff, dt) and ok
            ok = self._apply_effort_once(self.right_joint, reff, dt) and ok

            # Stall detection: if not near target and velocity ~0 over several cycles
            if abs(le) > self.error_tol and abs(lv) < self.vel_tol:
                stall_count_l += 1
            else:
                stall_count_l = 0
            if abs(re) > self.error_tol and abs(rv) < self.vel_tol:
                stall_count_r += 1
            else:
                stall_count_r = 0

            # If both stalled (contact), switch to hold or stop
            if (
                stall_count_l >= self.stall_cycles
                and stall_count_r >= self.stall_cycles
            ):
                if hold:
                    # Apply gentle inward hold effort continuously
                    le_sign = 1.0 if le > 0.0 else -1.0  # direction toward target
                    re_sign = 1.0 if re > 0.0 else -1.0
                    self._apply_effort_once(
                        self.left_joint, le_sign * self.hold_effort, dt
                    )
                    self._apply_effort_once(
                        self.right_joint, re_sign * self.hold_effort, dt
                    )
                    # Keep looping to maintain hold until another command
                else:
                    break

            # If both reached target within tolerance
            if abs(le) <= self.error_tol and abs(re) <= self.error_tol:
                break

            rospy.sleep(dt)

        # If not holding, clear forces so joint stops driving
        if not hold:
            self._clear_force(self.left_joint)
            self._clear_force(self.right_joint)

        return ok

    def set_angles(
        self, left_angle: float, right_angle: float, hold: bool = False
    ) -> bool:
        if self.use_effort:
            return self._move_with_effort(left_angle, right_angle, hold=hold)
        else:
            return self._move_smooth(left_angle, right_angle)

    def handle_set(self, req):
        # True => close, False => open
        base_target = self.close_angle if req.data else self.open_angle
        right_target = -base_target if self.mirror_right else base_target
        # When closing, hold grip; when opening, do not hold
        ok = self.set_angles(base_target, right_target, hold=bool(req.data))
        msg = ("closed" if req.data else "opened") if ok else "command failed"
        return SetBoolResponse(success=ok, message=msg)

    def handle_open(self, _req):
        # Clear any holding force first
        self._clear_force(self.left_joint)
        self._clear_force(self.right_joint)
        base = self.open_angle
        right_target = -base if self.mirror_right else base
        ok = self.set_angles(base, right_target, hold=False)
        return TriggerResponse(success=ok, message="opened" if ok else "open failed")

    def handle_close(self, _req):
        base = self.close_angle
        right_target = -base if self.mirror_right else base
        ok = self.set_angles(base, right_target, hold=True)
        return TriggerResponse(success=ok, message="closed" if ok else "close failed")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        GripperService().run()
    except Exception as e:
        rospy.logfatal(f"gripper_service crashed on startup: {e}")
        raise
