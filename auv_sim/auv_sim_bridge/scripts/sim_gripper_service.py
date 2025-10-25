#!/usr/bin/env python3
import rospy
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from gazebo_msgs.srv import SetModelConfiguration
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

        # Joint state cache (populated from /<ns>/joint_states)
        self._joint_positions: Dict[str, float] = {}
        self._js_sub = rospy.Subscriber(
            "joint_states", JointState, self._on_joint_states, queue_size=5
        )

        # Lazy-init service proxy (Gazebo might not be up yet)
        self.set_model_config = rospy.ServiceProxy(
            "/gazebo/set_model_configuration", SetModelConfiguration
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
        for name, pos in zip(msg.name, msg.position):
            self._joint_positions[name] = pos

    def _get_current(self, joint_name: str) -> Optional[float]:
        return self._joint_positions.get(joint_name, None)

    def _wait_for_gazebo(self) -> bool:
        try:
            rospy.wait_for_service("/gazebo/set_model_configuration", timeout=5.0)
            return True
        except Exception as e:
            rospy.logwarn(f"Gazebo service not available yet: {e}")
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
        if not self._wait_for_gazebo():
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

    def set_angles(self, left_angle: float, right_angle: float) -> bool:
        # Smoothly move joints instead of teleporting
        return self._move_smooth(left_angle, right_angle)

    def handle_set(self, req):
        # True => close, False => open
        base_target = self.close_angle if req.data else self.open_angle
        right_target = -base_target if self.mirror_right else base_target
        ok = self.set_angles(base_target, right_target)
        msg = ("closed" if req.data else "opened") if ok else "command failed"
        return SetBoolResponse(success=ok, message=msg)

    def handle_open(self, _req):
        base = self.open_angle
        right_target = -base if self.mirror_right else base
        ok = self.set_angles(base, right_target)
        return TriggerResponse(success=ok, message="opened" if ok else "open failed")

    def handle_close(self, _req):
        base = self.close_angle
        right_target = -base if self.mirror_right else base
        ok = self.set_angles(base, right_target)
        return TriggerResponse(success=ok, message="closed" if ok else "close failed")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        GripperService().run()
    except Exception as e:
        rospy.logfatal(f"gripper_service crashed on startup: {e}")
        raise
