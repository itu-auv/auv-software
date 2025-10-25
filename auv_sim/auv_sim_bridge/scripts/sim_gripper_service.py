#!/usr/bin/env python3
import rospy
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from gazebo_msgs.srv import SetModelConfiguration


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

        # Angles (radians)
        self.open_angle = float(rospy.get_param("~open_angle", 0.35))
        self.close_angle = float(rospy.get_param("~close_angle", 0.0))

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

    def set_angles(self, left_angle: float, right_angle: float) -> bool:
        # Try to ensure Gazebo service is available when called
        try:
            rospy.wait_for_service("/gazebo/set_model_configuration", timeout=5.0)
        except Exception as e:
            rospy.logwarn(f"Gazebo service not available yet: {e}")
            return False

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

    def handle_set(self, req):
        # True => close, False => open
        target = self.close_angle if req.data else self.open_angle
        # Left moves +target, Right moves -target for mirrored motion
        ok = self.set_angles(target, -target)
        msg = ("closed" if req.data else "opened") if ok else "command failed"
        return SetBoolResponse(success=ok, message=msg)

    def handle_open(self, _req):
        ok = self.set_angles(self.open_angle, -self.open_angle)
        return TriggerResponse(success=ok, message="opened" if ok else "open failed")

    def handle_close(self, _req):
        ok = self.set_angles(self.close_angle, -self.close_angle)
        return TriggerResponse(success=ok, message="closed" if ok else "close failed")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        GripperService().run()
    except Exception as e:
        rospy.logfatal(f"gripper_service crashed on startup: {e}")
        raise
