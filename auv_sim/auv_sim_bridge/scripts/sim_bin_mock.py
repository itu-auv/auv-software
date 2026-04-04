#!/usr/bin/env python3
import os
import rospy
import tf2_ros
import rospkg
from std_msgs.msg import Float32
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf2_ros import TransformException


class DropBallServer:
    def __init__(self):
        rospy.init_node("drop_ball_server")
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        self.base_frame = rospy.get_param("~base_frame", "taluy/base_link")

        self.ball_models = {
            "positive": {
                "file": "ball.sdf",
                "name": "sphere_one",
                "drop_frame": rospy.get_param(
                    "~drop_frame_1", "taluy/base_link/ball_dropper_1_link"
                ),
                "dropped": False,
            },
            "negative": {
                "file": "ball.sdf",
                "name": "sphere_two",
                "drop_frame": rospy.get_param(
                    "~drop_frame_2", "taluy/base_link/ball_dropper_2_link"
                ),
                "dropped": False,
            },
        }

        self.last_angle = 0.0

        self.model_pkg = rospy.get_param("~model_package", "auv_sim_description")
        self.model_dir = rospy.get_param("~model_subdir", "models/robosub_bin")

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.Subscriber(
            "actuators/ball_dropper/set_angle", Float32, self.handle_drop_ball
        )
        rospy.loginfo(
            f"[drop_ball_server] Ready. base_frame={self.base_frame}, "
            f"drop_frames=[{self.ball_models['positive']['drop_frame']}, {self.ball_models['negative']['drop_frame']}]"
        )

    def _select_model_by_angle(self, angle: float):
        if angle > 0.0:
            return self.ball_models["positive"]
        return self.ball_models["negative"]

    def lookup_drop_pose(self, drop_frame: str, timeout: float = 4.0) -> Pose:
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, drop_frame, rospy.Time(0), rospy.Duration(timeout)
            )
        except TransformException as e:
            raise

        pose = Pose()
        pose.position.x = trans.transform.translation.x
        pose.position.y = trans.transform.translation.y
        pose.position.z = trans.transform.translation.z
        pose.orientation = trans.transform.rotation
        return pose

    def handle_drop_ball(self, msg):
        angle = msg.data

        if abs(angle) <= 1e-3:
            rospy.loginfo("[drop_ball_server] Set angle 0")
            return

        if abs(angle - self.last_angle) <= 1e-3:
            return
        self.last_angle = angle

        try:
            if all(model["dropped"] for model in self.ball_models.values()):
                rospy.logwarn("[drop_ball_server] All balls already dropped.")
                return

            model = self._select_model_by_angle(angle)
            if model["dropped"]:
                rospy.logwarn(
                    f"[drop_ball_server] {model['name']} already dropped for angle {angle}."
                )
                return

            pose = self.lookup_drop_pose(model["drop_frame"])

            rp = rospkg.RosPack()
            pkg_path = rp.get_path(self.model_pkg)
            model_dir = os.path.join(pkg_path, self.model_dir)

            path = os.path.join(model_dir, model["file"])
            with open(path, "r") as f:
                xml = f.read()

            resp = self.spawn_model(
                model_name=model["name"],
                model_xml=xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame=self.base_frame,
            )
            if resp.success:
                rospy.loginfo(
                    f"[drop_ball_server] {model['name']} spawned from {model['drop_frame']}."
                )
                model["dropped"] = True
            else:
                rospy.logwarn(
                    f"[drop_ball_server] {model['name']} spawn failed: {resp.status_message}"
                )

        except TransformException as te:
            rospy.logerr(f"[drop_ball_server] TF lookup failed: {te}")

        except Exception as ex:
            rospy.logerr(f"[drop_ball_server] Spawn failed: {ex}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    DropBallServer().run()
