#!/usr/bin/env python3
import os
import rospy
import tf2_ros
import rospkg
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf2_ros import TransformException


class DropBallServer:
    def __init__(self):
        rospy.init_node("drop_ball_server")
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        self.base_frame = rospy.get_param("~base_frame", "taluy/base_link")

        self.ball_models = [
            {
                "file": "ball.sdf",
                "name": "sphere_one",
                "drop_frame": rospy.get_param(
                    "~drop_frame_1", "taluy/base_link/ball_dropper_1_link"
                ),
            },
            {
                "file": "ball.sdf",
                "name": "sphere_two",
                "drop_frame": rospy.get_param(
                    "~drop_frame_2", "taluy/base_link/ball_dropper_2_link"
                ),
            },
        ]

        self.drop_index = 0

        self.model_pkg = rospy.get_param("~model_package", "auv_sim_description")
        self.model_dir = rospy.get_param("~model_subdir", "models/robosub_bin")

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.Service("actuators/ball_dropper/drop", Trigger, self.handle_drop_ball)
        rospy.loginfo(
            f"[drop_ball_server] Ready. base_frame={self.base_frame}, "
            f"drop_frames=[{self.ball_models[0]['drop_frame']}, {self.ball_models[1]['drop_frame']}]"
        )

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

    def handle_drop_ball(self, req) -> TriggerResponse:
        try:
            if self.drop_index >= len(self.ball_models):
                return TriggerResponse(
                    success=False, message="All balls already dropped."
                )

            model = self.ball_models[self.drop_index]
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
                msg = f"{model['name']} spawned from {model['drop_frame']}."
                rospy.loginfo(msg)
                self.drop_index += 1
                return TriggerResponse(success=True, message=msg)
            else:
                rospy.logwarn(f"{model['name']} spawn failed: {resp.status_message}")
                return TriggerResponse(
                    success=False, message=f"Spawn failed: {resp.status_message}"
                )

        except TransformException as te:
            err = f"TF lookup failed: {te}"
            rospy.logerr(err)
            return TriggerResponse(success=False, message=err)

        except Exception as ex:
            err = f"Spawn failed: {ex}"
            rospy.logerr(err)
            return TriggerResponse(success=False, message=err)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    DropBallServer().run()
