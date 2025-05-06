#!/usr/bin/env python3

import os
import rospy
import tf2_ros
import rospkg
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf2_ros import TransformException


class TorpedoLauncherServer:
    def __init__(self):
        rospy.init_node("launch_torpedo_server")
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        self.base_frame = rospy.get_param("~base_frame", "taluy/base_link")

        self.drop_frames = {
            1: rospy.get_param(
                "~torpedo_1_frame", "taluy/base_link/torpedo_upper_link"
            ),
            2: rospy.get_param(
                "~torpedo_2_frame", "taluy/base_link/torpedo_bottom_link"
            ),
        }

        self.torpedo_models = {
            1: {"file": "torpedo.sdf", "name": "torpedo_one"},
            2: {"file": "torpedo.sdf", "name": "torpedo_two"},
        }
        self.model_pkg = rospy.get_param("~model_package", "auv_sim_description")
        self.model_dir = rospy.get_param("~model_subdir", "models/robosub_torpedo")

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        for tid in self.torpedo_models:
            srv_name = f"actuators/torpedo_{tid}/launch"
            rospy.Service(
                srv_name,
                Trigger,
                lambda req, torpedo_id=tid: self.handle_launch(req, torpedo_id),
            )

        rospy.loginfo(
            f"[launch_torpedo_server] Ready. base_frame={self.base_frame}, drop_frames={self.drop_frames}"
        )

    def lookup_drop_pose(self, torpedo_id: int) -> Pose:
        """
        Get the pose of the drop frame relative to base frame
        """
        drop_frame = self.drop_frames[torpedo_id]
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                drop_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
        except TransformException as e:
            raise RuntimeError(
                f"TF lookup failed: {self.base_frame} → {drop_frame}: {e}"
            )

        pose = Pose()
        pose.position.x = trans.transform.translation.x
        pose.position.y = trans.transform.translation.y
        pose.position.z = trans.transform.translation.z
        pose.orientation = trans.transform.rotation
        return pose

    def handle_launch(self, req, torpedo_id: int) -> TriggerResponse:
        """
        Handle a torpedo launch request
        """
        try:
            # 1) Get torpedo drop pose
            pose = self.lookup_drop_pose(torpedo_id)

            # 2) Load the torpedo model
            rp = rospkg.RosPack()
            pkg_path = rp.get_path(self.model_pkg)
            model_info = self.torpedo_models[torpedo_id]
            path = os.path.join(pkg_path, self.model_dir, model_info["file"])
            with open(path, "r") as f:
                xml = f.read()

            # 3) Spawn the torpedo
            resp = self.spawn_model(
                model_name=model_info["name"],
                model_xml=xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame=self.base_frame,  # Just like ball dropper
            )

            if resp.success:
                msg = f"Torpedo {torpedo_id} ({model_info['name']}) spawned."
                rospy.loginfo(msg)
                return TriggerResponse(success=True, message=msg)
            else:
                msg = f"Torpedo {torpedo_id} spawn failed: {resp.status_message}"
                rospy.logwarn(msg)
                return TriggerResponse(success=False, message=msg)

        except Exception as e:
            err = f"Torpedo {torpedo_id} error: {e}"
            rospy.logerr(err)
            return TriggerResponse(success=False, message=err)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    TorpedoLauncherServer().run()
