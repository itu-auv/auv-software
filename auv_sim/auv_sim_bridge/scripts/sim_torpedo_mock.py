#!/usr/bin/env python3

import os
import rospy
import tf2_ros
import rospkg
import threading
import time
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel, GetModelState, SetModelState
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelState
from tf2_ros import TransformException
from tf.transformations import euler_from_quaternion, quaternion_matrix


class TorpedoLauncherServer:
    def __init__(self):
        rospy.init_node("launch_torpedo_server")
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        self.base_frame = rospy.get_param("~base_frame", "taluy_mini/base_link")

        self.drop_frames = {
            1: rospy.get_param(
                "~torpedo_1_frame", "taluy_mini/base_link/torpedo_upper_link"
            ),
            2: rospy.get_param(
                "~torpedo_2_frame", "taluy_mini/base_link/torpedo_bottom_link"
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
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

        self.initial_velocity = 1.0

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

    def get_vehicle_state(self):
        vehicle_state = self.get_model_state("taluy_mini", "world")
        vehicle_orientation = vehicle_state.pose.orientation
        quat = [
            vehicle_orientation.x,
            vehicle_orientation.y,
            vehicle_orientation.z,
            vehicle_orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        return vehicle_state, quat, roll, pitch, yaw

    def calculate_torpedo_velocity(self, quat):
        rot_matrix = quaternion_matrix(quat)[:3, :3]
        initial_velocity = [self.initial_velocity, 0.0, 0.0]
        world_velocity = rot_matrix.dot(initial_velocity)

        twist = Twist()
        twist.linear.x = world_velocity[0]
        twist.linear.y = world_velocity[1]
        twist.linear.z = world_velocity[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        return twist, world_velocity

    def apply_torpedo_velocity(self, model_name, torpedo_id):
        try:
            vehicle_state, quat, _, _, _ = self.get_vehicle_state()

            twist, _ = self.calculate_torpedo_velocity(quat)

            torpedo_state = self.get_model_state(model_name, "world")

            self.set_model_state(
                ModelState(
                    model_name=model_name,
                    pose=torpedo_state.pose,
                    twist=twist,
                    reference_frame="world",
                )
            )

            rospy.loginfo(f"Torpedo {torpedo_id} velocity applied. Moving forward.")
        except Exception as e:
            rospy.logerr(f"Failed to apply velocity to torpedo {torpedo_id}: {str(e)}")

    def handle_launch(self, req, torpedo_id: int) -> TriggerResponse:
        try:
            pose = self.lookup_drop_pose(torpedo_id)
            vehicle_state, quat, _, _, _ = self.get_vehicle_state()
            rp = rospkg.RosPack()
            pkg_path = rp.get_path(self.model_pkg)
            model_info = self.torpedo_models[torpedo_id]
            path = os.path.join(pkg_path, self.model_dir, model_info["file"])
            with open(path, "r") as f:
                xml = f.read()

            resp = self.spawn_model(
                model_name=model_info["name"],
                model_xml=xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame=self.base_frame,
            )

            if resp.success:
                threading.Timer(
                    0.1,
                    lambda: self.apply_torpedo_velocity(model_info["name"], torpedo_id),
                ).start()

                msg = f"Torpedo {torpedo_id} ({model_info['name']}) spawned and moving."
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
