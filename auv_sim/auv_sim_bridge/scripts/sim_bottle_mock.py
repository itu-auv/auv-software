#!/usr/bin/env python3
import os
import rospy
import rospkg
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose


class ModelSpawner:
    def __init__(self):
        # Node name kept generic; service and model names are configurable via private params
        rospy.init_node('spawn_model_server')

        # parameters (private namespace)
        self.model_pkg = rospy.get_param('~model_package', 'auv_sim_description')
        # model_subdir should point to the folder containing model.sdf (default kept as-is to avoid moving mesh files)
        self.model_dir = rospy.get_param('~model_subdir', 'models/robosub_objects')
        # name the spawned model will have in Gazebo
        self.model_name = rospy.get_param('~model_name', 'robosub_ladle')
        # service name to expose for spawning; make this configurable so this script can spawn any model
        self.service_name = rospy.get_param('~service_name', 'actuators/ladle/spawn')

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        rospy.Service(self.service_name, Trigger, self.handle_spawn)
        rospy.loginfo(f'[spawn_model_server] Ready. Call {self.service_name} to spawn {self.model_name} at origin')

    def handle_spawn(self, req) -> TriggerResponse:
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path(self.model_pkg)
            path = os.path.join(pkg_path, self.model_dir, 'model.sdf')
            with open(path, 'r') as f:
                xml = f.read()

            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = 0.0

            resp = self.spawn_model(
                model_name=self.model_name,
                model_xml=xml,
                robot_namespace='',
                initial_pose=pose,
                reference_frame='world'
            )

            if resp.success:
                msg = f'{self.model_name} spawned at origin.'
                rospy.loginfo(msg)
                return TriggerResponse(success=True, message=msg)
            else:
                msg = f"Spawn failed: {resp.status_message}"
                rospy.logwarn(msg)
                return TriggerResponse(success=False, message=msg)

        except Exception as e:
            err = f"Error spawning {self.model_name}: {e}"
            rospy.logerr(err)
            return TriggerResponse(success=False, message=err)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    ModelSpawner().run()
