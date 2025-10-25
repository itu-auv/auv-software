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
        # We'll support two models/services (bottle, ladle). Each has its own set of private params:
        #   ~<key>_model_package, ~<key>_model_subdir, ~<key>_model_file, ~<key>_model_name, ~<key>_service_name
        # Defaults chosen to be sensible for this repo.
        self.keys = ['bottle', 'ladle']
        self.configs = {}
        for key in self.keys:
            pkg = rospy.get_param(f'~{key}_model_package', 'auv_sim_description')
            subdir = rospy.get_param(f'~{key}_model_subdir', f'models/robosub_{key}')
            fname = rospy.get_param(f'~{key}_model_file', 'model.sdf')
            model_name = rospy.get_param(f'~{key}_model_name', f'{key}_final1')
            service_name = rospy.get_param(f'~{key}_service_name', f'actuators/{key}/spawn')
            self.configs[key] = {
                'model_package': pkg,
                'model_subdir': subdir,
                'model_file': fname,
                'model_name': model_name,
                'service_name': service_name,
            }

        # Gazebo spawn proxy (shared)
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Register both services. Use a small closure to bind the key to the callback.
        for key in self.keys:
            svc_name = self.configs[key]['service_name']
            rospy.Service(svc_name, Trigger, lambda req, k=key: self.handle_spawn(req, k))

        registered = ', '.join([f"{k} -> {self.configs[k]['service_name']}" for k in self.keys])
        rospy.loginfo(f'[spawn_model_server] Ready. Registered services: {registered}')

    def handle_spawn(self, req, key='bottle') -> TriggerResponse:
        """Handle a Trigger request and spawn the model identified by `key`.

        key: one of the keys listed in self.keys (default 'bottle').
        """
        if key not in self.configs:
            msg = f'Unknown model key: {key}'
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

        cfg = self.configs[key]
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path(cfg['model_package'])
            path = os.path.join(pkg_path, cfg['model_subdir'], cfg['model_file'])
            with open(path, 'r') as f:
                xml = f.read()

            pose = Pose()
            pose.position.x = 13.0
            pose.position.y = -4.25
            pose.position.z = -1.25

            resp = self.spawn_model(
                model_name=cfg['model_name'],
                model_xml=xml,
                robot_namespace='',
                initial_pose=pose,
                reference_frame='world'
            )

            if resp.success:
                msg = f"{cfg['model_name']} spawned at origin. (from {path})"
                rospy.loginfo(msg)
                return TriggerResponse(success=True, message=msg)
            else:
                msg = f"Spawn failed: {resp.status_message}"
                rospy.logwarn(msg)
                return TriggerResponse(success=False, message=msg)

        except Exception as e:
            err = f"Error spawning {cfg.get('model_name','<unknown>')}: {e}"
            rospy.logerr(err)
            return TriggerResponse(success=False, message=err)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    ModelSpawner().run()
