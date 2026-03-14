#!/usr/bin/env python3
import os
import rospy
import rospkg
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import tf.transformations as tft
import random
import math


class ModelSpawner:
    def __init__(self):
        rospy.init_node("spawn_model_server")
        self.keys = ["bottle", "ladle"]
        self.configs = {}
        for key in self.keys:
            pkg = rospy.get_param(f"~{key}_model_package", "auv_sim_description")
            subdir = rospy.get_param(f"~{key}_model_subdir", f"models/robosub_{key}")
            fname = rospy.get_param(f"~{key}_model_file", "model.sdf")
            model_name = rospy.get_param(f"~{key}_model_name", f"{key}_final1")
            service_name = rospy.get_param(
                f"~{key}_service_name", f"actuators/{key}/spawn"
            )
            self.configs[key] = {
                "model_package": pkg,
                "model_subdir": subdir,
                "model_file": fname,
                "model_name": model_name,
                "service_name": service_name,
            }

        self.spawn_counts = {}
        self.spawned_positions = []

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        for key in self.keys:
            svc_name = self.configs[key]["service_name"]
            rospy.Service(
                svc_name, Trigger, lambda req, k=key: self.handle_spawn(req, k)
            )

        registered = ", ".join(
            [f"{k} -> {self.configs[k]['service_name']}" for k in self.keys]
        )
        rospy.loginfo(f"[spawn_model_server] Ready. Registered services: {registered}")

        self._auto_spawn_models()

    def _get_random_table_pose(self, min_dist=0.2):
        """Pick a random (x, y) on the table that is at least min_dist from all
        previously spawned positions."""
        for _ in range(100):
            x = random.uniform(12.8, 13.2)
            y = random.uniform(-4.45, -4.05)
            too_close = False
            for px, py in self.spawned_positions:
                if math.hypot(x - px, y - py) < min_dist:
                    too_close = True
                    break
            if not too_close:
                return x, y
        return 13.0, -4.25

    def _get_unique_model_name(self, key):
        """Return a unique Gazebo model name for the given key.
        First spawn: base name (e.g. bottle_final1)
        Subsequent:  base_name_N  (e.g. bottle_final1_2, bottle_final1_3, ...)
        """
        base = self.configs[key]["model_name"]
        count = self.spawn_counts.get(key, 0) + 1
        self.spawn_counts[key] = count
        if count == 1:
            return base
        return f"{base}_{count}"

    def _auto_spawn_models(self):
        """Automatically spawn the bottle and ladle models with 90 degree rotation on startup."""
        bottle_x, bottle_y = self._get_random_table_pose()
        self._spawn_single_model("bottle", bottle_x, bottle_y, -1.0)

        ladle_x, ladle_y = self._get_random_table_pose()
        self._spawn_single_model("ladle", ladle_x, ladle_y, -1.0)

    def _spawn_single_model(self, key, x, y, z):
        if key not in self.configs:
            rospy.logwarn(f"[spawn_model_server] Cannot spawn: '{key}' not configured")
            return False, f"'{key}' not configured"

        cfg = self.configs[key]
        unique_name = self._get_unique_model_name(key)
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path(cfg["model_package"])
            path = os.path.join(pkg_path, cfg["model_subdir"], cfg["model_file"])
            with open(path, "r") as f:
                xml = f.read()

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            q = tft.quaternion_from_euler(
                0, math.pi / 2, random.uniform(0, 2 * math.pi)
            )
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            resp = self.spawn_model(
                model_name=unique_name,
                model_xml=xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world",
            )

            if resp.success:
                self.spawned_positions.append((x, y))
                msg = (
                    f"[spawn_model_server] Spawned '{unique_name}' at {x:.2f}, {y:.2f}"
                )
                rospy.loginfo(msg)
                return True, msg
            else:
                msg = f"[spawn_model_server] Spawn failed: {resp.status_message}"
                rospy.logwarn(msg)
                return False, msg

        except Exception as e:
            err = f"[spawn_model_server] Error spawning {unique_name}: {e}"
            rospy.logerr(err)
            return False, err

    def handle_spawn(self, req, key="bottle") -> TriggerResponse:
        """Handle a Trigger request and spawn the model identified by `key`.

        key: one of the keys listed in self.keys (default 'bottle').
        """
        if key not in self.configs:
            msg = f"Unknown model key: {key}"
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

        x, y = self._get_random_table_pose()
        success, msg = self._spawn_single_model(key, x, y, -1.0)
        return TriggerResponse(success=success, message=msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    ModelSpawner().run()
