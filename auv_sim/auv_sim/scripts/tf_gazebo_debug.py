#!/usr/bin/env python3

import rclpy
from auv_common_lib.ros import get_param, service_client
from auv_common_lib import ros
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from auv_msgs.srv import SpawnDebugCube
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
import tf_transformations as tft
import numpy as np


class TFGazeboDebug(Node):
    def __init__(self):
        super().__init__("tf_gazebo_debug")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.spawned_models = []

        # for reference
        self.robot_name = get_param(self, "~robot_name", "taluy")
        self.robot_base_frame = get_param(
            self, "~robot_base_frame", f"{self.robot_name}/base_link"
        )
        self.world = get_param(self, "~world", "pool")

        # Gazebo Harmonic, through ros_gz_bridge: entity creation and removal are
        # services of the world, the robot's true pose is its odometry publisher.
        # (`remove` is not bridged by the sim launch; bridge it to use delete_all.)
        self.spawn_model_proxy = service_client(
            self, SpawnEntity, f"/world/{self.world}/create"
        )
        self.delete_model_proxy = service_client(
            self, DeleteEntity, f"/world/{self.world}/remove"
        )
        self.spawn_model_proxy.wait_for_service()
        self.gazebo_robot_pose = None
        self.create_subscription(
            Odometry, f"/model/{self.robot_name}/odometry", self.robot_pose_cb, 1
        )

        self.spawn_srv = self.create_service(
            SpawnDebugCube, "/debug/frame_to_model", self.handle_spawn_cube
        )
        self.reset_srv = self.create_service(
            Trigger, "/debug/delete_all", self.handle_reset
        )

        self.cube_sdf = """
<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='debug_cube'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        self.get_logger().info("gazebo tf debugger ready")

    def robot_pose_cb(self, msg):
        self.gazebo_robot_pose = msg.pose.pose

    def pose_to_matrix(self, translation, rotation):
        T = tft.translation_matrix([translation.x, translation.y, translation.z])
        R = tft.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        return np.dot(T, R)

    def matrix_to_pose(self, matrix):
        pose = Pose()
        trans = tft.translation_from_matrix(matrix)
        quat = tft.quaternion_from_matrix(matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = quat
        return pose

    def handle_spawn_cube(self, req, response):
        frame_name = req.frame_name
        try:
            trans_target = self.tf_buffer.lookup_transform(
                "odom", frame_name, Time(), Duration(seconds=1.0)
            )

            gazebo_robot = self.gazebo_robot_pose
            ros_robot = self.tf_buffer.lookup_transform(
                "odom", self.robot_base_frame, Time(), Duration(seconds=1.0)
            )

            if gazebo_robot is None:
                return SpawnDebugCube.Response(
                    success=False,
                    message=f"gazebo error: no pose on /model/{self.robot_name}/odometry",
                )

            M_target_in_odom = self.pose_to_matrix(
                trans_target.transform.translation, trans_target.transform.rotation
            )
            M_robot_in_world = self.pose_to_matrix(
                gazebo_robot.position, gazebo_robot.orientation
            )
            M_robot_in_odom = self.pose_to_matrix(
                ros_robot.transform.translation, ros_robot.transform.rotation
            )

            M_odom_in_world = np.dot(M_robot_in_world, np.linalg.inv(M_robot_in_odom))
            M_final = np.dot(M_odom_in_world, M_target_in_odom)

            pose = self.matrix_to_pose(M_final)
            pose.position.z = 0.0

            model_name = (
                f"debug_cube_{len(self.spawned_models)}_{frame_name.replace('/', '_')}"
            )

            request = SpawnEntity.Request()
            request.entity_factory.name = model_name
            request.entity_factory.sdf = self.cube_sdf
            request.entity_factory.pose = pose
            resp = self.spawn_model_proxy(request)

            if resp.success:
                self.spawned_models.append(model_name)
                return SpawnDebugCube.Response(
                    success=True,
                    message=f"spawned {model_name} at gazebo {pose.position.x}, {pose.position.y}",
                )
            else:
                return SpawnDebugCube.Response(
                    success=False, message="spawn error: Gazebo refused the model"
                )

        except Exception as e:
            return SpawnDebugCube.Response(success=False, message=f"error: {str(e)}")

    def handle_reset(self, req, response):
        success = True
        messages = []
        for model_name in self.spawned_models:
            request = DeleteEntity.Request()
            request.entity.name = model_name
            request.entity.type = Entity.MODEL
            resp = self.delete_model_proxy(request)
            if not resp.success:
                success = False
                messages.append(f"failed to delete {model_name}")

        self.spawned_models = []
        message = "success" if success else "\n".join(messages)
        return Trigger.Response(success=success, message=message)


def main():
    ros.init()
    node = TFGazeboDebug()
    ros.spin(node)


if __name__ == "__main__":
    main()
