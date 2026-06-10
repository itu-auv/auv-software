#!/usr/bin/env python3
import math
import rospy
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
from gazebo_msgs.srv import SpawnModel, SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import quaternion_matrix


class SimPingerMock:
    def __init__(self):
        rospy.init_node("sim_pinger_mock")

        self.robot_name = rospy.get_param("~robot_name", "taluy")

        self.pinger_x = rospy.get_param("~pinger_x", 10.0)
        self.pinger_y = rospy.get_param("~pinger_y", 0.0)
        self.pinger_z = rospy.get_param("~pinger_z", -1.0)

        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        self.topic_name = rospy.get_param(
            "~topic_name", "/taluy/acoustic/hydrophone/base_angle"
        )

        self.pub = rospy.Publisher(self.topic_name, Float32, queue_size=1)

        rospy.loginfo("[sim_pinger_mock] Waiting for Gazebo services...")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )

        # z = 0
        self.spawn_pinger_cube()

        rospy.loginfo(
            f"[sim_pinger_mock] Initialized. Pinger position: ({self.pinger_x}, {self.pinger_y}, {self.pinger_z}) "
            f"Publishing on topic '{self.topic_name}' at {self.publish_rate} Hz."
        )

    def spawn_pinger_cube(self):
        sdf_xml = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="pinger_cube">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>1.0 1.0 0.0 1.0</ambient>
          <diffuse>1.0 1.0 0.0 1.0</diffuse>
          <specular>0.1 0.1 0.1 1.0</specular>
          <emissive>0 0 0 0</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        pose = Pose()
        pose.position.x = self.pinger_x
        pose.position.y = self.pinger_y
        pose.position.z = 0.0
        pose.orientation.w = 1.0

        try:
            resp = self.spawn_model(
                model_name="pinger_cube",
                model_xml=sdf_xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world",
            )
            if resp.success:
                rospy.loginfo(
                    "[sim_pinger_mock] Successfully spawned yellow 10cm pinger cube in Gazebo."
                )
            else:
                rospy.logwarn(
                    f"[sim_pinger_mock] Failed to spawn pinger cube: {resp.status_message}"
                )
        except Exception as e:
            rospy.logwarn(f"[sim_pinger_mock] Exception during Gazebo spawn: {e}")

    def update_gazebo_cube_state(self):
        model_state = ModelState()
        model_state.model_name = "pinger_cube"
        model_state.pose.position.x = self.pinger_x
        model_state.pose.position.y = self.pinger_y
        model_state.pose.position.z = 0.0
        model_state.pose.orientation.w = 1.0
        model_state.reference_frame = "world"

        try:
            self.set_model_state(model_state)
        except Exception as e:
            rospy.logwarn(f"[sim_pinger_mock] Failed to update Gazebo cube state: {e}")

    def check_and_update_pinger_position(self):
        curr_x = rospy.get_param("~pinger_x", self.pinger_x)
        curr_y = rospy.get_param("~pinger_y", self.pinger_y)
        curr_z = rospy.get_param("~pinger_z", self.pinger_z)

        if (
            curr_x != self.pinger_x
            or curr_y != self.pinger_y
            or curr_z != self.pinger_z
        ):
            self.pinger_x = curr_x
            self.pinger_y = curr_y
            self.pinger_z = curr_z
            self.update_gazebo_cube_state()
            rospy.loginfo(
                f"[sim_pinger_mock] Pinger position updated to: ({self.pinger_x}, {self.pinger_y}, {self.pinger_z})"
            )

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.check_and_update_pinger_position()

            try:
                resp = self.get_model_state(self.robot_name, "world")
                if not resp.success:
                    rospy.logwarn_throttle(
                        5.0,
                        f"[sim_pinger_mock] Failed to get model state for '{self.robot_name}': {resp.status_message}",
                    )
                    rate.sleep()
                    continue

                pos = resp.pose.position
                ori = resp.pose.orientation
                quat = [ori.x, ori.y, ori.z, ori.w]

                dx = self.pinger_x - pos.x
                dy = self.pinger_y - pos.y
                dz = self.pinger_z - pos.z
                R = quaternion_matrix(quat)[:3, :3]
                v_world = np.array([dx, dy, dz])
                v_local = R.T.dot(v_world)

                angle = math.atan2(v_local[1], v_local[0])

                self.pub.publish(Float32(data=angle))

            except Exception as e:
                rospy.logwarn_throttle(
                    5.0, f"[sim_pinger_mock] Error calculating angle: {e}"
                )

            rate.sleep()


if __name__ == "__main__":
    try:
        mock = SimPingerMock()
        mock.run()
    except rospy.ROSInterruptException:
        pass
