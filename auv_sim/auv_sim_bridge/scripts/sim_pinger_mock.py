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
        self.pinger_name = rospy.get_param("~pinger_name", "pinger_cube")

        self.pinger_start_x = rospy.get_param("~pinger_x", 10.0)
        self.pinger_start_y = rospy.get_param("~pinger_y", 0.0)
        self.pinger_start_z = rospy.get_param("~pinger_z", -1.0)

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
            f"[sim_pinger_mock] Initialized. Pinger position: ({self.pinger_start_x}, {self.pinger_start_y}, {self.pinger_start_z}) "
            f"Publishing on topic '{self.topic_name}' at {self.publish_rate} Hz."
        )

    def spawn_pinger_cube(self):
        sdf_xml = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{self.pinger_name}">
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
        pose.position.x = self.pinger_start_x
        pose.position.y = self.pinger_start_y
        pose.position.z = 0.0
        pose.orientation.w = 1.0

        try:
            resp = self.spawn_model(
                model_name=self.pinger_name,
                model_xml=sdf_xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world",
            )
            if not resp.success:
                rospy.logwarn(
                    f"[sim_pinger_mock] Failed to spawn pinger cube: {resp.status_message}"
                )
        except Exception as e:
            rospy.logwarn(f"[sim_pinger_mock] Exception during Gazebo spawn: {e}")

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            try:
                robot_resp = self.get_model_state(self.robot_name, "world")
                if not robot_resp.success:
                    rospy.logwarn_throttle(
                        5.0,
                        f"[sim_pinger_mock] Failed to get model state for '{self.robot_name}': {resp.status_message}",
                    )
                    rate.sleep()
                    continue
                pinger_resp = self.get_model_state(self.pinger_name, "world")
                if not pinger_resp.success:
                    rospy.logwarn_throttle(
                        5.0,
                        f"[sim_pinger_mock] Failed to get model state for '{self.robot_name}': {resp.status_message}",
                    )
                    rate.sleep()
                    continue

                pos = robot_resp.pose.position
                ori = robot_resp.pose.orientation
                quat = [ori.x, ori.y, ori.z, ori.w]

                pinger = pinger_resp.pose.position

                dx = pinger.x - pos.x
                dy = pinger.y - pos.y
                dz = pinger.z - pos.z
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
