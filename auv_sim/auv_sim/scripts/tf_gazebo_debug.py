#!/usr/bin/env python3

import rospy
import tf2_ros
from auv_msgs.srv import SpawnDebugCube, SpawnDebugCubeResponse
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from geometry_msgs.msg import Pose
import tf.transformations as tft
import numpy as np

class TFGazeboDebug:
    def __init__(self):
        rospy.init_node('tf_gazebo_debug')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.spawned_models = []
        
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        rospy.wait_for_service('/gazebo/get_model_state')

        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # for reference
        self.robot_name = rospy.get_param('~robot_name', 'taluy')
        self.robot_base_frame = rospy.get_param('~robot_base_frame', f'{self.robot_name}/base_link')
        
        self.spawn_srv = rospy.Service('/debug/spawn_cube', SpawnDebugCube, self.handle_spawn_cube)
        self.reset_srv = rospy.Service('/debug/reset', Trigger, self.handle_reset)
        
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
        rospy.loginfo("gazebo tf debugger ready")

    def pose_to_matrix(self, translation, rotation):
        T = tft.translation_matrix([translation.x, translation.y, translation.z])
        R = tft.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        return np.dot(T, R)

    def matrix_to_pose(self, matrix):
        pose = Pose()
        trans = tft.translation_from_matrix(matrix)
        quat = tft.quaternion_from_matrix(matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose

    def handle_spawn_cube(self, req):
        frame_name = req.frame_name
        try:
            trans_target = self.tf_buffer.lookup_transform('odom', frame_name, rospy.Time(0), rospy.Duration(1.0))
            
            gazebo_robot = self.get_model_state_proxy(model_name=self.robot_name, relative_entity_name='world')
            ros_robot = self.tf_buffer.lookup_transform('odom', self.robot_base_frame, rospy.Time(0), rospy.Duration(1.0))
            
            if not gazebo_robot.success:
                return SpawnDebugCubeResponse(success=False, message=f"gazebo error: {gazebo_robot.status_message}")

            M_target_in_odom = self.pose_to_matrix(trans_target.transform.translation, trans_target.transform.rotation)
            M_robot_in_world = self.pose_to_matrix(gazebo_robot.pose.position, gazebo_robot.pose.orientation)
            M_robot_in_odom = self.pose_to_matrix(ros_robot.transform.translation, ros_robot.transform.rotation)

            M_odom_in_world = np.dot(M_robot_in_world, np.linalg.inv(M_robot_in_odom))
            M_final = np.dot(M_odom_in_world, M_target_in_odom)
            
            pose = self.matrix_to_pose(M_final)
            
            model_name = f"debug_cube_{len(self.spawned_models)}_{frame_name.replace('/', '_')}"
            
            resp = self.spawn_model_proxy(
                model_name=model_name,
                model_xml=self.cube_sdf,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world"
            )
            
            if resp.success:
                self.spawned_models.append(model_name)
                return SpawnDebugCubeResponse(success=True, message=f"spawned {model_name} at gazebo {pose.position.x}, {pose.position.y}")
            else:
                return SpawnDebugCubeResponse(success=False, message=f"spawn error: {resp.status_message}")

        except Exception as e:
            return SpawnDebugCubeResponse(success=False, message=f"error: {str(e)}")

    def handle_reset(self, req):
        success = True
        messages = []
        for model_name in self.spawned_models:
            resp = self.delete_model_proxy(model_name)
            if not resp.success:
                success = False
                messages.append(f"failed to delete {model_name}: {resp.status_message}")
        
        self.spawned_models = []
        message = "success" if success else "\n".join(messages)
        return TriggerResponse(success=success, message=message)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = TFGazeboDebug()
        node.run()
    except rospy.ROSInterruptException:
        pass
