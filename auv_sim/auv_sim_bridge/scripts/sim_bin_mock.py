#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel, GetModelState
from geometry_msgs.msg import Pose
import rospkg
import os


def handle_drop_ball(req):
    try:
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/get_model_state")

        spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("auv_sim_description")

        ball_models = [("ball.sdf", "sphere_one"), ("ball.sdf", "sphere_two")]

        vehicle_state = get_model_state("taluy", "world")
        pose = Pose()
        pose.position.x = vehicle_state.pose.position.x + 0.07
        pose.position.y = vehicle_state.pose.position.y - 0.1
        pose.position.z = vehicle_state.pose.position.z - 0.15

        for ball_file, ball_name in ball_models:
            model_path = os.path.join(pkg_path, "models", "robosub_bin", ball_file)
            with open(model_path, "r") as f:
                model_xml = f.read()

            spawn_response = spawn_model(ball_name, model_xml, "", pose, "world")
            if spawn_response.success:
                rospy.loginfo(f"{ball_file} dropped successfully!")
                return TriggerResponse(
                    success=True, message=f"{ball_file} successfully dropped."
                )
            else:
                rospy.logwarn(
                    f"Failed to drop {ball_file}, trying the next one if available."
                )

        return TriggerResponse(
            success=False, message="Both ball_one and ball_two could not be spawned."
        )

    except Exception as e:
        rospy.logerr(f"Failed to drop ball: {e}")
        return TriggerResponse(success=False, message=str(e))


def drop_ball_server():
    rospy.init_node("drop_ball_server")
    rospy.Service("actuators/ball_dropper/drop", Trigger, handle_drop_ball)
    rospy.loginfo("Drop ball service is ready.")
    rospy.spin()


if __name__ == "__main__":
    drop_ball_server()
