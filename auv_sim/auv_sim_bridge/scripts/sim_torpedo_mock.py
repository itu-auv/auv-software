#!/usr/bin/env python3

import rospy
import threading
import time
import os
import rospkg
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel, GetModelState, SetModelState
from geometry_msgs.msg import Pose, Quaternion, Twist
from gazebo_msgs.msg import ModelState
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_matrix,
)

TORPEDO_MODELS = {
    1: ("torpedo.sdf", "torpedo_one"),
    2: ("torpedo.sdf", "torpedo_two"),
}

TORPEDO_OFFSETS = {
    1: [0.30, -0.14, -0.05],
    2: [0.30, -0.14, -0.05],
}


def get_torpedo_model_details(torpedo_id):
    """Retrieve torpedo model details based on torpedo ID."""
    if torpedo_id not in TORPEDO_MODELS:
        raise ValueError(f"Invalid torpedo ID: {torpedo_id}")
    return TORPEDO_MODELS[torpedo_id]


def get_vehicle_orientation_and_position():
    """Retrieve vehicle's current orientation and position."""
    get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    vehicle_state = get_model_state("taluy", "world")
    vehicle_orientation = vehicle_state.pose.orientation
    quat = [
        vehicle_orientation.x,
        vehicle_orientation.y,
        vehicle_orientation.z,
        vehicle_orientation.w,
    ]
    roll, pitch, yaw = euler_from_quaternion(quat)
    return quat, roll, pitch, yaw, vehicle_state


def calculate_torpedo_pose(vehicle_state, quat, roll, pitch, yaw, torpedo_id):
    """Calculate torpedo's initial pose based on vehicle state."""
    torpedo_offset = TORPEDO_OFFSETS[torpedo_id]
    rot_matrix = quaternion_matrix(quat)[:3, :3]
    torpedo_position_world = rot_matrix.dot(torpedo_offset)

    pose = Pose()
    pose.position.x = vehicle_state.pose.position.x + torpedo_position_world[0]
    pose.position.y = vehicle_state.pose.position.y + torpedo_position_world[1]
    pose.position.z = vehicle_state.pose.position.z + torpedo_position_world[2]

    torpedo_quat = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation = Quaternion(*torpedo_quat)
    return pose, rot_matrix


def calculate_torpedo_twist(rot_matrix):
    """Calculate initial torpedo twist (velocity)."""
    initial_velocity = [1.0, 0.0, 0.35]  # Forward movement relative to vehicle
    deceleration_rate = 0.2
    world_velocity = rot_matrix.dot(initial_velocity)

    twist = Twist()
    twist.linear.x = world_velocity[0] * (1 - deceleration_rate)
    twist.linear.y = world_velocity[1]
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0

    return twist, world_velocity


def spawn_torpedo_model(model_name, pose, torpedo_id):
    """Spawn torpedo model in Gazebo."""
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("auv_sim_description")
    model_file, _ = get_torpedo_model_details(torpedo_id)

    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    model_path = os.path.join(pkg_path, "models", "robosub_torpedo", model_file)

    with open(model_path, "r") as f:
        model_xml = f.read()

    return spawn_model(model_name, model_xml, "", pose, "world")


def update_torpedo_z_velocity(model_name, torpedo_id, world_velocity):
    """Update torpedo's Z velocity in a separate thread."""
    get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    start_time = time.time()

    def _update_z_velocity():
        try:
            current_state = get_model_state(model_name, "world")
            delayed_twist = Twist()
            delayed_twist.linear.x = current_state.twist.linear.x
            delayed_twist.linear.y = current_state.twist.linear.y
            delayed_twist.linear.z = world_velocity[2]
            delayed_twist.angular = current_state.twist.angular
            delayed_twist.angular.x = -3.0

            set_model_state(
                ModelState(
                    model_name=model_name,
                    pose=current_state.pose,
                    twist=delayed_twist,
                    reference_frame="world",
                )
            )
            rospy.loginfo(f"Torpedo {torpedo_id} z velocity updated.")
            if time.time() - start_time < 2.0:
                threading.Timer(0.1, _update_z_velocity).start()
        except Exception as e:
            rospy.logerr(f"Failed to update torpedo {torpedo_id} z velocity: {str(e)}")

    threading.Timer(0.6, _update_z_velocity).start()


def handle_torpedo_launch(req, torpedo_id):
    """Main function to handle torpedo launch."""
    try:
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")

        model_file, model_name = get_torpedo_model_details(torpedo_id)
        quat, roll, pitch, yaw, vehicle_state = get_vehicle_orientation_and_position()

        pose, rot_matrix = calculate_torpedo_pose(
            vehicle_state, quat, roll, pitch, yaw, torpedo_id
        )
        twist, world_velocity = calculate_torpedo_twist(rot_matrix)

        spawn_response = spawn_torpedo_model(model_name, pose, torpedo_id)

        if spawn_response.success:
            set_model_state = rospy.ServiceProxy(
                "/gazebo/set_model_state", SetModelState
            )
            set_model_state(
                ModelState(
                    model_name=model_name,
                    pose=pose,
                    twist=twist,
                    reference_frame="world",
                )
            )
            rospy.loginfo(f"Torpedo {torpedo_id} launched successfully!")

            update_torpedo_z_velocity(model_name, torpedo_id, world_velocity)

            return TriggerResponse(
                success=True, message=f"Torpedo {torpedo_id} successfully launched."
            )
        else:
            rospy.logwarn(f"Failed to spawn torpedo {torpedo_id}.")
            return TriggerResponse(
                success=False, message=f"Torpedo {torpedo_id} launch failed."
            )

    except Exception as e:
        rospy.logerr(f"Torpedo {torpedo_id} launch error: {e}")
        return TriggerResponse(success=False, message=str(e))


def launch_torpedo_server():
    rospy.init_node("launch_torpedo_server")
    for i in [1, 2]:
        rospy.Service(
            f"actuators/torpedo_{i}/launch",
            Trigger,
            lambda req, id=i: handle_torpedo_launch(req, id),
        )
    rospy.loginfo("Torpedo launch services are ready.")
    rospy.spin()


if __name__ == "__main__":
    launch_torpedo_server()
