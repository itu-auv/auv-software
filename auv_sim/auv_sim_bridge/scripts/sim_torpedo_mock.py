#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SpawnModel, GetModelState, SetModelState
from geometry_msgs.msg import Pose, Quaternion, Twist
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_matrix,
)
import rospkg
import os


def handle_torpedo_launch(req):
    try:
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")

        spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        # Get package path dynamically
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("auv_sim_description")

        # Define torpedo models with fallback options
        torpedo_models = [
            ("torpedo_one.sdf", "torpedo"),
            ("torpedo_two.sdf", "torpedo"),
        ]

        vehicle_state = get_model_state("taluy", "world")

        # Convert vehicle quaternion to roll, pitch, yaw angles
        vehicle_orientation = vehicle_state.pose.orientation
        quat = [
            vehicle_orientation.x,
            vehicle_orientation.y,
            vehicle_orientation.z,
            vehicle_orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)

        # Torpedo offset in vehicle base_link frame
        torpedo_offset = [
            0.30,
            -0.14,
            -0.05,
        ]  # [x, y, z] position relative to base_link

        # Get vehicle rotation matrix and convert torpedo offset to world frame
        rot_matrix = quaternion_matrix(quat)[:3, :3]  # 3x3 transformation matrix
        torpedo_position_world = rot_matrix.dot(
            torpedo_offset
        )  # Convert to world frame

        # Torpedo initial position in world frame
        pose = Pose()
        pose.position.x = vehicle_state.pose.position.x + torpedo_position_world[0]
        pose.position.y = vehicle_state.pose.position.y + torpedo_position_world[1]
        pose.position.z = vehicle_state.pose.position.z + torpedo_position_world[2]

        # Align torpedo orientation with vehicle orientation
        torpedo_quat = quaternion_from_euler(roll, pitch, yaw)
        pose.orientation = Quaternion(*torpedo_quat)

        # Initial torpedo velocity in vehicle base_link frame
        initial_velocity = [1.0, 0.0, 0.2]  # Forward movement relative to vehicle
        deceleration_rate = 0.2  # Deceleration rate (m/s²)

        # Convert velocity to world frame
        world_velocity = rot_matrix.dot(initial_velocity)

        # Set torpedo velocity
        twist = Twist()
        twist.linear.x = world_velocity[0] * (
            1 - deceleration_rate
        )  # Gradual x-axis velocity reduction
        twist.linear.y = world_velocity[1]
        twist.linear.z = world_velocity[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        # Try spawning torpedoes with fallback
        for model_file, model_name in torpedo_models:
            try:
                model_path = os.path.join(
                    pkg_path, "models", "robosub_torpedo", model_file
                )
                with open(model_path, "r") as f:
                    model_xml = f.read()

                # Try to spawn the model
                spawn_response = spawn_model(model_name, model_xml, "", pose, "world")

                if spawn_response.success:
                    # Set torpedo model state with velocity
                    from gazebo_msgs.msg import ModelState

                    set_model_state(
                        ModelState(
                            model_name=model_name,
                            pose=pose,
                            twist=twist,
                            reference_frame="world",
                        )
                    )
                    rospy.loginfo(f"Torpedo launched successfully using {model_file}!")
                    return TriggerResponse(
                        success=True,
                        message=f"Torpedo successfully launched using {model_file}.",
                    )
                else:
                    rospy.logwarn(
                        f"Failed to spawn torpedo with {model_file}, trying next model if available."
                    )
            except Exception as e:
                rospy.logwarn(f"Failed to launch torpedo with {model_file}: {str(e)}")
                continue

        return TriggerResponse(
            success=False, message="Both torpedo models failed to launch."
        )

    except Exception as e:
        rospy.logerr(f"Failed to launch torpedo: {e}")
        return TriggerResponse(success=False, message=str(e))


def launch_torpedo_server():
    rospy.init_node("launch_torpedo_server")
    rospy.Service("/taluy/actuators/torpedo/launch", Trigger, handle_torpedo_launch)
    rospy.loginfo("Torpedo launch service is ready.")
    rospy.spin()


if __name__ == "__main__":
    launch_torpedo_server()
