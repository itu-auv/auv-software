from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="taluy"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="auv_navigation",
                executable="follow_path_action_server.py",
                name="follow_path_action_server",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "source_frame": [namespace, "/base_link"],
                        "use_sim_time": use_sim_time,
                    }
                ],
            ),
            Node(
                package="auv_navigation",
                executable="path_planner_node.py",
                name="path_planner_node",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "robot_frame": [namespace, "/base_link"],
                        "use_sim_time": use_sim_time,
                    }
                ],
            ),
        ]
    )
