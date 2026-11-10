from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    config = PathJoinSubstitution(
        [FindPackageShare("auv_bringup"), "config", ["ekf_", namespace, ".yaml"]]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="taluy"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("control_rate", default_value="20.0"),
            DeclareLaunchArgument(
                "localization_disabled_at_startup", default_value="true"
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="auv_localization_node",
                namespace=namespace,
                parameters=[
                    ParameterFile(config, allow_substs=True),
                    {
                        "disabled_at_startup": ParameterValue(
                            LaunchConfiguration("localization_disabled_at_startup"),
                            value_type=bool,
                        ),
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                    },
                ],
                remappings=[
                    ("odometry/filtered", "odometry"),
                    ("accel/filtered", "acceleration"),
                ],
                output="screen",
            ),
        ]
    )
