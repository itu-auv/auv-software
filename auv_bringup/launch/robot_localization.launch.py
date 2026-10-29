from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arguments = ['namespace', 'control_rate', 'localization_disabled_at_startup', 'use_sim_time']
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='taluy'),
        DeclareLaunchArgument('control_rate', default_value='20.0'),
        DeclareLaunchArgument('localization_disabled_at_startup', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('auv_bringup'), 'launch', 'inc', 'localization', 'start_robot_localization.launch.py'])), launch_arguments={name: LaunchConfiguration(name) for name in arguments}.items()),
    ])
