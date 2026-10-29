from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {'namespace': 'taluy', 'cmd_wrench_topic': 'cmd_wrench', 'odom_topic': 'odometry', 'wrench_topic': 'wrench', 'pre_hold_time': '3.0', 'settle_time': '5.0', 'record_time': '10.0', 'log_dir': '~/damping_tests', 'date_subdir': 'true', 'log_only_record': 'true'}
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=typ) for name, typ in [('cmd_wrench_topic', str), ('odom_topic', str), ('wrench_topic', str), ('pre_hold_time', float), ('settle_time', float), ('record_time', float), ('log_dir', str), ('date_subdir', bool), ('log_only_record', bool)]}
    return LaunchDescription([*(DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()), Node(package='auv_control', executable='damping_tester.py', name='damping_tester', namespace=LaunchConfiguration('namespace'), parameters=[params], output='screen')])
