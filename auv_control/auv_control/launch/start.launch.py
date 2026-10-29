from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def nodes(context):
    namespace = LaunchConfiguration('namespace').perform(context)
    config = LaunchConfiguration('config_file').perform(context)
    if not config:
        config = os.path.join(get_package_share_directory('auv_control'), 'config', namespace, 'default.yaml')
    rate = float(LaunchConfiguration('rate').perform(context))
    timeout = float(LaunchConfiguration('command_timeout').perform(context))
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')
    return [
        Node(package='auv_control', executable='auv_control_node', name='auv_control_node', namespace=namespace, output='screen', parameters=[config, {'rate': rate, 'body_frame': namespace + '/base_link', 'use_sim_time': use_sim_time}]),
        Node(package='auv_control', executable='thruster_manager_node', name='thruster_manager_node', namespace=namespace, output='screen', parameters=[config, {'body_frame': namespace + '/base_link', 'frame_prefix': namespace, 'minimum_voltage_threshold': 13.5, 'low_voltage_duration': 3.0, 'use_sim_time': use_sim_time}], remappings=[('power', 'mainboard/power_sensor/power')]),
        Node(package='auv_control', executable='gravity_compensation_estimator.py', name='gravity_compensation_estimator', namespace=namespace, output='screen', parameters=[{'controller_reconfigure_server': 'auv_control_node', 'use_odometry_twist': namespace == 'taluy', 'use_sim_time': use_sim_time}]),
        Node(package='auv_control', executable='reference_pose_publisher.py', name='reference_pose_publisher_node', namespace=namespace, output='screen', parameters=[{'update_rate': rate, 'command_timeout': timeout, 'namespace': namespace, 'use_sim_time': use_sim_time}], remappings=[('align_frame/start', 'control/align_frame/start'), ('align_frame/cancel', 'control/align_frame/cancel'), ('set_object_transform', 'map/set_object_transform')]),
        Node(package='auv_control', executable='battery_monitor_node.py', name='battery_monitor_node', namespace=namespace, output='screen', parameters=[{'minimum_voltage_threshold': 13.5, 'voltage_warn_threshold': 15.0, 'power_message_timeout': 5, 'min_log_interval': 1.0, 'max_log_interval': 120.0, 'use_sim_time': use_sim_time}], remappings=[('power', 'mainboard/power_sensor/power')]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='taluy'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rate', default_value='20'),
        DeclareLaunchArgument('use_gui', default_value='false'),
        DeclareLaunchArgument('start_state_publisher', default_value='false'),
        DeclareLaunchArgument('command_timeout', default_value='0.1'),
        DeclareLaunchArgument('config_file', default_value=''),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('auv_description'), 'launch', 'start_state_publisher.launch.py'])), condition=IfCondition(LaunchConfiguration('start_state_publisher')), launch_arguments={'namespace': LaunchConfiguration('namespace'), 'use_gui': LaunchConfiguration('use_gui')}.items()),
        OpaqueFunction(function=nodes),
    ])
