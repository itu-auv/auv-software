from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def nodes(context):
    namespace = LaunchConfiguration('namespace').perform(context)
    root = get_package_share_directory('auv_localization')
    control = LaunchConfiguration('control_config_file').perform(context)
    if not control:
        control = os.path.join(get_package_share_directory('auv_control'), 'config', namespace, 'default.yaml')
    with open(os.path.join(root, 'config', namespace, 'sensor_calibration_data.yaml')) as f:
        calibration = yaml.safe_load(f)
    sensor = calibration['sensors/external_pressure_sensor']
    dvl = calibration['sensors/dvl']['covariance']
    common = {'namespace': namespace, 'use_sim_time': LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')}
    velocity = LaunchConfiguration('dvl_velocity_topic').perform(context) or ('dvl/velocity' if namespace == 'taluy_mini' else 'sensors/dvl/velocity_raw')
    valid = LaunchConfiguration('dvl_is_valid_topic').perform(context) or ('dvl/is_valid' if namespace == 'taluy_mini' else 'sensors/dvl/is_valid')
    imu_calibration = LaunchConfiguration('imu_calibration_path').perform(context) or os.path.join(root, 'config', namespace, 'imu_calibration_data.yaml')
    depth_offset = LaunchConfiguration('depth_offset').perform(context)
    return [
        Node(package='auv_localization', executable='imu_odometry_node.py', name='imu_to_odom_node', namespace=namespace, output='screen', parameters=[common, {'imu_frame': namespace + '/base_link/imu_link', 'imu_calibration_path': imu_calibration}], remappings=[('imu/data', 'sensors/imu/data'), ('odom_imu', 'localization/odom_imu')]),
        Node(package='auv_localization', executable='pressure_odometry_node.py', name='pressure_to_odom_node', namespace=namespace, output='screen', parameters=[common, {'sensors.external_pressure_sensor.depth_offset': float(depth_offset) if depth_offset else float(sensor['depth_offset']), 'sensors.external_pressure_sensor.depth_covariance': float(sensor['depth_covariance']), 'env.pool_depth': float(LaunchConfiguration('pool_depth').perform(context))}], remappings=[('depth', 'sensors/external_pressure_sensor/depth'), ('odom_pressure', 'localization/odom_pressure')]),
        Node(package='auv_localization', executable='dvl_odometry_node.py', name='dvl_to_odom_node', namespace=namespace, output='screen', parameters=[control, common, {'cmdvel_tau': 0.2, **{'sensors.dvl.covariance.' + k: float(v) for k, v in dvl.items()}}], remappings=[('dvl/velocity_raw', velocity), ('dvl/is_valid', valid), ('odom_dvl', 'localization/odom_dvl')]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='taluy'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rate', default_value='20'),
        DeclareLaunchArgument('control_config_file', default_value=''),
        DeclareLaunchArgument('dvl_velocity_topic', default_value=''),
        DeclareLaunchArgument('dvl_is_valid_topic', default_value=''),
        DeclareLaunchArgument('pool_depth', default_value='2.134'),
        DeclareLaunchArgument('depth_offset', default_value=''),
        DeclareLaunchArgument('imu_calibration_path', default_value=''),
        OpaqueFunction(function=nodes),
    ])
