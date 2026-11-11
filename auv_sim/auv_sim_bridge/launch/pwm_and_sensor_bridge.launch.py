from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("auv_sim_bridge"))
    curve_path = package_share / "config" / "taluy" / "t200_pwm_force.yaml"
    curve = yaml.safe_load(curve_path.read_text())["pwm_to_force"]
    environment = yaml.safe_load(
        (
            Path(get_package_share_directory("auv_sim_description"))
            / "config"
            / "environment.yaml"
        ).read_text()
    )
    namespace = LaunchConfiguration("namespace")
    world = LaunchConfiguration("world")

    def bridge_arg(*parts):
        return [
            TextSubstitution(text=part) if isinstance(part, str) else part
            for part in parts
        ]

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="taluy_command_bridge",
        arguments=[
            bridge_arg("/world/", world, "/control@ros_gz_interfaces/srv/ControlWorld"),
        ]
        + [
            bridge_arg(
                "/model/",
                namespace,
                "/joint/",
                namespace,
                f"/thruster_{index}_propeller_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
            )
            for index in range(8)
        ],
        output="screen",
    )

    def sensor_bridge(context):
        ns = namespace.perform(context)
        model = f"/model/{ns}"
        return [
            Node(
                package="auv_sim_bridge",
                executable="gz_sensor_bridge",
                name="taluy_sensor_bridge",
                parameters=[
                    {
                        "topics": [
                            f"{model}/imu /{ns}/sensors/imu/data imu",
                            f"{model}/pressure {model}/pressure fluid_pressure",
                            f"{model}/odometry {model}/odometry odometry",
                            f"{model}/dvl/velocity {model}/dvl/velocity dvl",
                        ]
                    }
                ],
                output="screen",
            )
        ]

    pwm_adapter = Node(
        package="auv_sim_bridge",
        executable="sim_pwm_to_force.py",
        name="sim_pwm_to_force",
        parameters=[
            {
                "namespace": namespace,
                "use_sim_time": True,
                "pwm_curve": curve["pwm"],
                "force_curve": curve["force"],
            }
        ],
        output="screen",
    )
    dvl_adapter = Node(
        package="auv_sim_bridge",
        executable="sim_dvl_adapter.py",
        name="sim_dvl_adapter",
        parameters=[{"namespace": namespace}],
        output="screen",
    )
    pressure_adapter = Node(
        package="auv_sim_bridge",
        executable="sim_pressure_depth_adapter.py",
        name="sim_pressure_depth_adapter",
        parameters=[
            {
                "namespace": namespace,
                "standard_pressure": float(environment["/env/standard_pressure"]),
                "gravity": float(environment["/env/gravity"]),
                "density": float(environment["/env/density"]),
            }
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="taluy"),
            DeclareLaunchArgument("world", default_value="taluy_harmonic"),
            bridge,
            OpaqueFunction(function=sensor_bridge),
            pwm_adapter,
            dvl_adapter,
            pressure_adapter,
        ]
    )
