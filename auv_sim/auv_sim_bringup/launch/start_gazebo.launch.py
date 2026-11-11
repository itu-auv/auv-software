"""Simulation entry point (ROS 2 + Gazebo gz-sim); pool is the legacy default world."""

from pathlib import Path
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnShutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def start(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    world = LaunchConfiguration("world").perform(context)
    if namespace not in ("taluy", "taluy_mini") or world not in (
        "pool",
        "taluy_harmonic",
    ):
        raise ValueError(
            "supported namespaces: taluy, taluy_mini; worlds: pool, taluy_harmonic"
        )
    premap = LaunchConfiguration("premap").perform(context)
    if premap == "auto":
        # premaps are in odom, whose origin is the start pose
        moved = any(LaunchConfiguration(n).perform(context) for n in ("x", "y", "yaw"))
        at_gate = LaunchConfiguration("start").perform(context) == "gate" and not moved
        premap = "sim_pool" if world == "pool" and at_gate else "none"
    temporary = tempfile.TemporaryDirectory(prefix="auv-sim-chain-")
    root = Path(temporary.name)
    bridge_share = Path(get_package_share_directory("auv_sim_bridge"))
    config = yaml.safe_load(
        (bridge_share / "config" / namespace / "control_config.yaml").read_text()
    )
    # Explicit neutral-buoyancy simulation assumption.
    # PID gains, limits, allocation and control/model mathematics are preserved.
    config["/**/auv_control_node"]["ros__parameters"]["gravity_compensation_z"] = 0.0
    (root / "control.yaml").write_text(yaml.safe_dump(config))
    (root / "imu.yaml").write_text("drift: [0.0, 0.0, 0.0]\n")
    ekf = (
        Path(get_package_share_directory("auv_sim_bringup"))
        / "config"
        / f"ekf_{namespace}.yaml"
    ).read_text()
    ekf = yaml.safe_load(
        ekf.replace("$(arg control_rate)", "20.0").replace(
            "$(var namespace)", namespace
        )
    )
    ekf["/**/auv_localization_node"]["ros__parameters"].update(
        use_sim_time=True, disabled_at_startup=False
    )
    (root / "ekf.yaml").write_text(yaml.safe_dump(ekf))

    def include(package, filename, arguments):
        path = Path(get_package_share_directory(package)) / "launch" / filename
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(path)), launch_arguments=arguments.items()
        )

    def include_xml(package, filename, arguments):
        path = Path(get_package_share_directory(package)) / "launch" / filename
        return IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(path)), launch_arguments=arguments.items()
        )

    # What the ROS 1 start_gazebo.launch started on top of the simulator itself
    # (start_bridge.launch -> start_navigation.launch, parameter server, monitor, GUIs).
    description_share = Path(get_package_share_directory(f"{namespace}_description"))
    application = [
        include_xml(
            "auv_mapping",
            "start.launch",
            {
                "namespace": namespace,
                "control_rate": "20.0",
                "plane_fitter_sync_queue_size": "90",
                "premap": premap,
            },
        ),
        include(
            "auv_navigation",
            "start.launch.py",
            {"namespace": namespace, "use_sim_time": "true"},
        ),
        include_xml("auv_vision", "start.launch", {"namespace": namespace}),
        include_xml("auv_bringup", "param_server.launch", {"namespace": namespace}),
        Node(
            package="auv_smach",
            executable="smach_monitor_node.py",
            name="smach_monitor_node",
            namespace=namespace,
            output="screen",
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                str(
                    Path(get_package_share_directory("auv_teleop"))
                    / "launch"
                    / "start_taluy_gui.launch"
                )
            ),
            launch_arguments={"namespace": namespace, "use_sim_time": "true"}.items(),
            condition=IfCondition(LaunchConfiguration("use_taluy_gui")),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz",
            arguments=["-d", str(description_share / "rviz" / "araba.rviz")],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]

    camera_nodes = []
    for camera in (
        ["front", "bottom", "torpedo"] if namespace == "taluy" else ["front", "bottom"]
    ):
        camera_namespace = f"/{namespace}/cameras/cam_{camera}"
        camera_nodes.extend(
            [
                Node(
                    package="auv_sim_bridge",
                    executable="gz_sensor_bridge",
                    name=f"{camera}_application_camera_bridge",
                    parameters=[
                        {
                            "topics": [
                                f"/model/{namespace}/{camera}_camera/image {camera_namespace}/image_raw image",
                                f"/model/{namespace}/{camera}_camera/camera_info {camera_namespace}/camera_info camera_info",
                            ]
                        }
                    ],
                ),
                Node(
                    package="image_proc",
                    executable="rectify_node",
                    namespace=camera_namespace,
                    parameters=[{"use_sim_time": True}],
                    remappings=[
                        ("image", "image_raw"),
                        ("image_rect", "image_rect_color"),
                    ],
                ),
            ]
        )
    if namespace == "taluy":
        # RealSense: colour and depth images, and the cloud the ROS 1 plugin published
        gz, ros = f"/model/{namespace}/realsense", f"/{namespace}/camera"
        camera_nodes.extend(
            [
                Node(
                    package="auv_sim_bridge",
                    executable="gz_sensor_bridge",
                    name="realsense_bridge",
                    parameters=[
                        {
                            "topics": [
                                f"{gz}/image {ros}/color/image_raw image",
                                f"{gz}/depth_image {ros}/depth/image_raw image",
                                f"{gz}/camera_info {ros}/color/camera_info,{ros}/depth/camera_info camera_info",
                            ]
                        }
                    ],
                ),
                Node(
                    package="depth_image_proc",
                    executable="point_cloud_xyz_node",
                    name="realsense_cloud",
                    remappings=[
                        ("image_rect", f"{ros}/depth/image_raw"),
                        ("camera_info", f"{ros}/depth/camera_info"),
                        ("points", "/points_filtered"),
                    ],
                ),
            ]
        )
    return (
        [SetParameter(name="use_sim_time", value=True)]
        + camera_nodes
        + [
            include(
                "auv_sim_bringup",
                "start_harmonic.launch.py",
                {
                    "namespace": namespace,
                    "world": world,
                    "gui": LaunchConfiguration("gui"),
                    "initial_z": LaunchConfiguration("initial_z"),
                    "clock_period": LaunchConfiguration("clock_period"),
                    "start": LaunchConfiguration("start"),
                    "x": LaunchConfiguration("x"),
                    "y": LaunchConfiguration("y"),
                    "yaw": LaunchConfiguration("yaw"),
                },
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="simulation_spawn_bridge",
                arguments=[
                    f"/world/{world}/create@ros_gz_interfaces/srv/SpawnEntity",
                    f"/world/{world}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
                ],
            ),
            Node(
                package="auv_sim_bridge",
                executable="sim_hardware_mocks.py",
                namespace=namespace,
                parameters=[
                    {
                        "world": world,
                        "use_sim_time": True,
                        "pinger_mode": LaunchConfiguration("pinger_mode"),
                    }
                ],
                output="screen",
            ),
            include(
                "auv_localization",
                "start.launch.py",
                {
                    "namespace": namespace,
                    "use_sim_time": "true",
                    "control_config_file": str(root / "control.yaml"),
                    "depth_offset": "0.0",
                    "pool_depth": "20.0",
                    "imu_calibration_path": str(root / "imu.yaml"),
                    "dvl_velocity_topic": "sensors/dvl/velocity_raw",
                    "dvl_is_valid_topic": "sensors/dvl/is_valid",
                },
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="auv_localization_node",
                namespace=namespace,
                parameters=[str(root / "ekf.yaml")],
                remappings=[
                    ("odometry/filtered", "odometry"),
                    ("accel/filtered", "acceleration"),
                    ("enable", "auv_localization_node/enable"),
                    ("toggle", "auv_localization_node/toggle"),
                ],
                output="screen",
            ),
            include(
                "auv_control",
                "start.launch.py",
                {
                    "namespace": namespace,
                    "use_sim_time": "true",
                    "config_file": str(root / "control.yaml"),
                },
            ),
        ]
        + application
        + [
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[
                        OpaqueFunction(function=lambda context: temporary.cleanup())
                    ]
                )
            ),
        ]
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="taluy"),
            DeclareLaunchArgument("world", default_value="pool"),
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("initial_z", default_value="-0.5"),
            DeclareLaunchArgument(
                "clock_period",
                default_value="0.01",
                description="simulation seconds between /clock messages; raise it on a slow machine",
            ),
            DeclareLaunchArgument(
                "premap",
                default_value="auto",
                description="expected object positions (auv_mapping/config/premaps); "
                "auto: sim_pool when starting at the pool's gate, else none",
            ),
            DeclareLaunchArgument(
                "start",
                default_value="gate",
                description="pool world: start in front of gate, torpedo, bin, octagon, octagon_table or slalom",
            ),
            DeclareLaunchArgument(
                "x", default_value="", description="overrides the start pose"
            ),
            DeclareLaunchArgument("y", default_value=""),
            DeclareLaunchArgument("yaw", default_value=""),
            DeclareLaunchArgument("pinger_mode", default_value="torpedo"),
            DeclareLaunchArgument(
                "use_taluy_gui", default_value=LaunchConfiguration("gui")
            ),
            DeclareLaunchArgument("use_rviz", default_value=LaunchConfiguration("gui")),
            OpaqueFunction(function=start),
        ]
    )
