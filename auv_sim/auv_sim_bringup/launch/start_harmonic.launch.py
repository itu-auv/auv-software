"""Taluy and Mini physics/sensors using native Harmonic systems."""

import os
from pathlib import Path
import subprocess
import tempfile
import math
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    EmitEvent,
)
from launch.event_handlers import OnShutdown, OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro

ET.register_namespace("gz", "http://gazebosim.org/schemas")


def start(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    if namespace not in ("taluy", "taluy_mini"):
        raise RuntimeError("namespace must be taluy or taluy_mini")
    description = Path(get_package_share_directory(namespace + "_sim_description"))
    robot = xacro.process_file(
        str(description / f"urdf/{namespace}_sim.urdf.xacro")
    ).toxml()
    # Mini's legacy base inertia is not positive definite. Simulation-only sphere proxy.
    if namespace == "taluy_mini":
        urdf_tree = ET.fromstring(robot)
        inertia = urdf_tree.find("link[@name='taluy_mini/base_link']/inertial/inertia")
        total_mass = sum(
            float(e.get("value")) for e in urdf_tree.findall(".//inertial/mass")
        )
        radius = (3 * total_mass / (4 * math.pi * 1028)) ** (1 / 3)
        mass = float(
            urdf_tree.find("link[@name='taluy_mini/base_link']/inertial/mass").get(
                "value"
            )
        )
        for key in inertia.attrib:
            inertia.set(
                key,
                str(0.4 * mass * radius**2) if key in ("ixx", "iyy", "izz") else "0",
            )
        robot = ET.tostring(urdf_tree, encoding="unicode")
    temporary = tempfile.TemporaryDirectory(prefix="auv-harmonic-")
    urdf = Path(temporary.name) / "taluy.urdf"
    urdf.write_text(robot)
    sdf = subprocess.run(
        ["gz", "sdf", "-p", str(urdf)], check=True, capture_output=True, text=True
    ).stdout
    sdf = sdf.replace("<sdf ", '<sdf xmlns:gz="http://gazebosim.org/schemas" ', 1)
    ET.register_namespace("gz", "http://gazebosim.org/schemas")
    if namespace == "taluy_mini":
        tree = ET.fromstring(sdf)
        entity = tree.find("model")
        base_link = entity.find("link[@name='taluy_mini/base_link']")
        total_mass = sum(float(e.text) for e in entity.findall(".//inertial/mass"))
        radius = (3 * total_mass / (4 * math.pi * 1028)) ** (1 / 3)
        collision = ET.SubElement(
            base_link, "collision", name="ideal_neutral_displacement"
        )
        ET.SubElement(collision, "pose").text = "0 0 0.2 0 0 0"
        ET.SubElement(
            ET.SubElement(ET.SubElement(collision, "geometry"), "sphere"), "radius"
        ).text = str(radius)
        ET.SubElement(
            ET.SubElement(ET.SubElement(collision, "surface"), "contact"),
            "collide_bitmask",
        ).text = "0"
        sdf = ET.tostring(tree, encoding="unicode")
    entity = ET.fromstring(sdf).find("model")
    # the conversion from URDF drops <surface>, and with it the bitmask that keeps
    # the buoyancy sphere from touching things
    for collision in entity.iter("collision"):
        if "buoyancy_volume" in collision.get("name", ""):
            contact = ET.SubElement(ET.SubElement(collision, "surface"), "contact")
            ET.SubElement(contact, "collide_bitmask").text = "0"
    pose = entity.find("pose")
    if pose is None:
        pose = ET.SubElement(entity, "pose")
    # start poses of the ROS 1 launch (its `objects:=<task>,...` picked them by task)
    starts = {
        "gate": (-7.0, -15.5, 0.0),
        "torpedo": (0.8, -3.8, 0.33),
        "bin": (5.0, -6.6, -0.9),
        "octagon": (7.4, -7.0, 0.33),
        "octagon_table": (13.0, -4.25, 0.0),
        "slalom": (0.15, -13.3, -0.12),
    }
    start = (
        starts[LaunchConfiguration("start").perform(context)]
        if LaunchConfiguration("world").perform(context) == "pool"
        else (0.0, 0.0, 0.0)
    )
    x, y, yaw = (
        float(LaunchConfiguration(name).perform(context) or default)
        for name, default in zip(("x", "y", "yaw"), start)
    )
    pose.text = (
        f"{x} {y} {float(LaunchConfiguration('initial_z').perform(context))} 0 0 {yaw}"
    )
    model = ET.tostring(entity, encoding="unicode")
    world_name = LaunchConfiguration("world").perform(context)
    if world_name not in ("taluy_harmonic", "pool"):
        raise RuntimeError("Unknown Harmonic world: " + world_name)
    sim_share = Path(get_package_share_directory("auv_sim_description"))
    filename = (
        "taluy_harmonic.sdf"
        if world_name == "taluy_harmonic"
        else world_name + "_harmonic.sdf"
    )
    world = (sim_share / "worlds" / filename).read_text()
    world = world.replace(
        "taluy::taluy/base_link", f"{namespace}::{namespace}/base_link"
    )
    world_path = Path(temporary.name) / "taluy_harmonic.sdf"
    world_path.write_text(world.replace("</world>", model + "</world>"))
    plugin_dir = str(Path(get_package_prefix("auv_sim_bridge")) / "lib")
    resource_dir = os.pathsep.join(
        dict.fromkeys(
            str(Path(get_package_share_directory(package)).parent)
            for package in (namespace + "_description", "auv_common_descriptions")
        )
    )
    gui = LaunchConfiguration("gui").perform(context).lower() in ("true", "1")
    command = ["gz", "sim", "-r", "-v", "3", str(world_path)]
    if not gui:
        command += ["-s", "--headless-rendering"]
    bridge = (
        Path(get_package_share_directory("auv_sim_bridge"))
        / "launch/pwm_and_sensor_bridge.launch.py"
    )
    simulator = ExecuteProcess(cmd=command, output="screen")
    return [
        RegisterEventHandler(
            OnProcessExit(
                target_action=simulator,
                on_exit=[EmitEvent(event=Shutdown(reason="Gazebo exited"))],
            )
        ),
        SetEnvironmentVariable(
            "GZ_SIM_SYSTEM_PLUGIN_PATH",
            plugin_dir + os.pathsep + os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", ""),
        ),
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            str(sim_share / "models")
            + os.pathsep
            + resource_dir
            + os.pathsep
            + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ),
        simulator,
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {
                    "robot_description": ParameterValue(robot, value_type=str),
                    "use_sim_time": True,
                }
            ],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(bridge)),
            launch_arguments={"namespace": namespace, "world": world_name}.items(),
        ),
        Node(
            package="auv_sim_bridge",
            executable="clock_bridge",
            parameters=[
                {"period": float(LaunchConfiguration("clock_period").perform(context))}
            ],
        ),
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    OpaqueFunction(function=lambda context: temporary.cleanup())
                ]
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="taluy"),
            DeclareLaunchArgument("world", default_value="taluy_harmonic"),
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("initial_z", default_value="-0.5"),
            DeclareLaunchArgument(
                "clock_period",
                default_value="0.01",
                description="simulation seconds between /clock messages; raise it on a slow machine",
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
            OpaqueFunction(function=start),
        ]
    )
