#!/usr/bin/env python3

"""Simulation-only tracker with the same ROS interface as tracker_node.py."""

import os

import rclpy
from auv_common_lib import ros
from auv_common_lib.ros import get_param, latched_qos, get_node, init_node
from ament_index_python.packages import get_package_share_directory
import tf2_ros
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from auv_sim_vision.bbox import GazeboInterface, SimCamera, load_config


class SimTrackerNode:
    def __init__(self):
        init_node("sim_tracker_node")

        namespace = get_param(get_node(), "~namespace", "taluy")
        camera = get_param(get_node(), "~camera")
        default_config = os.path.join(
            get_package_share_directory("auv_sim_vision"),
            "config",
            "bbox_objects.yaml",
        )
        config_path = get_param(get_node(), "~config", default_config)
        self.enabled = get_param(get_node(), "~enabled", True)

        objects, camera_configs = load_config(config_path, namespace)
        if camera not in camera_configs:
            raise RuntimeError(
                f"Unknown simulation camera '{camera}'; "
                f"available cameras: {sorted(camera_configs)}"
            )

        camera_config = camera_configs[camera]
        result_topic = get_param(
            get_node(), "~result_topic", camera_config["result_topic"]
        )
        result_image_topic = get_param(
            get_node(),
            "~result_image_topic",
            camera_config["image_out_topic"] + "/compressed",
        )

        self.enabled_pub = get_node().create_publisher(Bool, "~/enabled", latched_qos())
        self.enable_srv = get_node().create_service(
            SetBool, "~/enable", self.enable_callback
        )
        ##sim part
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, get_node())
        self.gazebo = GazeboInterface(robot_name=namespace)
        camera_objects = [obj for obj in objects if camera in obj.cameras]
        self.camera = SimCamera(
            name=camera,
            image_topic=camera_config["image_topic"],
            camera_info_topic=camera_config["camera_info_topic"],
            optical_frame=camera_config["optical_frame"],
            base_frame=f"{namespace}/base_link",
            result_topic=result_topic,
            image_out_topic=result_image_topic,
            tf_buffer=self.tf_buffer,
            gazebo=self.gazebo,
            objects=camera_objects,
            enabled_fn=lambda: self.enabled,
            compressed_image_output=True,
        )
        ## sim part end
        self._publish_enabled_status()
        get_node().get_logger().info(
            "Simulation tracker '%s' started with %d configured objects"
            % (camera, len(camera_objects))
        )

    def enable_callback(self, req, response):
        self.enabled = req.data
        self._publish_enabled_status()
        message = "Simulation inference %s" % (
            "enabled" if self.enabled else "disabled"
        )
        get_node().get_logger().info(message)
        return SetBool.Response(success=True, message=message)

    def _publish_enabled_status(self):
        self.enabled_pub.publish(Bool(data=self.enabled))


def main():
    SimTrackerNode()
    ros.spin(get_node())


if __name__ == "__main__":
    main()
