#!/usr/bin/env python3

"""Simulation-only tracker with the same ROS interface as tracker_node.py."""

import os
import sys

import rospy
import rospkg
import tf2_ros
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse

scripts_dir = os.path.join(rospkg.RosPack().get_path("auv_vision"), "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from sim_bbox_node import GazeboInterface, SimCamera, load_config


class SimTrackerNode:
    def __init__(self):
        rospy.init_node("sim_tracker_node")

        namespace = rospy.get_param("~namespace", "taluy")
        camera = rospy.get_param("~camera")
        default_config = os.path.join(
            rospkg.RosPack().get_path("auv_vision"),
            "config",
            "sim_bbox_objects.yaml",
        )
        config_path = rospy.get_param("~config", default_config)
        self.enabled = rospy.get_param("~enabled", True)

        objects, camera_configs = load_config(config_path, namespace)
        if camera not in camera_configs:
            raise rospy.ROSInitException(
                f"Unknown simulation camera '{camera}'; "
                f"available cameras: {sorted(camera_configs)}"
            )

        camera_config = camera_configs[camera]
        result_topic = rospy.get_param("~result_topic", camera_config["result_topic"])
        result_image_topic = rospy.get_param(
            "~result_image_topic", camera_config["image_out_topic"] + "/compressed"
        )

        self.enabled_pub = rospy.Publisher("~enabled", Bool, queue_size=1, latch=True)
        self.enable_srv = rospy.Service("~enable", SetBool, self.enable_callback)
        ##sim part
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
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
        rospy.loginfo(
            "Simulation tracker '%s' started with %d configured objects",
            camera,
            len(camera_objects),
        )

    def enable_callback(self, req):
        self.enabled = req.data
        self._publish_enabled_status()
        message = "Simulation inference %s" % (
            "enabled" if self.enabled else "disabled"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def _publish_enabled_status(self):
        self.enabled_pub.publish(Bool(data=self.enabled))


if __name__ == "__main__":
    try:
        SimTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
