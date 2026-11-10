#!/usr/bin/env python3

import rclpy
from auv_common_lib.ros import service_client, ROSException, Rate
from auv_common_lib import ros
from rclpy.node import Node
from rclpy.exceptions import ROSInterruptException
from std_srvs.srv import Trigger, SetBool
from auv_msgs.srv import SetDetectionFocus


class SmachMonitor(Node):
    def __init__(self):
        super().__init__("smach_monitor_node")

        self.main_sm_node_name = "main_state_machine"
        self.teleop_node_name = "joystick_node"

        self.align_frame_service_name = "control/align_frame/cancel"
        self.detection_focus_service_name = "vision/set_front_camera_focus"

        # Camera detection disabling service names
        self.disable_segment_service_name = "vision/enable_segment_camera_detections"
        self.disable_bottom_service_name = "vision/enable_bottom_camera_detections"

        self.align_frame_service = service_client(
            self, Trigger, self.align_frame_service_name
        )
        self.detection_focus_service = service_client(
            self, SetDetectionFocus, self.detection_focus_service_name
        )

        # Service proxies for disabling detections
        self.disable_segment_service = service_client(
            self, SetBool, self.disable_segment_service_name
        )
        self.disable_bottom_service = service_client(
            self, SetBool, self.disable_bottom_service_name
        )

        self.teleop_shutdown_service = service_client(
            self, Trigger, self.teleop_node_name + "/shutdown"
        )

        self.smach_is_active = False
        self.rate = Rate(self, 1)  # 1hz

    def run(self):
        while rclpy.ok():
            if self.is_node_active(self.main_sm_node_name):
                if not self.smach_is_active:
                    self.get_logger().info("Main state machine has started.")
                    self.smach_is_active = True
                    if self.is_node_active(self.teleop_node_name):
                        self.get_logger().info("Teleop node is running. Killing it.")
                        # ROS 2 cannot kill a node by name; the node offers a service.
                        try:
                            self.teleop_shutdown_service(Trigger.Request())
                        except Exception as e:
                            self.get_logger().error(
                                "Could not shut the teleop node down: %s" % e
                            )
            else:
                if self.smach_is_active:
                    self.get_logger().info("Main state machine has died.")
                    self.smach_is_active = False
                    self.on_smach_death()

            self.rate.sleep()

    def is_node_active(self, node_name):
        # ROS 2 has no node ping; a node counts as active while it is in the graph.
        for node in self.get_node_names():
            if node.endswith(node_name):
                return True
        return False

    def on_smach_death(self):
        self.get_logger().info("Executing recovery actions.")

        # 1. Cancel align frame controller
        try:
            self.align_frame_service.wait_for_service(timeout=2.0)
            req = Trigger.Request()
            self.align_frame_service(req)
            self.get_logger().info("Cancelled align frame controller.")
        except (
            Exception,
            ROSException,
            ROSInterruptException,
        ) as e:
            self.get_logger().error(
                "Service call to cancel align frame controller failed: %s" % e
            )

        # 2. Set DetectionFocus to none
        try:
            self.detection_focus_service.wait_for_service(timeout=2.0)
            req = SetDetectionFocus.Request(focus_object="none")
            self.detection_focus_service(req)
            self.get_logger().info("Set DetectionFocus to 'none'.")
        except (
            Exception,
            ROSException,
            ROSInterruptException,
        ) as e:
            self.get_logger().error(
                "Service call to set detection focus failed: %s" % e
            )

        # 3. Disable other camera detections (segment, bottom)
        for service_name, service_proxy in [
            (self.disable_segment_service_name, self.disable_segment_service),
            (self.disable_bottom_service_name, self.disable_bottom_service),
        ]:
            try:
                service_proxy.wait_for_service(timeout=2.0)
                req = SetBool.Request(data=False)
                service_proxy(req)
                self.get_logger().info(
                    "Disabled detections for service: %s" % service_name
                )
            except (
                Exception,
                ROSException,
                ROSInterruptException,
            ) as e:
                self.get_logger().error(
                    "Service call to %s failed: %s" % (service_name, e)
                )


def main():
    ros.init()
    node = SmachMonitor()
    ros.spin(node, node.run)


if __name__ == "__main__":
    main()
