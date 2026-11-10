#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from auv_common_lib import ros
from auv_common_lib.ros import get_param
import tf2_ros
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from auv_msgs.srv import PlanPath
from auv_navigation.path_planning.path_planners import PathPlanners


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.path_planners = PathPlanners(self, self.tf_buffer)
        self.robot_frame = get_param(self, "~robot_frame", "taluy/base_link")

        # -- States --
        self.planning_active = False
        self.target_frame = None
        self.angle_offset = 0.0

        self.path_pub = self.create_publisher(Path, "/planned_path", 1)
        self.set_plan_service = self.create_service(
            PlanPath, "/set_plan", self.set_plan_cb
        )
        self.stop_planning_service = self.create_service(
            Trigger, "/stop_planning", self.stop_planning_cb
        )
        self.loop_rate = self.create_rate(get_param(self, "~loop_rate", 9))
        self.dynamic = True
        self.static_path = None
        self.get_logger().info("[path_planner_node] Path planner node started.")

    def set_plan_cb(self, req, response):
        self.target_frame = req.target_frame
        self.angle_offset = req.angle_offset
        self.dynamic = getattr(req, "dynamic", True)
        self.planning_active = False

        # If not dynamic, generate static path once
        if not self.dynamic:
            self.static_path = self.path_planners.straight_path_to_frame(
                source_frame=self.robot_frame,
                target_frame=self.target_frame,
                angle_offset=self.angle_offset,
            )
            if self.static_path:
                self.get_logger().info("[path_planner_node] Static path generated.")
            else:
                self.get_logger().warning(
                    "[path_planner_node] Static path generation failed."
                )
                response.success = False
                return response
        else:
            self.static_path = None

        self.planning_active = True
        self.get_logger().info("[path_planner_node] Planning activated.")

        self.get_logger().info(
            f"[path_planner_node] New plan set. Target: {self.target_frame}, Angle offset: {self.angle_offset}, dynamic: {self.dynamic}"
        )
        response.success = True
        return response

    def stop_planning_cb(self, req, response):
        self.planning_active = False
        self.get_logger().info("[path_planner_node] Planning deactivated.")
        response.success = True
        response.message = "Planning stopped."
        return response

    def run(self):
        while rclpy.ok():
            if self.planning_active and self.target_frame is not None:
                if self.dynamic:
                    path = None
                    try:
                        path = self.path_planners.straight_path_to_frame(
                            source_frame=self.robot_frame,
                            target_frame=self.target_frame,
                            angle_offset=self.angle_offset,
                        )
                        if path:
                            self.path_pub.publish(path)
                        else:
                            self.get_logger().warning(
                                "[path_planner_node] No path generated."
                            )
                    except Exception as e:
                        self.get_logger().error(
                            f"[path_planner_node] Error while planning path: {e}"
                        )
                else:
                    if self.static_path:
                        self.path_pub.publish(self.static_path)
            self.loop_rate.sleep()


def main():
    ros.init()
    node = PathPlannerNode()
    ros.spin(node, node.run)


if __name__ == "__main__":
    main()
