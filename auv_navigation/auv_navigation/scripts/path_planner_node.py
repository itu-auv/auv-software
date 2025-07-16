#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from auv_msgs.srv import PlanPath, PlanPathResponse
from auv_navigation.path_planning.path_planners import PathPlanners


class PathPlannerNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.path_planners = PathPlanners(self.tf_buffer)
        self.robot_frame = rospy.get_param("~robot_frame", "taluy/base_link")

        # -- States --
        self.planning_active = False
        self.target_frame = None
        self.planner_type = None

        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)
        self.set_plan_service = rospy.Service("/set_plan", PlanPath, self.set_plan_cb)
        self.loop_rate = rospy.Rate(rospy.get_param("~loop_rate", 10))
        rospy.loginfo("[path_planner_node] Path planner node started.")

    def set_plan_cb(self, req):
        if not self.planning_active:
            self.planning_active = True
            rospy.loginfo("[path_planner_node] Planning activated.")

            self.planner_type = req.planner_type
            self.target_frame = req.target_frame
            rospy.loginfo(
                f"[path_planner_node] New plan set. Type: {self.planner_type}"
            )
        return PlanPathResponse(success=True)

    def run(self):
        while not rospy.is_shutdown():
            if (
                self.planning_active
                and self.target_pose is not None
                and self.planner_type is not None
            ):
                path = None
                try:
                    if self.planner_type == "gate":
                        path = self.path_planners.path_for_gate()
                    elif self.planner_type == "bin":
                        path = self.path_planners.path_for_bin()
                    elif (
                        self.planner_type == "straight"
                        and self.target_frame is not None
                    ):
                        path = self.path_planners.straight_path_to_frame(
                            source_frame=self.robot_frame,
                            target_frame=self.target_frame,
                        )
                    if path:
                        self.path_pub.publish(path)
                    else:
                        rospy.logwarn("[path_planner_node] No path generated.")

                except Exception as e:
                    rospy.logerr(f"[path_planner_node] Error while planning path: {e}")

            self.loop_rate.sleep()


if __name__ == "__main__":
    rospy.init_node("path_planner_node")
    node = PathPlannerNode()
    node.run()
