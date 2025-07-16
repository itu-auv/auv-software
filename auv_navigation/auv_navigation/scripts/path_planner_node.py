#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Path
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

        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)
        self.set_plan_service = rospy.Service("/set_plan", PlanPath, self.set_plan_cb)
        self.loop_rate = rospy.Rate(rospy.get_param("~loop_rate", 9))
        rospy.loginfo("[path_planner_node] Path planner node started.")

    def set_plan_cb(self, req):
        self.planning_active = True
        rospy.loginfo("[path_planner_node] Planning activated.")

        self.target_frame = req.target_frame
        rospy.loginfo(f"[path_planner_node] New plan set. Target: {self.target_frame}")
        return PlanPathResponse(success=True)

    def run(self):
        while not rospy.is_shutdown():
            if self.planning_active and self.target_frame is not None:
                path = None
                try:
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
