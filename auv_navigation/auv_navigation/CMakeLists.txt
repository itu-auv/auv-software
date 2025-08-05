cmake_minimum_required(VERSION 3.10.2)
project(auv_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  nav_msgs
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS rospy actionlib nav_msgs tf2_ros
)

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/follow_path_action_server.py
  scripts/path_planner_node.py
  scripts/lane_calculator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
