cmake_minimum_required(VERSION 3.10.2)
project(auv_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  nav_msgs
  tf2_ros
  dynamic_reconfigure
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/Slalom.cfg
)

catkin_package(
  CATKIN_DEPENDS rospy actionlib nav_msgs tf2_ros dynamic_reconfigure
)

catkin_install_python(PROGRAMS
  scripts/follow_path_action_server.py
  scripts/slalom_pipes_to_waypoints.py
  scripts/slalom_pipe_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
