cmake_minimum_required(VERSION 3.10.2)
project(auv_localization)

find_package(catkin REQUIRED COMPONENTS
  angles
  auv_msgs
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS angles auv_msgs geometry_msgs nav_msgs std_msgs tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/imu_odometry_node.py
  scripts/pressure_odometry_node.py
  scripts/dvl_odometry_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
