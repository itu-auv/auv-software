cmake_minimum_required(VERSION 3.0.2)
project(auv_depth_estimation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/depth_anything_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
