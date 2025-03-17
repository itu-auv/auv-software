cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

find_package(catkin REQUIRED COMPONENTS
  auv_detection
  rospy
  cv_bridge
  sensor_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    rospy
    cv_bridge
    sensor_msgs
    tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dynamic_roll_compensation.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)
