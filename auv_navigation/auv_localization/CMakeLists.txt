cmake_minimum_required(VERSION 3.10.2)
project(auv_localization)

find_package(catkin REQUIRED COMPONENTS
  auv_msgs
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  std_msgs
  tf
)

add_service_files(
  FILES
  CalibrateIMU.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS auv_msgs geometry_msgs message_runtime nav_msgs std_msgs tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/euler_orientation_publisher.py
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
